# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import torch.optim as optim
import collections
import torch.cuda.amp as amp
import random
import torchvision.utils as vutils
from PIL import Image

from utils.utils import lr_cosine_policy, clip, create_folder
import wandb
import matplotlib.pyplot as plt


import random
from torchvision import transforms
import torch.nn as nn
import numpy as np


class CIFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, i.g. KL divergence
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(
            module.running_mean.data - mean, 2)

        self.r_feature = r_feature
        # must have no outputs

    def close(self):
        self.hook.remove()





def get_image_prior_losses(inputs_jit):
    # COMPUTE total variation regularization loss
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (
            diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2


def save_images(inputs, targets, variance, prefix, base_iteration, save_every, local_rank):
    print("saving image dir", prefix)
    vutils.save_image(inputs, '{}/best_images/output_{:05d}_gpu_{}_first.png'.format(prefix,
                                                                                     (base_iteration) // save_every,
                                                                                     local_rank),
                      normalize=True, scale_each=True, nrow=int(10))
    plt.style.use('dark_background')
    image = plt.imread('{}/best_images/output_{:05d}_gpu_{}_first.png'.format(prefix,
                                                                              (base_iteration) // save_every,
                                                                              local_rank))
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.axis('off')
    fig.set_size_inches(10 * 3, int((len(inputs) + 1) / 10) * 3 + 2)
    plt.title("variance = " + str(variance) + "\n" + str(targets), fontweight="bold")
    plt.savefig('{}/best_images/output_{:05d}_gpu_{}_first.png'.format(prefix,
                                                                       (base_iteration) // save_every,
                                                                       local_rank))


class ImpressionClass(object):
    def __init__(self, bs=84,
                 use_fp16=True, net_teacher=None, path="./gen_images/",
                 final_data_path="/gen_images_final/",
                 parameters=dict(),
                 setting_id=0,
                 jitter=30,
                 criterion=None,
                 coefficients=dict(),
                 network_output_function=lambda x: x,
                 hook_for_display=None,
                 hook_for_self_eval=None,
                 device=None,
                 target_classes_min=0,
                 target_classes_max=0,
                 mean_image_dir="./saved_Sample",
                 cm=None,
                 alpha=None,
                 gamma=None,
                 data='BloodMnist',
                 look_back=False,
                 synthesis=True,
                 order_mine=None):
        '''
        :param bs: batch size per GPU for image generation
        :param use_fp16: use FP16 (or APEX AMP) for model inversion, uses less memory and is faster for GPUs with Tensor Cores
        :parameter net_teacher: Pytorch model to be inverted
        :param path: path where to write temporal images and data
        :param final_data_path: path to write final images into
        :param parameters: a dictionary of control parameters:
            "resolution": input image resolution, single value, assumed to be a square, 224
            "random_label" : for classification initialize target to be random values
            "start_noise" : start from noise, def True, other options are not supported at this time
            "detach_student": if computing Adaptive DI, should we detach student?
        :param setting_id: predefined settings for optimization:
            0 - will run low resolution optimization for 1k and then full resolution for 1k;
            1 - will run optimization on high resolution for 2k
            2 - will run optimization on high resolution for 20k

        :param jitter: amount of random shift applied to image at every iteration
        :param coefficients: dictionary with parameters and coefficients for optimization.
            keys:
            "r_feature" - coefficient for feature distribution regularization
            "tv_l1" - coefficient for total variation L1 loss
            "tv_l2" - coefficient for total variation L2 loss
            "l2" - l2 penalization weight
            "lr" - learning rate for optimization
            "main_loss_multiplier" - coefficient for the main loss optimization
        network_output_function: function to be applied to the output of the network to get the output
        hook_for_display: function to be executed at every print/save call, useful to check accuracy of verifier
        '''

        print("Class Impression generation")
        # for reproducibility
        torch.manual_seed(torch.cuda.current_device())

        self.net_teacher = net_teacher

        if "resolution" in parameters.keys():
            self.image_resolution = parameters["resolution"]
            self.random_label = parameters["random_label"]
            self.start_noise = parameters["start_noise"]
            self.detach_student = parameters["detach_student"]
            self.do_flip = parameters["do_flip"]
            self.store_best_images = parameters["store_best_images"]
        else:
            self.image_resolution = 224
            self.random_label = False
            self.start_noise = True
            self.detach_student = False
            self.do_flip = True
            self.store_best_images = False

        self.setting_id = setting_id
        self.bs = bs  # batch size
        self.use_fp16 = use_fp16
        self.save_every = 4000
        self.jitter = jitter
        self.criterion = criterion
        self.network_output_function = network_output_function

        if "r_feature" in coefficients:
            self.bn_reg_scale = coefficients["r_feature"]
            self.first_bn_multiplier = coefficients["first_bn_multiplier"]
            self.var_scale_l1 = coefficients["tv_l1"]
            self.var_scale_l2 = coefficients["tv_l2"]
            self.l2_scale = coefficients["l2"]
            self.lr = coefficients["lr"]
            self.main_loss_multiplier = coefficients["main_loss_multiplier"]
        else:
            print("Provide a dictionary with ")

        self.num_generations = 0
        self.final_data_path = final_data_path

        ## Create folders for images and logs
        prefix = path
        self.prefix = prefix
        local_rank = torch.cuda.current_device()
        if local_rank == 0:
            create_folder(prefix)
            create_folder(prefix + "/best_images/")
            if self.store_best_images:
                create_folder(self.final_data_path)

        self.base_iteration = 0

        ## Create hooks for feature statistics
        self.loss_r_feature_layers = []

        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(CIFeatureHook(module))

        self.hook_for_display = None
        if hook_for_display is not None:
            self.hook_for_display = hook_for_display

        self.hook_for_self_eval = None
        if hook_for_self_eval is not None:
            self.hook_for_self_eval = hook_for_self_eval

        self.device = device
        self.target_classes_min = target_classes_min
        self.target_classes_max = target_classes_max
        self.mean_image_dir = mean_image_dir

        self.cm = cm
        self.alpha = alpha
        self.gamma = gamma
        self.data = data
        self.look_back = look_back
        self.synthesis = synthesis
        self.order_mine = order_mine

    def get_images(self, net_student=None, targets=None, use_mean_initialization=False, beta_2=0.9):
        print("get_images call")

        net_teacher = self.net_teacher
        use_fp16 = self.use_fp16
        save_every = self.save_every

        kl_loss = nn.KLDivLoss(reduction='batchmean').to(self.device)
        local_rank = torch.cuda.current_device()
        best_cost = 1e4
        criterion = self.criterion

        # setup target labels
        if targets is None:
            # only works for classification now, for other tasks need to provide target vector
            targets = torch.LongTensor(
                [random.randint(self.target_classes_min, self.target_classes_max) for _ in range(self.bs)]).to(
                self.device)
            if not self.random_label:
                targets = [i for i in np.arange(self.target_classes_min, self.target_classes_max + 1)]
                targets = torch.LongTensor(targets * (int(self.bs / len(targets)) + 1))[0:self.bs].to(self.device)

        if self.look_back:
            # print(targets,self.cm[0]/np.sum(self.cm[0]),self.cm[1]/np.sum(self.cm[1]),self.cm[2]/np.sum(self.cm[2]))
            targets_prob = torch.zeros((self.bs, len(self.cm))).to(self.device)
            for i, t in enumerate(targets):
                dirichlet = np.random.dirichlet(self.alpha, size=None)
                while np.sum(np.abs(dirichlet - self.cm[t] / np.sum(self.cm[t]))) > self.gamma:
                    dirichlet = np.random.dirichlet(self.alpha, size=None)
                targets_prob[i] = torch.tensor(dirichlet, device=self.device).float()

        img_original = self.image_resolution

        variance = 1

        if self.data == 'BloodMnist' or self.data == 'PathMnist':
            inputs_layer = torch.from_numpy(np.random.normal(0, variance
                                                             , (self.bs, 3, img_original, img_original))).type(
                torch.FloatTensor).to(self.device)
        elif self.data == 'TissueMnist' or self.data == 'OrganaMnist':
            inputs_layer = torch.from_numpy(np.random.normal(0, variance
                                                             , (self.bs, 1, img_original, img_original))).type(
                torch.FloatTensor).to(self.device)
        else:
            inputs_layer = torch.from_numpy(np.random.normal(0, variance
                                                             , (self.bs, 1, img_original, img_original))).type(
                torch.FloatTensor).to(self.device)
        inputs_layer.requires_grad = False

        if self.data == 'BloodMnist' or self.data == 'PathMnist' or self.data == 'TissueMnist' or self.data == 'OrganaMnist':
            mean = [0, 0, 0]
            std = [1, 1, 1]
        else:
            mean = [0.122, 0.122, 0.122]
            std = [0.184, 0.184, 0.184]

        if use_mean_initialization:
            for t in range(len(targets)):
                initialized_image_dir = self.mean_image_dir + "/label_" + str(
                    self.order_mine[targets[t].item()]) + "_integrated.png"
                image = Image.open(initialized_image_dir)
                convert_tensor = transforms.ToTensor()
                image_array = convert_tensor(np.divide(((np.array(image) / 255.0) - mean), std)).to(self.device)
                if self.synthesis:
                    if self.data == 'BloodMnist' or self.data == 'PathMnist':
                        inputs_layer[t] = inputs_layer[t] / 10 + torch.reshape(image_array, (
                            3, self.image_resolution, self.image_resolution))
                    elif self.data == 'TissueMnist' or self.data == 'OrganAMnist':
                        inputs_layer[t] = inputs_layer[t] / 10 + torch.reshape(image_array[0, :, :], (
                            1, self.image_resolution, self.image_resolution))
                    else:
                        inputs_layer[t] = inputs_layer[t] / 10 + torch.reshape(image_array[:, :, 0], (
                            1, self.image_resolution, self.image_resolution))
                else:
                    if self.data == 'BloodMnist' or self.data == 'PathMnist':
                        inputs_layer[t] = torch.reshape(image_array, (
                            3, self.image_resolution, self.image_resolution))
                    elif self.data == 'TissueMnist' or self.data == 'OrganAMnist':
                        inputs_layer[t] = torch.reshape(image_array[0, :, :], (
                            1, self.image_resolution, self.image_resolution))
                    else:
                        inputs_layer[t] = torch.reshape(image_array[:, :, 0], (
                            1, self.image_resolution, self.image_resolution))

        inputs_layer.requires_grad = True
        inputs = inputs_layer
        save_images(inputs_layer, targets, variance, self.prefix, self.base_iteration, save_every, local_rank)
        if self.setting_id == 0:
            skipfirst = False
        else:
            skipfirst = True
        print(self.setting_id)
        iteration = 0
        if self.synthesis:
            for lr_it, lower_res in enumerate([2, 1]):
                if lr_it == 0:
                    iterations_per_layer = 3000
                else:
                    iterations_per_layer = 1000 if not skipfirst else 5000
                    if self.setting_id == 2:
                        iterations_per_layer = 20000

                if lr_it == 0 and skipfirst:
                    continue

                lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res

                if self.setting_id == 0:
                    # multi resolution, 2k iterations with low resolution, 1k at normal, ResNet50v1.5 works the best, ResNet50 is ok
                    optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, beta_2], eps=1e-8)
                    do_clip = True
                elif self.setting_id == 1:
                    # 2k normal resolultion, for ResNet50v1.5; Resnet50 works as well
                    optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, beta_2], eps=1e-8)
                    do_clip = True
                elif self.setting_id == 2:
                    # 20k normal resolution the closes to the paper experiments for ResNet50
                    optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, beta_2], eps=1e-8)
                    do_clip = True

                if use_fp16:
                    static_loss_scale = 256
                    static_loss_scale = "dynamic"
                    _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)

                lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)

                for iteration_loc in range(iterations_per_layer):

                    iteration += 1
                    # learning rate scheduling
                    lr = lr_scheduler(optimizer, iteration_loc, iteration_loc)
                    inputs_jit = inputs

                    # apply random jitter offsets
                    off1 = random.randint(-lim_0, lim_0)
                    off2 = random.randint(-lim_1, lim_1)
                    inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2, 3))

                    # Flipping
                    flip = random.random() > 0.5
                    if flip and self.do_flip:
                        inputs_jit = torch.flip(inputs_jit, dims=(3,))

                    # forward pass
                    optimizer.zero_grad()
                    net_teacher.zero_grad()
                    outputs = net_teacher(inputs_jit)
                    outputs = self.network_output_function(outputs)

                    # R_cross classification loss
                    if self.look_back:
                        loss = kl_loss(outputs, targets_prob)
                    else:
                        loss = criterion(outputs, targets)

                    # R_prior losses
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
                    # R_feature loss
                    rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers) - 1)]

                    loss_r_feature = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])

                    # l2 loss on images
                    loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()

                    # combining losses
                    loss_aux = self.var_scale_l2 * loss_var_l2 + \
                               self.var_scale_l1 * loss_var_l1 + \
                               self.bn_reg_scale * loss_r_feature + \
                               self.l2_scale * loss_l2

                    loss = self.main_loss_multiplier * loss + loss_aux

                    if local_rank == 0:
                        ce = criterion(outputs, targets).item()
                        if iteration % save_every == 0:
                            print("------------iteration {}----------".format(iteration))
                            print("total loss", loss.item())
                            print("loss_r_feature", loss_r_feature.item())
                            print("main criterion", ce)

                            if self.hook_for_display is not None:
                                acc = self.hook_for_display(inputs, targets)
                            else:
                                acc = 0

                            if self.hook_for_self_eval is not None:
                                acc_self = self.hook_for_self_eval(inputs, targets)
                            else:
                                acc_self = 0

                            metrics = {"total loss": loss.item(),
                                       "loss batch normalization": self.bn_reg_scale * loss_r_feature.item(),
                                       "batch normalization value": loss_r_feature.item(),
                                       "loss variation_l2": self.var_scale_l2 * loss_var_l2.item(),
                                       "loss l2 on images": self.l2_scale * loss_l2.item(),
                                       "Cross Entropy": self.main_loss_multiplier * ce,
                                       "Verifier Acc": acc,
                                       "Self Acc": acc_self,
                                       "learning rate": lr}
                            wandb.log(metrics)

                    # do image update
                    if use_fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()

                    optimizer.step()

                    # clip color outlayers
                    if do_clip:
                        inputs.data = clip(inputs.data, use_fp16=use_fp16)

                    if best_cost > loss.item() or iteration == 1:
                        best_inputs = inputs.data.clone()
                        best_cost = loss.item()

                    if iteration % save_every == 0 and (save_every > 0):
                        if local_rank == 0:
                            save_images(inputs, targets, variance, self.prefix, self.base_iteration, save_every,
                                        local_rank)

            optimizer.state = collections.defaultdict(dict)
            acc_self = self.hook_for_self_eval(inputs, targets)

        else:
            best_inputs = inputs.data.clone()
        if self.store_best_images:
            save_images(best_inputs, targets, variance, self.prefix, self.base_iteration, save_every,
                        local_rank)

        # to reduce memory consumption by states of the optimizer we deallocate memory
        self.base_iteration += iteration
        print("iteratiooooooooon ======================", iteration)
        return best_inputs, targets

    def generate_batch(self, net_student=None, targets=None, use_mean_initialization=False, beta_2=0.9):

        use_fp16 = self.use_fp16

        # fix net_student
        if not (net_student is None):
            net_student = net_student.eval()

        if targets is not None:
            targets = torch.from_numpy(np.array(targets).squeeze()).to(self.device)
            if use_fp16:
                targets = targets.half()

        self.net_teacher.eval()

        images, targets = self.get_images(net_student=net_student, targets=targets,
                                          use_mean_initialization=use_mean_initialization, beta_2=beta_2)

        self.num_generations += 1
        return images.cpu(), targets.cpu()
