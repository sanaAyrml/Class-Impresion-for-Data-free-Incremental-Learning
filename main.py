#!/usr/bin/env python
# coding=utf-8


from __future__ import division, print_function
from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import re

try:
    import cPickle as pickle
except:
    import pickle
import math

import utils_pytorch
from incremental_train_and_eval import incremental_train_and_eval
from models.layers import modified_linear

import torch.nn.parallel
import torch.utils.data

from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataloader.heart_loader import get_heart_dataset

from utils.compute_confusion_matrix import compute_confusion_matrix
from utils.utils import compute_mean_images
from data_synthesis.continual_class_specific_impression import ImpressionClass

import wandb
import matplotlib.pyplot as plt
from utils.visualize import visualize, fit_umap
from utils.compute_features import compute_features
from utils.compute_accuracy import compute_accuracy
import random
import argparse
import os
import torchvision.utils as vutils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
import numpy as np
import copy


def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        model.eval()
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 1))

    print("model accuracy: ", prec1.item())
    return prec1.item()


######### Modifiable Settings ##########
parser = argparse.ArgumentParser()

##### Log and Checkpoint variables
parser.add_argument('--project_name', default='continual_learning_new', type=str,
                    help='project name in wandb')
parser.add_argument('--wandb_acc', default='your_acc', type=str,
                    help='account name in wandb')
parser.add_argument('--wandb_key', default='your_key', type=str,
                    help='account key in wandb')
parser.add_argument('--main_directory', default='./medical_checkpoint/', type=str, \
                    help='Checkpoint directory')
parser.add_argument('--ckp_prefix', default='', type=str, \
                    help='Checkpoint prefix')
parser.add_argument('--saved_model_address', default=" ", type=str,
                    help='address of saved model')

##### Run type Variables
parser.add_argument('--random_seed', default=2022, type=int, \
                    help='random seed')
parser.add_argument('--mode', default='CCSI', type=str,
                    help='use which sampler strategy')
parser.add_argument('--cuda_number', default=0, type=int,
                    help='which cuda use to train')
parser.add_argument('--validate', default=False, type=bool,
                    help='run the validate part of network')
parser.add_argument('--resume', default=False, type=bool, \
                    help='resume from checkpoint')
parser.add_argument('--store_best_images', default=False, type=bool,
                    help='save best images as separate files')
parser.add_argument('--compute_mean', default=False, type=bool,
                    help='save some samples')
parser.add_argument('--mean_images_dir', type=str, default='./saved_Sample',
                    help='place to save samples')
parser.add_argument('--visualize', default=False, type=bool, \
                    help='visualize umap of generated data and original data')
parser.add_argument('--load_dricet_mode', default=False, type=bool,
                    help='load model weights directly')

##### Data Spilit variables and data features
parser.add_argument('--num_classes', default=14, type=int)
parser.add_argument('--nb_cl_fg', default=2, type=int, \
                    help='the number of classes in first group')
parser.add_argument('--nb_cl', default=1, type=int, \
                    help='Classes per group')
parser.add_argument('--nb_phases', default=2, type=int, \
                    help='the number of phases')
parser.add_argument('--nb_protos', default=20, type=int, \
                    help='Number of prototypes per class at the end')
parser.add_argument('--nb_runs', default=1, type=int, \
                    help='Number of configs (random ordering of classes at each run)')
parser.add_argument('--start_generate_phase', default=0, type=int,
                    help='start generate phase')

##### Data type and features
parser.add_argument('--input_dim', type=int, default=1,
                    help='input dimension')
parser.add_argument('--data', default='None', type=str,
                    help='data Type')
parser.add_argument('--image_size', default=224, type=int,
                    help='Image size')
parser.add_argument('--download_data',  default=False, type=bool,
                    help='Enable download data')

##### Model Variables
parser.add_argument('--cosine_normalization', default=False, type=bool,
                    help='change last layer of networks')
parser.add_argument('--small_model', default=False, type=bool,
                    help='use model with 3 layers')
parser.add_argument('--enable_drop_out', default=False, type=bool,
                    help='if model got activated dropout in it')
parser.add_argument('--continual_norm', default=False, type=bool,
                    help='if model has continual norm instead of batch norm')
parser.add_argument('--gn_size', default=4, type=int,
                    help='size of group norm')

##### Synthesis variable
parser.add_argument('--epochs_generat', default=4000, type=int,
                    help='number of epochs')
parser.add_argument('--generation_lr', type=float, default=0.2,
                    help='learning rate for optimization')
parser.add_argument('--setting_id', default=1, type=int,
                    help='settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations')
parser.add_argument('--bs', default=64, type=int,
                    help='batch size for generation')
parser.add_argument('--jitter', default=30, type=int,
                    help='batch size')
parser.add_argument('--fp16', default=False, type=bool,
                    help='use FP16 for optimization')
parser.add_argument('--do_flip', default=False, type=bool,
                    help='apply flip during model inversion')
parser.add_argument('--resolution', default=224, type=int,
                    help='resolution of generated images')
parser.add_argument('--random_label', default=False, type=bool,
                    help='generate random label for optimization')
parser.add_argument('--r_feature', type=float, default=0.05,
                    help='coefficient for feature distribution regularization')
parser.add_argument('--first_bn_multiplier', type=float, default=10.,
                    help='additional multiplier on first bn layer of R_feature')
parser.add_argument('--tv_l1', type=float, default=0.0,
                    help='coefficient for total variation L1 loss')
parser.add_argument('--tv_l2', type=float, default=0.0001,
                    help='coefficient for total variation L2 loss')
parser.add_argument('--l2', type=float, default=0.00001,
                    help='l2 loss on the image')
parser.add_argument('--main_loss_multiplier', type=float, default=1.0,
                    help='coefficient for the main loss in optimization')
parser.add_argument('--use_mean_initialization', default=False, type=bool,
                    help='use mean of classes to initialize vectors')

##### Sampler Variables
parser.add_argument('--add_sampler', default=False, type=bool,
                    help='enable deep inversion part')
parser.add_argument('--add_data', default=False, type=bool,
                    help='enable deep to add generated data')
parser.add_argument('--nb_generation', default=0, type=int,
                    help='number of batch to train')
parser.add_argument('--look_back', default=False, type=bool,
                    help='Enable look back')
parser.add_argument('--not_synthesis', default=False, type=bool,
                    help='Enable not synthesis')
parser.add_argument('--generate_more', default=False, type=bool,
                    help='generate more batches of data')

##### Training variabes
parser.add_argument('--epochs', default=100, type=int, \
                    help='Epochs')
parser.add_argument('--batch_size_1', type=int, default=64,
                    help='batch size for training model')
parser.add_argument('--lr', type=float, default=0.2,
                    help='learning rate for optimization')
parser.add_argument('--rs_ratio', default=0, type=float, \
                    help='The ratio for resample')
parser.add_argument('--beta_2', default=0.9, type=float,
                    help='beta 2 for adam optimizer in generating')

### knowledge transfer between two tasks
parser.add_argument('--imprint_weights', default=False, type=bool, \
                    help='Imprint the weights for novel classes')
parser.add_argument('--less_forget', default=False, type=bool, \
                    help='Less forgetful')
parser.add_argument('--lamda', default=5, type=float, \
                    help='Lamda for LF')
parser.add_argument('--adapt_lamda', default=False, type=bool, \
                    help='Adaptively change lamda')

### Distilatillation
parser.add_argument('--T', default=2, type=float, \
                    help='Temporature for distialltion')
parser.add_argument('--beta', default=0.25, type=float, \
                    help='Beta for distialltion')

### Margin ranking
parser.add_argument('--alpha_3', default=1, type=float,
                    help='Margin ranking loss coeficient')
parser.add_argument('--mr_loss', default=False, type=bool, \
                    help='Margin ranking loss v1')
parser.add_argument('--amr_loss', default=False, type=bool, \
                    help='Margin ranking loss v2')
parser.add_argument('--dist', default=0.5, type=float, \
                    help='Dist for MarginRankingLoss')
parser.add_argument('--K', default=1, type=int, \
                    help='K for MarginRankingLoss')
parser.add_argument('--lw_mr', default=1, type=float, \
                    help='loss weight for margin ranking loss')

### domain adaptation contrastive loss
parser.add_argument('--da_coef', default=1, type=float,
                    help='domain adoption coeficient')
parser.add_argument('--ro', default=0.9, type=float,
                    help='ro for updating centroids')
parser.add_argument('--temprature', default=5, type=float,
                    help='temprature for contrastive loss')

##### ?????
parser.add_argument('--mimic_score', default=False, type=bool, \
                    help='To mimic scores for cosine embedding')
parser.add_argument('--lw_ms', default=1, type=float, \
                    help='loss weight for mimicking score')

#####################################################################################################

args = parser.parse_args()
os.environ["WANDB_API_KEY"] = args.wandb_key
wandb.init(project=args.project_name, entity=args.wandb_acc, config=args)

if args.small_model:
    if args.cosine_normalization:
        from models.Medical_predictor_model_3_layers_modified import ResNet, ResidualBlock
    else:
        from models.Medical_predictor_model_3_layers import ResNet, ResidualBlock
else:
    if args.cosine_normalization:
        from models.Medical_predictor_model_modified import ResNet, ResidualBlock
    else:
        from models.Medical_predictor_model import ResNet, ResidualBlock

########################################
train_batch_size = args.batch_size_1  # Batch size for train
test_batch_size = args.batch_size_1  # Batch size for test
eval_batch_size = args.batch_size_1  # Batch size for eval
base_lr = args.lr  # Initial learning rate
lr_factor = 0.3  # Learning rate decrease factor
lr_patience = 5
lr_threshold = 0.0001
custom_weight_decay = 0  # Weight Decay
custom_momentum = 0  # Momentum

if not os.path.exists(args.main_directory):
    os.makedirs(args.main_directory)

if not os.path.exists(args.main_directory + '/' + args.mode):
    os.makedirs(args.main_directory + '/' + args.mode)

main_ckp_prefix = '{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.ckp_prefix,
                                                               args.nb_cl_fg,
                                                               args.nb_cl,
                                                               args.lr,
                                                               args.batch_size_1)

np.random.seed(args.random_seed)  # Fix the random seed
random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
print(args)

if args.data == 'PathMnist':
    from dataloader.medmnist_loader_pathmnist import get_medmnist_dataset

    sub_order = [0, 1, 2, 3, 4, 5, 6, 7, 8]
elif args.data == 'OrganAMnist':
    from dataloader.medmnist_loader_organamnist import get_medmnist_dataset

    sub_order = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
elif args.data == 'TissueMnist':
    from dataloader.medmnist_loader_tissuemnist import get_medmnist_dataset

    sub_order = [0, 1, 2, 3, 4, 5, 6, 7]
elif args.data == 'BloodMnist':
    from dataloader.medmnist_loader_bloodmnist import get_medmnist_dataset

    sub_order = [0, 1, 2, 3, 4, 5, 6, 7]
else:
    sub_order = [0, 1, 2, 8, 3]

########################################
print("cuda:" + str(args.cuda_number))
device = torch.device("cuda:" + str(args.cuda_number) if torch.cuda.is_available() else "cpu")
if args.data != 'None':
    trainset = get_medmnist_dataset(mode='train', image_size=args.image_size)
    evalset = get_medmnist_dataset(mode='test', image_size=args.image_size)
    testset = get_medmnist_dataset(mode='test', image_size=args.image_size)

else:
    trainset = get_heart_dataset(mode='train', used_labels=None, order=sub_order)
    evalset = get_heart_dataset(mode='valid', used_labels=None, order=sub_order)
    testset = get_heart_dataset(mode='valid', used_labels=None, order=sub_order)

# Initialization
top1_acc_list_cumul = np.zeros((int(args.num_classes / args.nb_cl), 3, args.nb_runs))
top1_acc_list_ori = np.zeros((int(args.num_classes / args.nb_cl), 3, args.nb_runs))

X_train_total = np.array(trainset.data)
Y_train_total = np.array(trainset.targets)
X_valid_total = np.array(testset.data)
Y_valid_total = np.array(testset.targets)
curupted_last_task = False

# Launch the different configs
for iteration_total in range(args.nb_runs):
    # Select the order for the class learning
    order_name = args.main_directory + "/seed_{}_rder_run_{}.pkl".format(args.random_seed, iteration_total)
    print("Order name:{}".format(order_name))
    if os.path.exists(order_name):
        print("Loading orders")
        order = utils_pytorch.unpickle(order_name)
    else:
        print("Generating orders")
        order = np.arange(args.num_classes)
        utils_pytorch.savepickle(order, order_name)

    order_list = list(order)

    # Initialization of the variables for this run
    dictionary_size = 60000
    X_valid_cumuls = []
    X_protoset_cumuls = []
    X_train_cumuls = []
    Y_valid_cumuls = []
    Y_protoset_cumuls = []
    Y_train_cumuls = []

    prototypes = [[] for i in range(args.num_classes)]
    for orde in range(args.num_classes):
        prototypes[orde] = X_train_total[np.where(Y_train_total == order[orde])]
    prototypes = np.array(prototypes, dtype=object, )

    start_iter = int(args.nb_cl_fg / args.nb_cl) - 1
    extra_fg = args.nb_cl_fg % args.nb_cl
    last_iter = int(args.num_classes / args.nb_cl)

    if args.compute_mean:
        print('Computing means:')
        compute_mean_images(trainset, args)

    for iteration in range(start_iter, min(args.nb_phases + start_iter, int(args.num_classes / args.nb_cl) + 1)):
        print('iteration', iteration)
        if iteration == max(args.nb_phases + start_iter, int(args.num_classes / args.nb_cl) + 1) - 1:
            curupted_last_task = True

        # Rename Checkpoint

        main_ckp_prefix = '{}/{}_nb_cl_fg_{}_nb_cl_{}_lr_{}_bs_{}'.format(args.mode, args.ckp_prefix,
                                                                          args.nb_cl_fg,
                                                                          args.nb_cl,
                                                                          args.lr,
                                                                          args.batch_size_1)
        print("main_ckp_prefix", main_ckp_prefix)
        wandb.run.name = '{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
        wandb.run.save()

        if iteration > start_iter:

            main_ckp_prefix = main_ckp_prefix + '_bsg_' + str(args.bs) + '_lrg_' + str(
                args.generation_lr) + '_rfg_' + str(args.r_feature) + '_tv_l2g_' + str(args.tv_l2) + '_l2g_' + str(
                args.l2) + '_beta2_' + str(args.beta_2) + '_alpha3_' + str(args.alpha_3) + '_dist_' + str(
                args.dist) + '_mlm_' + str(args.main_loss_multiplier)
            if args.da_coef != 0:
                main_ckp_prefix = main_ckp_prefix + '_ro_' + str(args.ro) + '_temprature_' + str(args.temprature)

            wandb.run.name = '{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total, iteration)
            wandb.run.save()

        # init model
        if iteration == start_iter:
            ############################################################
            last_iter = 0
            ############################################################
            print("making original model")
            if args.small_model:
                print("small resnet original model with layers 1 1 1 ")
                tg_model = ResNet(ResidualBlock, [1, 1, 1], input_dim=args.input_dim,
                                  num_classes=(iteration - start_iter) * args.nb_cl + args.nb_cl_fg).to(device)
            else:
                print("resnet original model with layers 2 2 2 2 ")
                tg_model = ResNet(ResidualBlock, [2, 2, 2, 2], input_dim=args.input_dim,
                                  num_classes=(iteration - start_iter) * args.nb_cl + args.nb_cl_fg).to(device)
            ref_model = None
            new_feature = args.nb_cl
            
            print("============================= Here is the model ============================")
        elif iteration == start_iter + 1:
            ############################################################
            last_iter = iteration
            ############################################################
            # incerement classes
            if not args.continual_norm:
                ref_model = copy.deepcopy(tg_model)

            in_features = tg_model.fc.in_features
            out_features = tg_model.fc.out_features
            print("in_features:", in_features, "out_features:", out_features)
            new_feature = args.nb_cl

            if args.cosine_normalization:
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features, new_feature)
                new_fc.sigma.data = tg_model.fc.sigma.data
            else:
                new_fc = modified_linear.SplitLinear(in_features, out_features, new_feature)

            new_fc.fc1.weight.data = tg_model.fc.weight.data
            tg_model.fc = new_fc
            lamda_mult = out_features * 1.0 / args.nb_cl
        else:
            ############################################################
            last_iter = iteration
            ############################################################
            if not args.continual_norm:
                ref_model = copy.deepcopy(tg_model)
            
            in_features = tg_model.fc.in_features
            out_features1 = tg_model.fc.fc1.out_features
            out_features2 = tg_model.fc.fc2.out_features
            print("in_features:", in_features, "out_features1:", \
                  out_features1, "out_features2:", out_features2)
            if curupted_last_task:
                new_feature = (args.num_classes - args.nb_cl_fg) % args.nb_cl
                if new_feature == 0:
                    new_feature = args.nb_cl
            else:
                new_feature = args.nb_cl
            print('new_feature===>', new_feature)
            if args.cosine_normalization:
                new_fc = modified_linear.SplitCosineLinear(in_features, out_features1 + out_features2, new_feature)
                new_fc.sigma.data = tg_model.fc.sigma.data
            else:
                new_fc = modified_linear.SplitLinear(in_features, out_features1 + out_features2, new_feature)
            new_fc.fc1.weight.data[:out_features1] = tg_model.fc.fc1.weight.data
            new_fc.fc1.weight.data[out_features1:] = tg_model.fc.fc2.weight.data
            tg_model.fc = new_fc
            lamda_mult = (out_features1 + out_features2) * 1.0 / (args.nb_cl)

        if iteration > start_iter and args.less_forget and args.adapt_lamda:
            cur_lamda = args.lamda * math.sqrt(lamda_mult)
        else:
            cur_lamda = args.lamda
        if iteration > start_iter and args.less_forget:
            print("###############################")
            print("Lamda for less forget is set to ", cur_lamda)
            print("###############################")

        if iteration == start_iter:
            st = 0
        else:
            st = last_iter * args.nb_cl + extra_fg

        # Prepare the training data for the current batch of classes
        lt = iteration * args.nb_cl + new_feature + extra_fg
        actual_cl = order[range(st, lt)]
        print("classes to be trained:", st, "-", lt)

        indices_train_10 = np.array([i in order[range(st, lt)] for i in Y_train_total])
        indices_test_10 = np.array([i in order[range(st, lt)] for i in Y_valid_total])

        X_train = X_train_total[indices_train_10]
        X_valid = X_valid_total[indices_test_10]
        print("len data to be trained ==> train:", len(X_train), "  validation:", len(X_valid))
        X_valid_cumuls.append(X_valid)
        X_train_cumuls.append(X_train)
        X_valid_cumul = np.concatenate(X_valid_cumuls)
        X_train_cumul = np.concatenate(X_train_cumuls)
        print("len total data seen till this phase ==> train:", len(X_train_cumul), "  validation:", len(X_valid_cumul))

        Y_train = Y_train_total[indices_train_10]
        Y_valid = Y_valid_total[indices_test_10]
        Y_valid_cumuls.append(Y_valid)
        Y_train_cumuls.append(Y_train)
        Y_valid_cumul = np.concatenate(Y_valid_cumuls)
        Y_train_cumul = np.concatenate(Y_train_cumuls)

        # Add the stored exemplars to the training data
        scale_factor = 0

        # Launch the training loop
        print('Batch of classes number {0} arrives ...'.format(iteration + 1))
        map_Y_train = np.array([order_list.index(i) for i in Y_train])
        map_Y_valid = np.array([order_list.index(i) for i in Y_valid])
        map_Y_train_cumul = np.array([order_list.index(i) for i in Y_train_cumul])
        map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
        print("making original dataloader")
        trainset.data = X_train
        trainset.targets = map_Y_train
        ori_sample_weights = np.ones((len(map_Y_train)))

        if iteration == start_iter:
            X_valid_ori = X_valid
            Y_valid_ori = map_Y_valid
            rs_sample_weights = ori_sample_weights
            rs_num_samples = len(X_train)
        else:
            if (args.add_data and iteration > args.start_generate_phase):
                print("Add protoset data with size: ", len(Y_protoset_cumuls))
                X_protoset = np.concatenate(X_protoset_cumuls)
                Y_protoset = np.concatenate(Y_protoset_cumuls)
                if args.rs_ratio > 0:
                    scale_factor = (len(X_train) * args.rs_ratio) / (len(X_protoset) * (1 - args.rs_ratio))
                    rs_sample_weights = np.concatenate((ori_sample_weights, np.ones(len(X_protoset)) * scale_factor))
                    # number of samples per epoch, undersample on the new classes
                    rs_num_samples = int(len(X_train) / (1 - args.rs_ratio))
                    print("X_train:{}, X_protoset:{}, rs_num_samples:{}".format(len(X_train), len(X_protoset),
                                                                                rs_num_samples))

                trainset.proto_sets_x = X_protoset
                trainset.proto_sets_y = Y_protoset
                trainset.comput_mean_and_std()

        # imprint weights
        if iteration > start_iter and args.imprint_weights:
            print("Imprint weights")
            #########################################
            # compute the average norm of old embdding
            old_embedding_norm = tg_model.fc.fc1.weight.data.norm(dim=1, keepdim=True)
            average_old_embedding_norm = torch.mean(old_embedding_norm, dim=0).to('cpu').type(torch.DoubleTensor)

            tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
            num_features = tg_model.fc.in_features
            novel_embedding = torch.zeros((new_feature, num_features))

            for cls_idx in range(st, lt):
                cls_indices = np.array([i == cls_idx for i in map_Y_train])
                cls_indices_1 = cls_indices[np.where(cls_indices < len(X_train))]
                assert (len(np.where(cls_indices == 1)[0]) <= dictionary_size)
                evalset.data = X_train[cls_indices[0:len(X_train)]]
                evalset.targets = np.zeros(evalset.data.shape[0])  # zero labels
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                         shuffle=False, num_workers=2)
                num_samples = evalset.data.shape[0]
                cls_features = compute_features(tg_feature_model, evalloader, num_samples, num_features, device=device)

                norm_features = F.normalize(torch.from_numpy(cls_features), p=2, dim=1)
                cls_embedding = torch.mean(norm_features, dim=0)
                novel_embedding[cls_idx - st] = F.normalize(cls_embedding, p=2, dim=0) * average_old_embedding_norm
            tg_model.to(device)
            tg_model.fc.fc2.weight.data = novel_embedding.to(device)

        ############################################################

        if args.rs_ratio > 0:
            print("Weights from sampling:", rs_sample_weights)
            train_sampler = torch.utils.data.sampler.WeightedRandomSampler(rs_sample_weights, rs_num_samples)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, \
                                                      shuffle=False, sampler=train_sampler, num_workers=2)
        else:
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size,
                                                      shuffle=True, num_workers=2)

        evalset.data = X_valid_cumul
        evalset.targets = map_Y_valid_cumul
        evalloader = torch.utils.data.DataLoader(evalset, batch_size=test_batch_size,
                                                 shuffle=True, num_workers=2)
        testset.data = X_valid_cumul
        testset.targets = map_Y_valid_cumul
        testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size,
                                                 shuffle=False, num_workers=2)

        print('Max and Min of train labels: {}, {}'.format(min(map_Y_train), max(map_Y_train)))
        print('Max and Min of valid labels: {}, {}'.format(min(map_Y_valid_cumul), max(map_Y_valid_cumul)))

        ##############################################################

        ckp_name = args.main_directory + '/{}_run_{}_iteration_{}_model.pth'.format(main_ckp_prefix, iteration_total,
                                                                                    iteration)
        print('check point address of original model', ckp_name)

        if args.load_dricet_mode and iteration == args.start_generate_phase:
            print("###############################")
            print("Loading original models weights from checkpoint")
            torch_loaded = torch.load(args.saved_model_address)
            tg_model.load_state_dict(torch_loaded['model_state_dict'])
            
            model_loaded = True
            print("###############################")
        elif args.load_dricet_mode and iteration < args.start_generate_phase:
            continue
        elif args.resume and os.path.exists(ckp_name):
            print("###############################")
            print("Loading original models weights from checkpoint")
            torch_loaded = torch.load(ckp_name)
            tg_model.load_state_dict(torch_loaded['model_state_dict'])
            
            print("###############################")

        else:
            ###############################
            if iteration > start_iter and args.less_forget:
                # fix the embedding of old classes
                ignored_params = list(map(id, tg_model.fc.fc1.parameters()))
                base_params = filter(lambda p: id(p) not in ignored_params, \
                                     tg_model.parameters())
                tg_params = [{'params': base_params, 'lr': base_lr, 'weight_decay': custom_weight_decay}, \
                             {'params': tg_model.fc.fc1.parameters(), 'lr': 0, 'weight_decay': 0}]
            else:
                tg_params = tg_model.parameters()

            ###############################
            tg_model = tg_model.to(device)
            if iteration > start_iter:
                ref_model = ref_model.to(device)
            tg_optimizer = optim.SGD(tg_params, lr=base_lr, momentum=custom_momentum, weight_decay=custom_weight_decay)
            tg_lr_scheduler = ReduceLROnPlateau(tg_optimizer, factor=lr_factor, patience=lr_patience,
                                                threshold=lr_threshold)
            #############################
            weights = trainset.compute_class_weights()
            tg_model = incremental_train_and_eval(ckp_name, args.epochs, \
                                                  tg_model, ref_model, tg_optimizer, tg_lr_scheduler, \
                                                  trainloader, testloader, evalloader, \
                                                  iteration, start_iter, cur_lamda, \
                                                  args.dist, args.K, args.lw_mr, args.ro, device=device, \
                                                  da_coef=args.da_coef, alpha_3=args.alpha_3,
                                                  temprature=args.temprature, \
                                                  weight_per_class=torch.tensor(weights).float().to(device),
                                                  continual_norm=args.continual_norm)

        ### Exemplars
        nb_protos_cl = args.nb_protos
        nn.Sequential(*list(tg_model.children())[:-1])
        tg_feature_model = nn.Sequential(*list(tg_model.children())[:-1])
        num_features = tg_model.fc.in_features

        # Prepare the protoset
        X_protoset_cumuls = []
        Y_protoset_cumuls = []
        if args.add_data:
            print("generation")

            if iteration == start_iter:
                main_ckp_prefix = main_ckp_prefix + '_bsg_' + str(args.bs) + '_lrg_' + str(
                    args.generation_lr) + '_rfg_' + str(args.r_feature) + '_tv_l2g_' + str(args.tv_l2) + '_l2g_' + str(
                    args.l2) + '_beta2_' + str(args.beta_2) + '_alpha3_' + str(args.alpha_3) + '_dist_' + str(
                    args.dist) + '_mlm_' + str(args.main_loss_multiplier)
            print("new_checkpoint_prefix: ")
            print(main_ckp_prefix)
            if args.generate_more:
                generation_path = "generation_more_optimized"
            else:
                generation_path = "generations"
            # final images will be stored here:
            adi_data_path = args.main_directory + '/final_images/{}_run_{}_iteration_{}_model.pth'.format(
                main_ckp_prefix, iteration_total, iteration)

            # temporal data and generations will be stored here
            exp_name = args.main_directory + '/{}/{}_run_{}_iteration_{}_rand_model.pth'.format(generation_path,
                                                                                                main_ckp_prefix,
                                                                                                iteration_total,
                                                                                                iteration)
            generated_batch_add = args.main_directory + '/{}/{}_generated_data_run_{}_iteration_{}_rand.pkl'.format(
                generation_path, main_ckp_prefix, iteration_total, iteration)
            generated_target_add = args.main_directory + '/{}/{}_generated_label_run_{}_iteration_{}_rand.pkl'.format(
                generation_path, main_ckp_prefix, iteration_total, iteration)
            print("generated_batch_add", generated_batch_add)
            print("generated_samples", exp_name)

            if args.add_sampler and iteration >= args.start_generate_phase and iteration < args.nb_phases - 1:
                args.iterations = 2000
                args.start_noise = True
                bs = args.bs
                jitter = 30

                parameters = dict()
                parameters["resolution"] = args.resolution
                parameters["random_label"] = False
                parameters["start_noise"] = True
                parameters["detach_student"] = False
                parameters["do_flip"] = args.do_flip
                parameters["random_label"] = args.random_label
                parameters["store_best_images"] = args.store_best_images
                criterion = nn.CrossEntropyLoss()
                coefficients = dict()
                coefficients["r_feature"] = args.r_feature
                coefficients["first_bn_multiplier"] = args.first_bn_multiplier
                coefficients["tv_l1"] = args.tv_l1
                coefficients["tv_l2"] = args.tv_l2
                coefficients["l2"] = args.l2
                coefficients["lr"] = args.generation_lr
                coefficients["main_loss_multiplier"] = args.main_loss_multiplier

                network_output_function = lambda x: x

                hook_for_display = None
                hook_for_self_eval = lambda x, y: validate_one(x, y, tg_model)
                print("labels", min(map_Y_train), max(map_Y_train))
                map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])

                evalset.data = X_valid_cumul
                evalset.targets = map_Y_valid_cumul
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                         shuffle=False, num_workers=2)

                cm = compute_confusion_matrix(tg_model, evalloader, device=device)
                gamma = 0.5
                alpha = torch.FloatTensor([1 for _ in range(tg_model.fc.out_features)])
                ImpressionEngine = ImpressionClass(net_teacher=tg_model,
                                                   final_data_path=adi_data_path,
                                                   path=exp_name,
                                                   parameters=parameters,
                                                   setting_id=args.setting_id,
                                                   bs=bs,
                                                   use_fp16=args.fp16,
                                                   jitter=jitter,
                                                   criterion=criterion,
                                                   coefficients=coefficients,
                                                   network_output_function=network_output_function,
                                                   hook_for_display=hook_for_display,
                                                   hook_for_self_eval=hook_for_self_eval,
                                                   device=device,
                                                   target_classes_min=0,
                                                   target_classes_max=max(map_Y_train),
                                                   mean_image_dir='mean/' + args.data + '/',
                                                   order_mine=sub_order,
                                                   cm=cm,
                                                   alpha=alpha,
                                                   gamma=gamma,
                                                   data=args.data,
                                                   look_back=args.look_back,
                                                   synthesis=not args.not_synthesis)

                if args.generate_more:
                    if args.nb_generation == 0:
                        number_of_batches = int(len(prototypes[iteration]) * (iteration + 1) / (2 * bs))
                    else:
                        number_of_batches = args.nb_generation
                else:
                    number_of_batches = 1
                print("number of generated batch loops", number_of_batches)
                for j in range(number_of_batches):
                    generated, targets = ImpressionEngine.generate_batch(net_student=None,
                                                                         use_mean_initialization=args.use_mean_initialization,
                                                                         beta_2=args.beta_2)
                    X_protoset_cumuls.append(generated)
                    Y_protoset_cumuls.append(targets)
                    with open(generated_batch_add, 'wb') as f:
                        pickle.dump(X_protoset_cumuls, f)
                    with open(generated_target_add, 'wb') as f:
                        pickle.dump(Y_protoset_cumuls, f)
                for hook in ImpressionEngine.loss_r_feature_layers:
                    hook.close()
            else:
                if os.path.exists(generated_batch_add):
                    with open(generated_batch_add, 'rb') as f:
                        X_protoset_cumuls = pickle.load(f)
                    with open(generated_target_add, 'rb') as f:
                        Y_protoset_cumuls = pickle.load(f)
                        print("read previouse generated data with size of ",len(Y_protoset_cumuls))
                else:
                    print("no new generated data for this phase")

        if args.visualize:
                if not os.path.exists('../saved_umap'):
                    os.makedirs('../saved_umap')
                if not os.path.exists('../saved_umap' + '/' + args.mode):
                    os.makedirs('../saved_umap' + '/' + args.mode)
                # Calculate validation error of model on the cumul of classes:
                print('Visualizing TSNE...')
                evalset.data = X_train_cumul
                evalset.targets = map_Y_train_cumul
                evalset.proto_sets_x = []
                evalset.proto_sets_y = []
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                         shuffle=False, num_workers=2)
                trans_fitting = fit_umap(tg_model, evalloader, evalset, device, 'train', '', args.nb_phases)

                X_protoset = np.concatenate(X_protoset_cumuls)
                Y_protoset = np.concatenate(Y_protoset_cumuls)
                evalset.proto_sets_x = X_protoset
                evalset.proto_sets_y = Y_protoset
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                         shuffle=False, num_workers=2)
                visualize(trans_fitting, tg_model, evalloader, evalset, device, '../saved_umap/{}_run_{}_iteration_{}'.format(main_ckp_prefix, iteration_total,
                                                                                    iteration), 'train', args.mode, args.nb_phases)
                plt.clf()

                evalset.data = []
                evalset.targets = []
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                         shuffle=False, num_workers=2)
                visualize(trans_fitting, tg_model, evalloader, evalset, device, '../saved_umap/{}_run_{}_iteration_{}'.format(main_ckp_prefix, iteration_total,
                                                                                    iteration), 'train', args.mode + '_only_synthesized',
                          args.nb_phases)
                plt.clf()

                evalset.proto_sets_x = []
                evalset.proto_sets_y = []
                map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
                evalset.data = X_valid_cumul
                evalset.targets = map_Y_valid_cumul
                evalloader = torch.utils.data.DataLoader(evalset, batch_size=eval_batch_size,
                                                         shuffle=False, num_workers=2)
                visualize(trans_fitting, tg_model, evalloader, evalset, device, '../saved_umap/{}_run_{}_iteration_{}'.format(main_ckp_prefix, iteration_total,
                                                                                    iteration),'eval', args.mode, args.nb_phases)
                plt.clf()

        ##############################################################
        # Calculate validation error of model on the first nb_cl classes:
        if args.validate:
            if (args.load_dricet_mode) or args.load_dricet_mode == False:
                map_Y_valid_ori = np.array([order_list.index(i) for i in Y_valid_ori])
                print('Computing accuracy on the original batch of classes...')
                testset.data = X_valid_ori
                testset.targets = map_Y_valid_ori
                testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size,
                                                         shuffle=False, num_workers=2)
                ori_acc = compute_accuracy(tg_model, testloader, device=device)
                top1_acc_list_ori[iteration, :, iteration_total] = np.array(ori_acc).T
                ##############################################################
                # Calculate validation error of model on the cumul of classes:
                map_Y_valid_cumul = np.array([order_list.index(i) for i in Y_valid_cumul])
                print('Computing cumulative accuracy...')
                testset.data = X_valid_cumul
                testset.targets = map_Y_valid_cumul
                testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size,
                                                         shuffle=False, num_workers=2)

                if iteration == (args.nb_phases):
                    metrics = {"Best Acc": cumul_acc[0]}
                    wandb.log(metrics)
                cumul_acc = compute_accuracy(tg_model, testloader, device=device)
                top1_acc_list_cumul[iteration, :, iteration_total] = np.array(cumul_acc).T
                ##############################################################
                # Calculate confusion matrix
                print('Computing confusion matrix...')
                cm = compute_confusion_matrix(tg_model, evalloader, device=device)
                print(cm)

            ##############################################################
