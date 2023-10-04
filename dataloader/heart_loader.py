# Helper functions to load data for in-distribution and out-of-distribution datasets
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset

import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import scipy.io as sio
import random


class Heart(data.Dataset):

    def __init__(self, data_list, transform=None, used_labels=None, order=None,
                 mode='train'):  # label_option='iid', dataset_option='single',
        # if outlier_exposure is true, then the dataset will include outlier images
        self.transform = transform
        self.data_list = [data_list]
        self.mode = mode

        self.labels = ['quality_label', 'view_label', 'quality_label_t2']
        self.option = 'single'  # dataset_option
        self.order_list = [2, 3, 0, 4, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        if order != None:
            self.order_list = order
        self.data_dict = self.read_from_list(self.data_list, '')
        self.quality_classes = ['garb', 'poor', 'fair', 'good', 'xlnt']

        labels = np.unique(self.data_dict['view_label'])
        if used_labels != None:
            ignore_list = np.delete(labels, used_labels)
            print(ignore_list)
            self.data_dict = self.ignore_class(self.data_dict, ignore_list.tolist(), tag='view_label')

        self.view_classes = ['AP2', 'AP3', 'AP4', 'AP5', 'PLAX', 'RVIF', 'SUBC4', 'SUBC5',
                             'SUBIVC', 'PSAX(A)', 'PSAX(M)', 'PSAX(PM)', 'PSAX(APIX)', 'SUPRA', 'unknown', 'garbage']
        if order != None:
            self.view_classes = ['AP2', 'AP4', 'PLAX', 'AP3', 'AP5', 'RVIF', 'SUBC4', 'SUBC5',
                                 'SUBIVC', 'PSAX(A)', 'PSAX(M)', 'PSAX(PM)', 'PSAX(APIX)', 'SUPRA', 'unknown',
                                 'garbage']
        if used_labels != None:
            self.view_classes = [self.view_classes[index] for index in used_labels]

        # compute class weights using inverse frequency
        self.quality_label_weights = self.compute_class_weights('quality_label')

        # this view weight has a size of 15, not like the 13 output classes. We use this to weigh the random sampler.
        self.view_label_weights = self.compute_class_weights('view_label')
        print(self.view_label_weights)
        print("Heart dataset loaded, N = " + str(len(self.data_dict['file_address'])))

        self.data = self.data_dict['file_address']
        self.targets = self.data_dict['view_label']
        print(len(self.data), len(self.targets))
        self.flipped_version = False
        self.proto_sets_x = []
        self.proto_sets_y = []
        self.mean_generate = [0]
        self.std_generate = [1]
        # print(self.mean,self.std)
        self.transform_generate = _get_transform_fake(self.mean_generate, self.std_generate, 'train', to_pil_image=True)
        self.o = 0

    def __getitem__(self, index):
        if index < len(self.data):

            img_path = self.data[index]

            # augment the view label so "other" and "garbage" mean something when it comes to uncertainty
            # the view label being one-hot, when it is "other" or "garbage", it is uniform distribution
            view = self.targets[index]

            # load the imag
            if self.mode == 'valid' or self.mode == 'test':
                img = self.read_image(img_path, frame_option='middle')  # returns a 2D DxD uint8 array
            else:
                img = self.read_image(img_path)  # returns a 2D DxD uint8 array
            if self.flipped_version:
                img_3ch_tensor = self.transform(img[:, ::-1])
            # print("hereeeeeeee",img_3ch_tensor[0,:,:].unsqueeze(0).shape)
            else:
                img_3ch_tensor = self.transform(img)
            return img_3ch_tensor[0, :, :].unsqueeze(0), view
        else:
            # print("using protoset",self.o)
            # self.o += 1
            img_3ch_tensor = self.transform_generate(self.proto_sets_x[index - len(self.data)].reshape((224, 224, 1)))
            view = self.proto_sets_y[index - len(self.data)]
            return img_3ch_tensor[0, :, :].unsqueeze(0), view

    def __len__(self):
        return len(self.data) + len(self.proto_sets_x)

    def get_label_loss_weights(self):
        return self.view_label_weights[:len(self.view_classes)], self.quality_label_weights[:len(self.quality_classes)]

    def get_label_sampling_weights(self):
        return self.view_label_weights, self.quality_label_weights

    def get_label_sampling_weights_per_item(self):
        # directly supports pytorch's WeightedRandomSampler, associates the sampling weight of each sample with its view class
        weights = [0] * len(self.data_dict['view_label'])
        for i in range(len(self.data_dict['view_label'])):
            class_of_i = self.data_dict['view_label'][i]
            weights[i] = self.view_label_weights[class_of_i]
        return weights

    def get_classes(self):
        return self.view_classes, self.quality_classes

    # read file addresses and labels from Jorden's list
    # added some options for taking different subsets of data
    def read_from_list(self, list_of_files, data_root):

        def read_text_file(fname, data_root=''):
            f = open(fname, 'r')
            lines = f.readlines()
            file_address = []
            quality_label = []
            view_label = []
            quality_label_t2 = []

            for line in lines:
                parts = line.split('\n')
                line = parts[0]
                parts = line.split(',')
                fa = '../..' + data_root + parts[0]
                if not os.path.isfile(fa):
                    print(fa + ' does not exist')
                    continue
                if self.option == 'inter' and int(parts[3]) == -1:
                    continue
                file_address.append(fa)
                quality_label.append(int(parts[1]))
                view_label.append(int(parts[2]))

                if len(parts) == 4:
                    quality_label_t2.append(int(parts[3]))
                else:
                    quality_label_t2.append(int(-1))

            return file_address, quality_label, view_label, quality_label_t2

        file_address = []
        quality_label = []
        view_label = []
        quality_label_t2 = []

        for fname in list_of_files:
            fa, ql, vl, ql2 = read_text_file(fname, data_root)
            file_address += fa
            quality_label += ql
            vl_map = []
            for v in vl:
                vl_map.append(self.order_list[v])
            view_label += vl_map
            quality_label_t2 += ql2

        quality_label = np.asarray(quality_label)
        view_label = np.asarray(view_label)
        quality_label_t2 = np.asarray(quality_label_t2)

        return {'file_address': file_address, 'quality_label': quality_label, 'view_label': view_label,
                'quality_label_t2': quality_label_t2, 'num_files': len(file_address)}

    # remove a subset of classes, however, it does not rebase the label, all classes continue using the original index
    # input is a data_dict as defined by read_from_list
    def ignore_class(self, data_dict, class_no=None, tag=None):
        # checker
        if type(class_no) == int:
            class_no = [class_no]
        elif type(class_no) == list:
            for c in class_no:
                if type(c) != int:
                    raise ValueError('A class index must be a int value or a list of int value')
        else:
            raise ValueError('A class index must be a int value or a list of int value')

        file_address = []
        quality_label = []
        quality_label_t2 = []
        view_label = []

        for i, c in enumerate(data_dict[tag]):
            if c in class_no:
                continue
            file_address.append(data_dict['file_address'][i])
            quality_label.append(data_dict['quality_label'][i])
            quality_label_t2.append(data_dict['quality_label_t2'][i])
            view_label.append(data_dict['view_label'][i])

        return {'file_address': file_address, 'quality_label': quality_label, 'view_label': view_label,
                'quality_label_t2': quality_label_t2, 'num_files': len(file_address)}

    # slide the classification labels "leftward", ie. (0, 2, 3, 5) -> (0, 1, 2, 3)
    def rebase_label(self, data_dict, tag=None, convert_dict=None):

        if convert_dict is None or type(convert_dict) is not dict:
            convert_dict = dict()
            for new_index, old_index in enumerate(np.unique(data_dict[tag])):
                convert_dict[old_index] = new_index

        for i, c in enumerate(data_dict[tag]):
            data_dict[tag][i] = convert_dict[data_dict[tag][i]]

        return data_dict

    # compute class weights using inverse frequency, should only be used if there are no gaps in class labels
    def compute_class_weights(self, tag):
        arr = self.data_dict[tag]
        freq = np.bincount(arr)
        total_samples = len(arr)
        inv_freq = 1. / freq
        # print(total_samples * inv_freq / len(np.unique(arr)) )
        return (total_samples * inv_freq / len(np.unique(arr)))

    # read image as HxW uint8 from the cine mat file at img_path
    def read_image(self, img_path, frame_option='random'):
        try:
            matfile = sio.loadmat(img_path, verify_compressed_data_integrity=False)
        # except ValueError:
        #     print(fa)
        #     raise ValueError()
        except TypeError:
            print(img_path)
            raise TypeError()

        d = matfile['Patient']['DicomImage'][0][0]

        # if 'ImageCroppingMask' in matfile['Patient'].dtype.names:
        # mask = matfile['Patient']['ImageCroppingMask'][0][0]
        # # d = d*np.expand_dims(mask, axis=2)
        # print(d.shape)
        if frame_option == 'start':
            d = d[:, :, 0]
        elif frame_option == 'middle':
            l = len(d[0, 0, :])
            d = d[:, :, int(l // 2)]
        elif frame_option == 'end':
            d = d[:, :, -1]
        elif frame_option == 'random':
            r = np.random.randint(0, d.shape[2])
            d = d[:, :, r]

        return d

    def comput_mean_and_std(self):
        sum_m = 0
        temp_transform = _get_transform_fake(mean=0, std=1, mode='valid', to_pil_image=True)
        for i in range(len(self.proto_sets_x)):
            index = temp_transform(self.proto_sets_x[i].reshape((224, 224, 1)))[0, :, :]
            m = torch.mean(index)
            sum_m += m
        self.mean_generate = sum_m / len(self.proto_sets_x)
        print(self.mean_generate)
        sum_v = 0
        for i in range(len(self.proto_sets_x)):
            index = temp_transform(self.proto_sets_x[i].reshape((224, 224, 1)))[0, :, :]
            sum_v += torch.sum((index - self.mean_generate) ** 2)
        self.std_generate = sum_v / (len(self.proto_sets_x) * 224 * 224)
        print(self.std_generate)
        return


def get_heart_dataset(mode='train', used_labels=[0, 1], order=None):
    mean = [0.122, 0.122, 0.122]
    std = [0.184, 0.184, 0.184]
    # mean = [0, 0, 0]
    # std = [1, 1, 1]
    transform = _get_transform(mean, std, mode, to_pil_image=True)
    if mode == 'train':
        dataset = Heart('./database_path/train_labels_2.txt', transform, used_labels, order=order, mode=mode)
    elif mode == 'valid':
        dataset = Heart('./database_path/valid_labels_2.txt', transform, used_labels, order=order, mode=mode)
    elif mode == 'test':
        dataset = Heart('./database_path/test_labels_2.txt', transform, used_labels, order=order, mode=mode)
    #     view_c, qual_c = dataset.get_classes()
    #     view_w, qual_w = dataset.get_label_sampling_weights()
    #     view_w_per_item = dataset.get_label_sampling_weights_per_item()

    #     if (subset_size == -1):
    #         if mode=='train':
    #             samp = torch.CCSI_utils.data.WeightedRandomSampler(view_w_per_item, num_samples=len(dataset) )
    #         else:
    #             samp = torch.CCSI_utils.data.SequentialSampler(dataset)
    #     else:
    #         if mode=='train':
    #             samp = torch.CCSI_utils.data.WeightedRandomSampler(view_w_per_item, num_samples=subset_size)
    #         else:
    #             samp = torch.CCSI_utils.data.RandomSampler(dataset, replacement=True, num_samples=subset_size)

    #     data_loader = torch.CCSI_utils.data.DataLoader(dataset, batch_size=bs, sampler=samp, num_workers=2)

    return dataset


def _get_transform(mean, std, mode='valid', to_pil_image=False):
    if mode == 'train':
        aug_transform = transforms.Compose(
            [transforms.Resize((256, 256)),
             transforms.RandomRotation(5),
             transforms.RandomCrop((224, 224))])
    else:  # mode=='valid'
        aug_transform = transforms.Resize((224, 224))
    t_transform = transforms.Compose(
        [aug_transform,
         transforms.Grayscale(num_output_channels=3),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
         ])
    if to_pil_image:
        t_transform = transforms.Compose([transforms.ToPILImage(),
                                          t_transform])
    return t_transform


def _get_transform_fake(mean, std, mode='valid', to_pil_image=False):
    if mode == 'train':
        aug_transform = transforms.Compose(
            [transforms.RandomRotation(5),
             transforms.RandomCrop((224, 224))])
    else:  # mode=='valid'
        aug_transform = transforms.Resize((224, 224))
    t_transform = transforms.Compose(
        [aug_transform,
         transforms.ToTensor(),
         transforms.Normalize(mean, std)
         ])

    # t_transform = transforms.Compose(
    #     [
    #      transforms.ToTensor()
    #     ])

    if to_pil_image:
        t_transform = transforms.Compose([transforms.ToPILImage(),
                                          t_transform])
    return t_transform
