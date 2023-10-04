# Helper functions to load data for in-distribution and out-of-distribution datasets
import numpy as np
import torch
import torchvision.transforms as transforms

from PIL import Image
import numpy as np
from medmnist import OrganAMNIST


def _get_transform(to_pil_image=False, image_size=32):
    t_transform = transforms.Compose(
        [transforms.Pad(4),
         transforms.CenterCrop(image_size),
         transforms.ToTensor()
         ])
    if to_pil_image:
        t_transform = transforms.Compose([transforms.ToPILImage(),
                                          t_transform])
    return t_transform


def _get_transform_fake(mean, std, to_pil_image=False):
    t_transform = transforms.Compose(
        [transforms.Normalize(mean, std)]
    )
    if to_pil_image:
        t_transform = transforms.Compose([transforms.ToPILImage(),
                                          t_transform])
    return t_transform


class Modified_medmnist(OrganAMNIST):

    def __init__(self,
                 split,
                 transform=None,
                 target_transform=None,
                 download=False,
                 as_rgb=False,
                 image_size=32):
        super().__init__(split, transform, target_transform, download, as_rgb)
        self.data = self.imgs
        self.targets = np.array(self.labels).squeeze(1)
        self.view_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.proto_sets_x = []
        self.proto_sets_y = np.array([])
        self.mean_generate = [0]
        self.std_generate = [1]
        self.image_size = image_size
        self.transform = _get_transform(image_size=self.image_size)
        self.transform_generate = _get_transform_fake(self.mean_generate, self.std_generate, to_pil_image=False)
        self.o = 0
        self.return_type = False

    def __getitem__(self, index):
        if index < len(self.data):

            img, target = self.data[index], self.targets[index].astype(int)
            img = Image.fromarray(img)
            if self.as_rgb:
                img = img.convert('RGB')

            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            rt = 0
        else:

            img = torch.from_numpy(self.proto_sets_x[index - len(self.data)])
            target = self.proto_sets_y[index - len(self.data)]
            if self.transform_generate is not None:
                img = self.transform_generate(img)
            if self.target_transform is not None:
                target = self.target_transform(target)
            rt = 1

        if self.return_type:
            return img, target, rt
        else:
            return img, target

    def __len__(self):
        return len(self.data) + len(self.proto_sets_x)

    def compute_class_weights(self):

        arr = np.concatenate((self.proto_sets_y, self.targets))
        freq = np.bincount((arr).astype(int))

        freq = freq / freq.max()
        inv_freq = 1. / freq
        return inv_freq

    def comput_mean_and_std(self):
        sum_m = 0
        for i in range(len(self.proto_sets_x)):
            index = torch.from_numpy(self.proto_sets_x[i])
            m = torch.mean(index, dim=(1, 2))
            sum_m += m
        self.mean_generate = sum_m / len(self.proto_sets_x)
        sum_v = torch.zeros(3)
        for i in range(len(self.proto_sets_x)):
            index = torch.from_numpy(self.proto_sets_x[i])
            sum_v += torch.sum((index - self.mean_generate.unsqueeze(1).unsqueeze(1)) ** 2, dim=(1, 2))
        self.std_generate = sum_v / (len(self.proto_sets_x) * self.image_size * self.image_size)
        return


def get_medmnist_dataset(mode='train', image_size=32, download=False):
    if mode == 'train':
        dataset = Modified_medmnist(split='train', download=download, as_rgb=False, image_size=image_size)
    elif mode == 'test':
        dataset = Modified_medmnist(split='test', download=download, as_rgb=False, image_size=image_size)
    return dataset
