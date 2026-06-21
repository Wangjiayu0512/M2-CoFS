import numpy as np
import random
# import SimpleITK as sitk
import copy
import os
import torch
from torch.utils.data import Dataset
from glob import glob
from torchvision import transforms
from utils import mkdir, read_img, augment_img, mask2one_hot, preprocess
from utils import get_image_paths

np.random.seed(0)








class AEDataset(Dataset):
    def __init__(self, io, root, transform=True, ssl_transform=True, partition='train'):
        super(AEDataset, self).__init__()
        # data path
        self.files1 = get_image_paths(os.path.join(root, './t1'))
        self.files2 = get_image_paths(os.path.join(root, './t2-flair'))
        self.partition = partition
        # basic operation
        self.transform = transform
        self.ssl_transform = ssl_transform
        self.num_examples = len(self.files1)
        if self.partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples)]).astype(int)
            np.random.shuffle(self.train_ind)
        elif self.partition == 'valid':
            self.val_ind = np.asarray([i for i in range(self.num_examples)]).astype(int)
            np.random.shuffle(self.val_ind)
        elif self.partition == 'test':
            self.test_ind = [i for i in range(self.num_examples)]

        io.cprint("number of " + partition + " examples in dataset" + ": " + str(self.num_examples))

    def __len__(self):
        return self.num_examples
    
    def __normalize__(self, x):
        x = x.astype(np.float32)
        x_min = x.min()
        x_max = x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        return x
    def __getitem__(self, idx):
        img1_path = self.files1[idx]
        img2_path = self.files2[idx]

        T1 = read_img(img1_path)
        T2 = read_img(img2_path)
        T1 = self.__normalize__(T1)
        T2 = self.__normalize__(T2)
        if self.transform:
            mode = random.randint(0, 7)
            T1 = augment_img(T1, mode)
            T2 = augment_img(T2, mode)

        if self.ssl_transform:
        

            img1 = self._tensor(np.expand_dims(T1, axis=-1))
            img2 = self._tensor(np.expand_dims(T2, axis=-1))

            return img1, img2
        else:
            img1 = self._tensor(np.expand_dims(T1, axis=-1))
            img2 = self._tensor(np.expand_dims(T2, axis=-1))
            return img1, img2

    def _tensor(self, x):
        if x.ndim == 3:
            x = np.expand_dims(x, axis=-1)
        return torch.FloatTensor(x.copy()).permute(3, 0, 1, 2)


def random_intensity_shift(imgs_array, brain_mask, limit=0.1):
    """
    Only do intensity shift on brain voxels
    :param imgs_array: The whole input image with shape of (4, 48, 128, 128)
    :param brain_mask:
    :param limit:
    :return:
    """

    shift_range = 2 * limit
    for i in range(len(imgs_array) - 1):
        factor = -limit + shift_range * np.random.random()
        std = imgs_array[i][brain_mask].std()
        imgs_array[i][brain_mask] = imgs_array[i][brain_mask] + factor * std
    return imgs_array


def random_mirror_flip(imgs_array, prob=0.5):
    """
    Perform flip along each axis with the given probability; Do it for all voxels；
    labels should also be flipped along the same axis.
    :param imgs_array:
    :param prob:
    :return:
    """
    for axis in range(1, len(imgs_array.shape)):
        random_num = np.random.random()
        if random_num >= prob:
            if axis == 1:
                imgs_array = imgs_array[:, ::-1, :, :]
            if axis == 2:
                imgs_array = imgs_array[:, :, ::-1, :]
            if axis == 3:
                imgs_array = imgs_array[:, :, :, ::-1]
    return imgs_array


class FusionDataset(Dataset):
    def __init__(self, io, root, transform=True, partition='train'):
        super(FusionDataset, self).__init__()
        # data path
        self.img1_paths = get_image_paths(os.path.join(root, 't1/'))  # T1
        self.img2_paths = get_image_paths(os.path.join(root, 't2-flair/'))  # T2
        self.label_paths = get_image_paths(os.path.join(root, 'label/'))  # Segmentation label

        assert len(self.img1_paths) == len(self.img2_paths), "the number of image pair should be the same"
        assert len(self.img1_paths) == len(self.label_paths), 'The label should correspond to the image'
        # basic operation
        self.palette = [0, 1, 2, 3, 4]  # [BK:0, CSF:1, GM:2, WM:3, MS:4]
        self.transform = transform
        self.partition = partition

        self.num_examples = len(self.label_paths)

        if self.partition == 'train':
            self.train_ind = np.asarray([i for i in range(self.num_examples)]).astype(int)
            np.random.shuffle(self.train_ind)
        elif self.partition == 'valid':
            self.val_ind = np.asarray([i for i in range(self.num_examples)]).astype(int)
            np.random.shuffle(self.val_ind)
        elif self.partition == 'test':
            self.test_ind = [i for i in range(self.num_examples)]
        else:
            raise AssertionError("the declare parameter of partition should be train or test")
        io.cprint("number of " + partition + " examples in dataset" + ": " + str(self.num_examples))

    def __len__(self):
        return self.num_examples
    
    def __normalize__(self, x):
        x = x.astype(np.float32)
        x_min = x.min()
        x_max = x.max()
        if x_max > x_min:
            x = (x - x_min) / (x_max - x_min)
        return x
    
    def __getitem__(self, idx):
        img1_path = self.img1_paths[idx]
        img2_path = self.img2_paths[idx]
        label_path = self.label_paths[idx]

        T1 = read_img(img1_path)
        T1 = self.__normalize__(T1)

        T2 = read_img(img2_path)
        T2 = self.__normalize__(T2)
        seg = read_img(label_path)

        if self.transform:
            mode = random.randint(0, 7)
            T1 = augment_img(T1, mode=mode)
            T2 = augment_img(T2, mode=mode)
            seg = augment_img(seg, mode=mode)

        T1 = np.expand_dims(T1, axis=-1)
        T2 = np.expand_dims(T2, axis=-1)
        seg = mask2one_hot(seg, self.palette)  # [BK:0, CSF:1, GM:2, WM:3, MS:4]
        img1 = self._tensor(T1)
        img2 = self._tensor(T2)
        seg = self._tensor(seg)
        return img1, img2, seg

    def _tensor(self, x):
        return torch.FloatTensor(x.copy()).permute(3, 0, 1, 2)




