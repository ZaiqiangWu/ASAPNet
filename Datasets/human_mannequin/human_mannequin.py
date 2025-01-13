import random

import PIL
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
from util.random_affine_batch import RandomAffineBatch
from util.densepose_util import IUV2UpperBodyImg


class HumanMannequin(data.Dataset):
    def __init__(self, path='./data/HumanMannequin/azuma_yellow', img_size=512):
        super(HumanMannequin, self).__init__()
        self.img_dir = path
        self.image_list = self.__get_image_list()
        self.randomaffine = RandomAffineBatch(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.5),
                                              shear=(-5, 5, -5, 5))
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        garment_path = self.image_list[index]
        garment_img = Image.open(garment_path)
        vm_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_vm.jpg')
        vm_img = Image.open(vm_path)
        atten_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_atten.png')
        atten_mask = Image.open(atten_path)

        if self.transform is not None:
            garment_img = self.transform(garment_img)
            vm_img = self.transform(vm_img)
            atten_mask = self.transform(atten_mask)
        atten_mask = 1.0 - atten_mask
        garment_img, vm_img, atten_mask = self.randomaffine.forward([garment_img, vm_img, atten_mask])
        atten_mask = 1.0 - atten_mask
        if torch.cuda.is_available():
            garment_img, vm_img, atten_mask = garment_img.cuda(), vm_img.cuda(), atten_mask.cuda()
        return self._normalize(garment_img), self._normalize(vm_img), atten_mask

    def __get_image_list(self):
        image_list = []
        filelist = os.listdir(self.img_dir)
        for item in filelist:
            if item.endswith('_garment.jpg'):
                image_list.append(os.path.join(self.img_dir, item))
        return image_list

    def __len__(self):
        return len(self.image_list)

    def _normalize(self, x):
        # map from 0,1 to -1,1
        return x * 2.0 - 1.0


class HumanMannequinDp(HumanMannequin):
    def __init__(self, path='./data/HumanMannequin/azuma_yellow', img_size=512):
        super(HumanMannequinDp, self).__init__(path, img_size=img_size)

    def __getitem__(self, index):
        garment_path = self.image_list[index]
        garment_img = Image.open(garment_path)
        vm_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_vm.jpg')
        vm_img = Image.open(vm_path)
        atten_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_atten.png')
        atten_mask = Image.open(atten_path)
        mask_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_mask.png')
        mask_img = Image.open(mask_path)
        iuv_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_iuv.npy')
        IUV = np.load(iuv_path)
        dp_img = IUV2UpperBodyImg(IUV)
        dp_img = PIL.Image.fromarray(dp_img)

        if self.transform is not None:
            garment_img = self.transform(garment_img)
            vm_img = self.transform(vm_img)
            atten_mask = self.transform(atten_mask)
            mask_img = self.transform(mask_img)
            dp_img = self.transform(dp_img)
        atten_mask = 1.0 - atten_mask
        garment_img, vm_img, atten_mask, dp_img, mask_img = self.randomaffine.forward(
            [garment_img, vm_img, atten_mask, dp_img, mask_img])
        atten_mask = 1.0 - atten_mask
        if torch.cuda.is_available():
            garment_img, vm_img, atten_mask, dp_img, mask_img = garment_img.cuda(), vm_img.cuda(), atten_mask.cuda(), dp_img.cuda(), mask_img.cuda()
        return self._normalize(garment_img), self._normalize(vm_img), self._normalize(dp_img), atten_mask, mask_img
