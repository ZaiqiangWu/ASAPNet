import random

import PIL
import torchvision.transforms as transforms
import torch
import numpy as np
import os
import torch.utils.data as data
from PIL import Image
from util.random_affine_batch import RandomAffineBatch
from util.densepose_util import IUV2UpperBodyImg, IUV2TorsoLeg
import cv2
from .raw_human_mannequin import RawHumanMannequin


class SeqRawHumanMannequin(RawHumanMannequin):
    def __init__(self, path='./data/HumanMannequin/jin_raw_01', img_size=512):
        super().__init__(path, img_size)

    def __getitem__(self, index):

        return self.load_data(index)

    def load_data(self, index):
        garment_path = self.image_list[index]
        garment_img = np.array(Image.open(garment_path))
        vm_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_vm.jpg')
        vm_img = np.array(Image.open(vm_path))
        mask_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_mask.png')
        mask_img = np.array(Image.open(mask_path))
        iuv_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_iuv.npy')
        IUV = np.load(iuv_path)
        dpi_img = IUV2TorsoLeg(IUV)
        trans2roi_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_trans2roi.npy')
        trans2roi = np.load(trans2roi_path)
        new_trans2roi = self.randomaffine(trans2roi)

        p = random.uniform(0.0, 1.0)
        if p < 0.6 and index != 0 and index!=self.__len__()-1:
            pp = random.uniform(0.0, 1.0)
            if pp<0.5:
                delta_index=1
            else:
                delta_index=-1
            pre_garment_path = self.image_list[index+delta_index]
            pre_garment_img = np.array(Image.open(pre_garment_path))
            pre_trans2roi_path = os.path.join(self.img_dir,
                                          (os.path.basename(self.image_list[index-1])).split('_')[0] + '_trans2roi.npy')
            pre_trans2roi = np.load(pre_trans2roi_path)
            new_trans2roi, new_pre_trans2roi = self.randomaffine.batch_forward([trans2roi, pre_trans2roi])
            roi_pre_garment_img = cv2.warpAffine(pre_garment_img, new_pre_trans2roi, (1024, 1024),
                                                 flags=cv2.INTER_LINEAR,
                                                 borderMode=cv2.BORDER_CONSTANT,
                                                 borderValue=(0, 0, 0))
        else:
            roi_pre_garment_img = None




        roi_garment_img = cv2.warpAffine(garment_img, new_trans2roi, (1024,1024),
                                              flags=cv2.INTER_LINEAR,
                                              borderMode=cv2.BORDER_CONSTANT,
                                              borderValue=(0, 0, 0))
        roi_dpi_img = cv2.warpAffine(dpi_img, new_trans2roi, (1024, 1024),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        roi_vm_img = cv2.warpAffine(vm_img, new_trans2roi, (1024, 1024),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        roi_mask_img = cv2.warpAffine(mask_img, new_trans2roi, (1024, 1024),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0))
        if self.transform is not None:
            if roi_pre_garment_img is not None:
                roi_pre_garment_img = self.transform(PIL.Image.fromarray(roi_pre_garment_img))
            roi_garment_img = self.transform(PIL.Image.fromarray(roi_garment_img))
            roi_vm_img = self.transform(PIL.Image.fromarray(roi_vm_img))
            roi_mask_img = self.transform(PIL.Image.fromarray(roi_mask_img))
            roi_dpi_img = self.transform(PIL.Image.fromarray(roi_dpi_img))
        if roi_pre_garment_img is None:
            roi_pre_garment_img = torch.zeros_like(roi_garment_img)
        if torch.cuda.is_available():
            roi_garment_img = roi_garment_img.cuda()
            roi_vm_img = roi_vm_img.cuda()
            roi_mask_img = roi_mask_img.cuda()
            roi_dpi_img = roi_dpi_img.cuda()
            roi_pre_garment_img = roi_pre_garment_img.cuda()

        return self._normalize(roi_garment_img),self._normalize(roi_vm_img),self._normalize(roi_dpi_img),roi_mask_img, self._normalize(roi_pre_garment_img)






if __name__ == '__main__':
    # Example usage
    dataset = SeqRawHumanMannequin()
    print(dataset.__len__())
