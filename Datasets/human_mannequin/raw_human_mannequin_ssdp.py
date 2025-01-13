from Datasets.human_mannequin.raw_human_mannequin import RawHumanMannequin, get_inverse_trans
import torch
import cv2
import numpy as np
import os
import PIL
from PIL import Image

class RawHumanMannequinSSDP(RawHumanMannequin):
    def __init__(self,path,img_size=512):
        super().__init__(path=path,img_size=img_size,simplified_dp=True)

    def __getitem__(self, index):
        garment_path = self.image_list[index]
        garment_img = np.array(Image.open(garment_path))
        raw_h, raw_w = self.raw_height, self.raw_width
        trans2roi_path = os.path.join(self.img_dir,
                                      (os.path.basename(self.image_list[index])).split('_')[0] + '_trans2roi.npy')
        trans2roi = np.load(trans2roi_path)
        inv_trans = get_inverse_trans(trans2roi)
        garment_img = cv2.warpAffine(garment_img, inv_trans, (raw_w, raw_h),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(0, 0, 0))
        # todo:test
        # Apply Gaussian Blur
        # blurred = cv2.GaussianBlur(garment_img, (21, 21), 0)

        # Subtract the blurred image from the original image
        # garment_img = cv2.subtract(garment_img, blurred)

        vm_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_vm.jpg')
        vm_img = np.array(Image.open(vm_path))
        vm_img = cv2.resize(vm_img, (raw_w, raw_h))
        mask_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_mask.png')
        mask_img = np.array(Image.open(mask_path))
        mask_img = cv2.resize(mask_img, (raw_w, raw_h))
        iuv_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_iuv.npy')
        IUV = np.load(iuv_path)
        ssdp_path = os.path.join(self.img_dir, (os.path.basename(self.image_list[index])).split('_')[0] + '_ssdp.jpg')
        ssdp_img = np.array(Image.open(ssdp_path))
        ssdp_img = cv2.resize(ssdp_img, (raw_w, raw_h))

        new_trans2roi = self.randomaffine(trans2roi)

        roi_garment_img = cv2.warpAffine(garment_img, new_trans2roi, (1024, 1024),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0))
        roi_ssdp_img = cv2.warpAffine(ssdp_img, new_trans2roi, (1024, 1024),
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
            roi_garment_img = self.transform(PIL.Image.fromarray(roi_garment_img))
            roi_vm_img = self.transform(PIL.Image.fromarray(roi_vm_img))
            roi_mask_img = self.transform(PIL.Image.fromarray(roi_mask_img))
            roi_ssdp_img = self.transform(PIL.Image.fromarray(roi_ssdp_img))
        if torch.cuda.is_available():
            roi_garment_img = roi_garment_img.cuda()
            roi_vm_img = roi_vm_img.cuda()
            roi_mask_img = roi_mask_img.cuda()
            roi_ssdp_img = roi_ssdp_img.cuda()
        return self._normalize(roi_garment_img), self._normalize(roi_vm_img), self._normalize(roi_ssdp_img), roi_mask_img