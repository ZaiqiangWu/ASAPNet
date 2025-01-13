import numpy as np
import cv2
import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, "..","..","..")))
from Datasets.human_mannequin.raw_human_mannequin import RawHumanMannequin, RandomAffineMatrix
import util.util as util
from matplotlib import pyplot as plt

def main():
    dataset = RawHumanMannequin()
    dataset.randomaffine = RandomAffineMatrix(degrees=20, translate=(0.2, 0.2), scale=(0.8, 1.5),
                                              shear=(-5, 5, -5, 5))
    print(len(dataset))
    garment_img, vm_img, dpi_img, mask_img= dataset[0]
    img=util.tensor2im(dpi_img, normalize=True,rgb=True)
    plt.imshow(img)
    plt.show()

if __name__ == '__main__':
    main()