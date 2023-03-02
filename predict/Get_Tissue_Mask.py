import sys
import os
import openslide
import cv2
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu

parser = argparse.ArgumentParser(description="Get Tissue region Mask")
parser.add_argument("wsi_path",default="None",metavar="WSI_PATH",
                    type=str,help="Path to the WSI file")##position parameter
parser.add_argument("npy_path",default="None",metavar="NPY_PATH",
                    type=str,help="The path to save mask npy")
parser.add_argument("--level",default=3,metavar="LEVEL",type=int
                    ,help="which level of wsi to deal with")
parser.add_argument("--RGB_min",default=50,metavar="RGB_MIN",type=int,
                    help="The threshold value of RGB color space")

class GetTissueMask(object):
    def __init__(self,wsiPath,npyPath,level,RGBmin):
        self.wsi_path = wsiPath
        self.npy_path = npyPath
        self.level = level
        self.RGB_min = RGBmin
    def get_tissue_mask(self):
        slide = openslide.OpenSlide(self.wsi_path)
        print('{} Slide Reading ...'.format(self.wsi_path))
        if slide.level_count > self.level:
            level = self.level
        else:
            level = slide.level_count - 1
        Whole_slide_RGB = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')
        print('Slide level {} dims: {}'.format(str(level),str(slide.level_dimensions[level])))
        factor = slide.level_downsamples[level]
        print('Slide level {} downsample factor {}'.format(level, factor))
        print('To numpy array ...')
        Whole_slide_RGB = np.array(Whole_slide_RGB)
        Whole_slide_RGB = np.transpose(Whole_slide_RGB, axes=[1, 0, 2])
        print('To HSV color space ...')
        Whole_Slide_HSV = cv2.cvtColor(Whole_slide_RGB, cv2.COLOR_BGR2HSV)

        background_R = Whole_slide_RGB[:, :, 0] > threshold_otsu(Whole_slide_RGB[:, :, 0])
        background_G = Whole_slide_RGB[:, :, 1] > threshold_otsu(Whole_slide_RGB[:, :, 1])
        background_B = Whole_slide_RGB[:, :, 2] > threshold_otsu(Whole_slide_RGB[:, :, 2])
        tissue_RGB = np.logical_not(background_R & background_G & background_B)
        tissue_S = Whole_Slide_HSV[:, :, 1] > threshold_otsu(Whole_Slide_HSV[:, :, 1])
        min_R = Whole_slide_RGB[:, :, 0] > self.RGB_min
        min_G = Whole_slide_RGB[:, :, 1] > self.RGB_min
        min_B = Whole_slide_RGB[:, :, 2] > self.RGB_min

        tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B
        np.save(self.npy_path, tissue_mask)

        return tissue_mask,factor


def run(args):
    logging.basicConfig(level=logging.INFO)
    tissue_mask,factor = GetTissueMask(args.wsi_path,args.npy_path,args.level,args.RGB_min).get_tissue_mask()
    plt.figure()
    plt.imshow(tissue_mask)
    plt.savefig(args.npy_path+".jpg")
    # cv2.imwrite(args.npy_path+"_mask.jpg",tissue_mask)

def main():
    args = parser.parse_args()
    run(args)

if __name__ == '__main__':
    main()





