import openslide
import os
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.color import rgb2hsv
from skimage import measure,data,color
from openslide.deepzoom import DeepZoomGenerator
import numpy as np
import cv2
import sys
import argparse

parser = argparse.ArgumentParser(description="Get Grid points for wsi")
parser.add_argument("wsi_path",default="None",metavar="WSI_PATH",
                    type=str,help="Path to the WSI file")##position parameter
parser.add_argument("out_path",default="None",metavar="out_PATH",
                    type=str,help="The path to save mask npy")
parser.add_argument("--level",default=3,metavar="LEVEL",type=int
                    ,help="which level of wsi to deal with default 3")
parser.add_argument("--RGB_min",default=50,metavar="RGB_MIN",type=int,
                    help="The threshold value of RGB color space")
parser.add_argument("--patch_size",default=224,metavar="PATCH_SIZE",type=int,
                    help="The size of patch default 224 px")


# sys.path.append('/home/myang/PycharmProjects/PyTorch_test/Pathology_WSI/CTC_predict')
sys.path.append(os.path.split(os.path.abspath(__file__))[0])

args = parser.parse_args()

wsipath = args.wsi_path
basename = os.path.basename(wsipath).split('.')[0]

from Get_Tissue_Mask import GetTissueMask
tissue_mask ,factor = GetTissueMask(wsipath,os.path.join(args.out_path,basename+".mask.npy"),args.level,args.RGB_min).get_tissue_mask()
plt.figure()
plt.imshow(np.transpose(tissue_mask,axes=[1, 0]))
plt.savefig(os.path.join(args.out_path,basename+".mask.jpg"))

from sampled_spot_gen import patch_point_in_mask_gen
grid_point = patch_point_in_mask_gen(os.path.join(args.out_path,basename+".mask.npy"),os.path.join(args.out_path,basename+".grid.npy"),args.patch_size,factor).get_patch_point()
