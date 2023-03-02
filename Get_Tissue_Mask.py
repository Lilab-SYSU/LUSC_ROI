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
def Get_Tissue_mask(wsi_path,RGB_min,level):
    slide = openslide.OpenSlide(wsi_path)
    print('Slide Reading ...')
    Whole_slide_RGB = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')
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
    min_R = Whole_slide_RGB[:, :, 0] > RGB_min
    min_G = Whole_slide_RGB[:, :, 1] > RGB_min
    min_B = Whole_slide_RGB[:, :, 2] > RGB_min

    tissue_mask = tissue_S & tissue_RGB & min_R & min_G & min_B

    return tissue_mask