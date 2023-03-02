import os
import sys
import logging
import argparse

import numpy as np
import openslide
import cv2
import json
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Get_Tissue_Mask import Get_Tissue_mask

parser = argparse.ArgumentParser(description='Get tumor mask and rest tissue mask of tumor-WSI and '
                                             'save it in npy format')
parser.add_argument('wsi_path', default=None, metavar='WSI_PATH', type=str,
                    help='Path to the WSI file')
parser.add_argument('json_path', default=None, metavar='JSON_PATH', type=str,
                    help='Path to the JSON file')
parser.add_argument('npy_path', default=None, metavar='NPY_PATH', type=str,
                    help='Path to the output npy mask file')
parser.add_argument('--level', default=3, type=int, help='at which WSI level'
                    ' to obtain the mask, default 3')
parser.add_argument('--type', default="Tumor", type=str, help='Type of ROI :Tumor/Normal'
                    ' default Tumor')

def run(args):

    # get the level * dimensions e.g. tumor0.tif level 6 shape (1589, 7514)
    slide = openslide.OpenSlide(args.wsi_path)
    w, h = slide.level_dimensions[args.level]
    mask_tumor = np.zeros((h, w)) # the init mask, and all the value is 0

    # get the factor of level * e.g. level 6 is 2^6
    factor = slide.level_downsamples[args.level]
    Type = args.type
    print(Type)
    with open(args.json_path) as f:
        dicts = json.load(f)
    if dicts[Type]:
        tumor_polygons = dicts[Type]
    else:
        print("This WSI has no %s Region."%Type)
        sys.exit()

    for tumor_polygon in tumor_polygons:
        # plot a polygon
        name = tumor_polygon["name"]
        vertices = np.array(tumor_polygon["vertices"]) / factor
        vertices = vertices.astype(np.int32)

        cv2.fillPoly(mask_tumor, [vertices], (255))

    mask_tumor = mask_tumor[:] > 127
    plt.figure()
    plt.imshow(mask_tumor,cmap='gray')
    plt.savefig(os.path.join((args.npy_path).split('.')[0] + '.'+Type+".mask.jpg"))
    mask_tumor = np.transpose(mask_tumor)

    #####Obtained no tumor region from tissue
    TissueMask = Get_Tissue_mask(args.wsi_path, 50, args.level)
    Notumor = mask_tumor !=True
    NoTumorMask = Notumor & TissueMask
    plt.figure()
    plt.imshow(np.transpose(NoTumorMask), cmap='gray')
    plt.savefig(os.path.join((args.npy_path).split('.')[0] + '.' + "No_tumor.mask.jpg"))

    np.save(args.npy_path.split('npy')[0]+Type+'.mask.npy', mask_tumor)
    np.save(args.npy_path.split('npy')[0]+'No_Tumor.mask.npy', NoTumorMask)

def main():
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
