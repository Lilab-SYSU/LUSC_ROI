#-*-coding:utf-8-*-
import sys
import os
import argparse

import pandas as pd
import openslide
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from skimage.filters import threshold_otsu

parser = argparse.ArgumentParser(description="Plot Probility heatmap for a whole slide")
parser.add_argument('--pred_csv',metavar="PRED_CSV",type=str,help="Path to the predicted csv file")
parser.add_argument('--wsi_path',metavar="WSI_PATH",type=str,help="Path to the WSI file")
parser.add_argument('--level',metavar="LEVEL",type=int,help="Which level of WSI to plot")
parser.add_argument('--heatmap_name',metavar="HEATMAP_NAME",type=str,help="Heatmap name to save")

colorDict = {'Normal':sns.light_palette("royalblue", as_cmap=True),'Tumor':sns.light_palette("palegreen", as_cmap=True)}
def Get_heatmap(csv,colorDict,slide,level,nclass,heatmap_name,patchSize=224):
    '''csv : predict result [<U+FEFF>,Normal,Tumor,fnames,coords_X,coords_Y]
       colorDict: color map to class
       slide: path to slide
       level : level of predict
       nclss: number of classes
       heatmap_name: the filename of heatmap to save
       patchSize : patch size used for prediction'''
    df = pd.read_csv(csv)

    names=list(df.columns)[1:nclass+1]

    slide = openslide.OpenSlide(slide)
    Whole_slide_RGB = slide.read_region((0, 0), level, slide.level_dimensions[level]).convert('RGB')

    Whole_slide_RGB = np.array(Whole_slide_RGB)
    Whole_slide_RGB = np.transpose(Whole_slide_RGB, axes=[1, 0, 2])

    background_R = Whole_slide_RGB[:, :, 0] > threshold_otsu(Whole_slide_RGB[:, :, 0])
    background_G = Whole_slide_RGB[:, :, 1] > threshold_otsu(Whole_slide_RGB[:, :, 1])
    background_B = Whole_slide_RGB[:, :, 2] > threshold_otsu(Whole_slide_RGB[:, :, 2])
    tissue_mask = background_R & background_G & background_B
    tissue_mask_3d = np.array([tissue_mask, tissue_mask, tissue_mask])
    tissue_mask_3d_T = np.transpose(tissue_mask_3d, axes=[1, 2, 0])
    Whole_slide_RGB[tissue_mask_3d_T] = 255
    Whole_slide_RGB = np.transpose(Whole_slide_RGB, axes=[1, 0, 2])

    Whole_slide_dim = slide.level_dimensions[level]
    print(Whole_slide_dim)

    sample_factor = slide.level_downsamples[level]

    Whole_slide_arr = np.ones(Whole_slide_dim)
    # plt.figure()
    fig, ax = plt.subplots()
    ax.grid()
    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_xlabel('Width')
    ax.set_ylabel('Height')
    # ax.grid()

    ax.imshow(Whole_slide_RGB,alpha= 0.5)
    for i in range(df.shape[0]):
        line = df.iloc[i,:]
        tileX = line['coords_X'] / sample_factor
        tileY = line['coords_Y'] / sample_factor
        tile_H = patchSize/sample_factor
        tile_W = patchSize/sample_factor
        probs = list(line)[1:nclass+1]
        max_prob=max(probs)
        camp = colorDict[names[probs.index(max(probs))]]
        print(tileX,tileY)
        print(tile_H,tile_W)
        # print(names[probs.index(max(probs))])
        print(max(probs))
        # plt.gca().add_patch(plt.Rectangle((tileX, tileY), tile_W, tile_H, color=mpl.colors.to_hex(camp(max_prob))))
        ax.add_patch(plt.Rectangle((tileX, tileY), tile_W, tile_H, color=mpl.colors.to_hex(camp(max_prob))))
    tem=0.9
    for i in colorDict.keys():
        tem = tem - 0.1
        ax1 = fig.add_axes([0.95, tem, 0.16, 0.035])
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm1 = mpl.cm.ScalarMappable(norm=norm, cmap=colorDict[i])
        sm1.set_array([])
        cb1 = plt.colorbar(sm1, cax=ax1, orientation='horizontal', label=i, ticklocation='top')
        cb1.set_ticks([])
        # ax2 = fig.add_axes([0.95, 0.7, 0.16, 0.035])
    # cmap1 = plt.get_cmap("Reds")
    # norm = mpl.colors.Normalize(vmin=0, vmax=1)
    plt.savefig(heatmap_name, bbox_inches='tight')
    plt.show()

# os.chdir('/public5/lilab/student/myang/project/lili/GG/Predict')
# Get_heatmap('Predict_Result1.csv',colorDict,'L14733-C.kfb.svs',4,2,'L14733-C.heatmap.png')

# os.chdir('/public5/lilab/student/myang/project/lili/GG/test')
# Get_heatmap('Predict_Result1.csv',colorDict,'TCGA-G3-A25Y-01Z-00-DX1.C6BF2202-9030-4460-B0F5-E846C8A44C1E.svs',3,2,'test.heatmap.png')
def main():
    args= parser.parse_args()
    Get_heatmap(args.pred_csv,colorDict,args.wsi_path,args.level,len(colorDict),args.heatmap_name)

if __name__ == "__main__":
    main()