import sys
import os
import numpy as np
import argparse
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd
import staintools
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from ColorNorm import ColorNorm
class CTCdataset(data.Dataset):
    def __init__(self,csv_file,size,targetImage=None,train=True,level=0,colornorm=False,savepatch=False,transform=None):
        self.data = pd.read_csv(csv_file)
        # self.train = train
        # datatrain, datatest = train_test_split(self.dat)
        # if self.train:
        #     self.data=datatrain
        #     print(self.data)
        # else:
        #     self.data=datatest
        slides = []
        grid = []
        for i in self.data['slides']:
            slides.append(openslide.OpenSlide(i))
        slideIDx = []
        for i in range(self.data.shape[0]):
            print(self.data['grids'][i])
            tempoints = np.load(self.data['grids'][i])
            slideIDx.extend([i]*tempoints.shape[0])
            for p in range(tempoints.shape[0]):
                grid.append(tuple(tempoints[p,:]))
        self.level = level
        self.size = size
        self.grid = grid
        self.slides = slides
        self.targets = list(self.data['targets'])
        self.slideIDx = slideIDx
        self.transform = transform
        self.train = train
        self.savepatch = savepatch
        self.colornorm = colornorm
        self.targetImage = targetImage
        if self.colornorm:
            i2 = staintools.read_image(self.targetImage)
            self.normalizer = staintools.StainNormalizer(method='vahadane')
            self.normalizer.fit(i2)

    def __len__(self):
        return len(self.grid)
    def __getitem__(self, index):
        if  torch.is_tensor(index):
            index = index.tolist()
        slideIDx = self.slideIDx[index]
        target = self.targets[slideIDx]
        coord = self.grid[index]
        print(self.train)
        print('Patch {slide} {coord}'.format(slide=os.path.basename(self.data['slides'][slideIDx]),coord=str(coord)))
        img = self.slides[slideIDx].read_region(coord,0,(self.size,self.size)).convert('RGB')
        if self.colornorm:
            if self.targetImage is None:
                print('Color Normalization need target Image of WSI')
            else:
                print('staintools color normalizing....')
                #img= Image.fromarray(ColorNorm(img,self.targetImage))###use tensorflow ColorNorm color normalization
                img = self.normalizer.transform(np.array(img)) ##use staintools to color normalization
                #print(img)
        if self.savepatch:
            dir = (self.data['grids'][slideIDx]).split('.grid.npy')[0]
            print(dir)
            if not os.path.exists(dir):
                os.mkdir(dir)
            if self.colornorm:
                img.save((dir+'/'+ str(coord[0]) + '_' + str(coord[1]) + '_colornorm.jpg'))
            else:
                img.save((dir + '/' + str(coord[0]) + '_' + str(coord[1]) + '.jpg'))

        if self.transform is not None:
            img = self.transform(img)
        if self.train :
            print('This is train model.........')
            return img, target
        else:
            print('This is testing model.........')
            return img,target,os.path.basename(self.data['slides'][slideIDx]),coord
def main():
    os.chdir('/public5/lilab/student/myang/project/lili/GG/test')
    target = '/public5/lilab/student/myang/project/lili/GG/N02171-A.kfb.mask.Tumor/59712_45621.jpg'
    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    test_dset = CTCdataset(csv_file='/public5/lilab/student/myang/project/lili/GG/test/slide.csv',targetImage=target,colornorm=True,size=224,savepatch=False,train=False)#,transform=trans)
    # test_dset = CTCdataset(csv_file='/public5/lilab/student/myang/project/lili/GG/test/slide.csv',
    #                       size=224, savepatch=True, train=False)
    # loader = torch.utils.data.DataLoader(
    #     test_dset,
    #     batch_size=256, shuffle=False,
    #     num_workers=1, pin_memory=False)

    i = 0
    for img,target ,image, coord in test_dset:
        i+=1
        if i < 100:
            print(target)
            print(image)
            print(coord)
            plt.imshow(img)
            plt.show()
        else:
            break

if __name__ == '__main__':
    main()