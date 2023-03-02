import sys
import os
import numpy as np
import argparse
import time
import copy
import random
import openslide
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler

sys.path.append('/Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC')
from CTCdataset import CTCdataset


parser = argparse.ArgumentParser(description='CTC tile classifier training script')
parser.add_argument('--train_csv', type=str, default='', help='path to train MIL library binary')
# parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs (default: 128)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
# parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

# normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.1, 0.1, 0.1])
# trans = transforms.Compose([transforms.ToTensor(),normalize])
# datatrain = CTCdataset(csv_file='/public7/lilab/myang/CTC/2-SCC_convert/slide.csv',size=224,transform=trans)
# next(datatrain)
# train_loader = torch.utils.data.DataLoader(
#     datatrain,
#     batch_size=4, shuffle=False,
#     num_workers=4, pin_memory=False)
# def imshow(img):
#     img = img / 2 +0.5
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1,2,0)))
#     plt.show()
# dataiter = iter(train_loader)
#
# images,labels = dataiter.next()
# imshow(torchvision.utils.make_grid(images))

def imshow(inp, title=None):
    """Imshou for Tensor"""

    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.1, 0.1, 0.1])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

# def train_model(model,criterion,optimizer,scheduler,num_epochs=25):
#     since = time.time()
#
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch,num_epochs - 1))
#         print('-' * 10)
#
#         for phase in ['train','val']:
#             if phase == 'train':
#                 scheduler.step()
#                 model.train()
#             else:
#                 model.eval()
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 optimizer.zero_grad()
#
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     _,preds = torch.max(outputs,1)
#                     loss = criterion(outputs,labels)
#
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()
#                 running_loss += loss.item() * inputs.size(0)
#                 running_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = running_loss / dataset_sizes
def train(run, loader, model, criterion, optimizer,scheduler):
    scheduler.step()
    model.train()
    running_loss = 0.0
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def main():
    global args, best_acc
    args = parser.parse_args()

    #cnn
    model = models.resnet50(True)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = nn.DataParallel(model)
    model.cuda()

    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    cudnn.benchmark = True

    #normalization
    normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    # train_dset = CTCdataset(csv_file='/public5/lilab/student/myang/project/lili/GG/slide22.csv',size=224,train=True,transform=trans)
    train_dset = CTCdataset(csv_file=args.train_csv, size=224, train=True,
                            transform=trans)

    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=10, pin_memory=False)
    ####get image
    # inputs, classes = next(iter(train_loader))
    # ####make grid images
    # out = torchvision.utils.make_grid(inputs)
    # imshow(out)
    fconv = open(os.path.join(args.output, 'convergence.csv'), 'w')
    fconv.write('epoch,metric,value\n')
    fconv.close()
    loss_tem = 0.1
    for epoch in range(args.nepochs):
        loss = train(epoch, train_loader, model, criterion, optimizer,exp_lr_scheduler)
        fconv = open(os.path.join(args.output, 'convergence.csv'), 'a')
        fconv.write('{},loss,{}\n'.format(epoch + 1, loss))
        fconv.close()
        print('{},loss,{}\n'.format(epoch + 1, loss))
        if loss < loss_tem:
            loss_tem = loss
            obj = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': loss,
                'optimizer': optimizer.state_dict()
            }
            torch.save(obj, os.path.join(args.output, 'checkpoint_best.pth'))
        else:
            pass

if __name__ == '__main__':
    main()