#-*-coding:utf-8-*-
import torch
import time
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import torch.backends.cudnn as cudnn
import pandas as pd
import sys
sys.path.append('/Data/yangml/Important_script/PycharmProjects/PyTorch_test/Pathology_WSI/LUSC')
from CTCdataset import CTCdataset

class AverageMeter(object):
    '''Computes and stores the average and current value'''

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output,target,topk=(1,)):
    '''Computes the precision @k for the specified values of k'''
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk,1,True,True)
        pred = pred.t()
        correct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0,keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model, criterion,outdir, label="Test"):

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()

    # switch to evaluate mode 转到验证模式
    model.eval()

    preds = []
    result = {}
    result['fnames'] = []
    result['coords_X'] = []
    result['coords_Y'] = []
    with torch.no_grad():
        end = time.time()
        for i, (input, target,filename,coord) in enumerate(val_loader):

            # input = input.cuda(non_blocking=True)
            # target = target.cuda(non_blocking=True)
            filenames_batch = filename
            result['fnames'].extend(filenames_batch)
            print(filename)

            coords_batch = coord
            print(coord)
            result['coords_X'].extend(coords_batch[0].numpy())
            result['coords_Y'].extend(coords_batch[1].numpy())

            input = input.cuda()
            target = target.cuda()

            # compute output 计算输出
            output = model(input)
            loss = criterion(output, target)

            # probability compute
            batch_preds = (F.softmax(output,dim=1)).detach().cpu().numpy()
            preds.append(batch_preds)


            # measure accuracy and record loss 计算准确率并记录 loss
            prec1, = accuracy(output, target)
            # print(prec1)
            # print(prec1[0])
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0].cpu().numpy(), input.size(0))
            # top5.update(prec5[0], input.size(0))

            # measure elapsed time 计算花费时间
            batch_time.update(time.time() - end)
            end = time.time()
            # print(top1.val)

            if i % 100 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1))

        print(' * Prec@1 {top1.avg:.3f} '
              .format(top1=top1))
    preds_cat = np.concatenate(preds,axis=0)
    df_preds = pd.DataFrame(preds_cat,columns=["Normal","Tumor"])
    # print(len(result['fnames']),len(result['coords']))
    df_result = pd.DataFrame(result)
    result = pd.concat([df_preds,df_result],axis=1)
    result.to_csv(outdir+'/'+label+'_Predict_Result.csv',encoding='utf_8_sig')
    return top1.avg

parser = argparse.ArgumentParser(description='CTC tile classifier predict script')
parser.add_argument('--test_csv', type=str, default='filelist', help='path to data file')
parser.add_argument('--output', type=str, default='.', help='name of output directory')
parser.add_argument('--model', type=str, default='', help='path to pretrained model')
parser.add_argument('--batch_size', type=int, default=100, help='how many images to sample per slide (default: 100)')
parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--colornorm', dest='colornorm', action='store_true', default=False, help='if color normaled for patches')
parser.add_argument('--targetImage', type=str, help='Path to the target image if colornorm ')
parser.add_argument('--prefix', type=str, help='Prefix of output result ')

global args
args = parser.parse_args()
model = models.resnet50(True)
model.fc = nn.Linear(model.fc.in_features, 2)
ch = torch.load(args.model)
# model = nn.DataParallel(model)
# model.load_state_dict(ch['state_dict'])
model = nn.DataParallel(model)
model.load_state_dict(ch['state_dict'])
model.cuda()
cudnn.benchmark = True
criterion = nn.CrossEntropyLoss().cuda()

#normalization
normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
trans = transforms.Compose([transforms.ToTensor(), normalize])

#load data
if args.colornorm:
    print('image color normal dataset....')
    dset = CTCdataset(csv_file=args.test_csv, size=224, train=False,
                      transform=trans,colornorm=True,targetImage=args.targetImage)
else:
    dset = CTCdataset(csv_file=args.test_csv, size=224, train=False,
                            transform=trans)
loader = torch.utils.data.DataLoader(
        dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

validate(loader,model,criterion,args.output,args.prefix)
