# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 20:02:59 2020

@author: fatemeh nourian
"""

#my first attempt to train convTasNet - 99/06/04

import torch

from data import AudioDataLoader, AudioDataset
from solverr import Solver
from conv_tasnet import ConvTasNet

# General config
# Task related
train_dir='D:/Amir Kabir University/thesis/conv-tasnet/Conv-TasNet-master/outdata/tr'
#directory including mix.json, s1.json and s2.json
valid_dir='D:/Amir Kabir University/thesis/conv-tasnet/Conv-TasNet-master/outdata/cv'
sample_rate=16000
segment=4
#Segment length (seconds)
cv_maxlen=8
#max audio length (seconds) in cv, to avoid OOM issue
# Network architecture
N=256
#Number of filters in autoencoder
L=20
#Length of the filters in samples (40=5ms at 8kHZ)
B=256
#Number of channels in bottleneck 1 × 1-conv block
H=512
#Number of channels in convolutional blocks
P=3
#Kernel size in convolutional blocks
X=8
#Number of convolutional blocks in each repeat
R=4
#Number of repeats
C=2
#Number of speakers
norm_type='gLN'
#Layer norm type:['gLN', 'cLN', 'BN']
causal=0
#Causal (1) or noncausal(0) training
mask_nonlinear='relu'
#non-linear to generate mask:['relu','softmax']
# Training config
use_cuda=0
#Whether use GPU, default=1
epochs=30
#Number of maximum epochs
half_lr=0
#Halving learning rate when get small improvement
early_stop=0
#Early stop training when no improvement for 10 epochs
max_norm=5
#Gradient norm threshold to clip
# minibatch
shuffle=0
#reshuffle the data at every epoch
batch_size=128
num_workers=4
#Number of workers to generate minibatch
# optimizer
optimizer='adam'
#['sgd', 'adam']
lr=1e-3
#Init learning rate
momentum=0.0
#Momentum for optimizer
l2=0.0
#weight decay (L2 penalty)
# save and load model
save_folder='exp/temp'
#Location to save epoch models
checkpoint=0
#Enables checkpoint saving of model
continue_from=''
#Continue from checkpoint model
model_path='final.pth.tar'
#Location to save best validation model
# logging
print_freq=10
#Frequency of printing training infomation
visdom=0
#Turn on visdom graphing
visdom_epoch=0
#Turn on visdom graphing each epoch
visdom_id='TasNet training'
#Identifier for visdom run

def main(train_dir,batch_size,sample_rate, segment,valid_dir,cv_maxlen,shuffle,num_workers,N, L, B, H, P, X, R, C,norm_type, causal, mask_nonlinear,use_cuda,optimizer,lr,momentum,l2):
     # Construct Solver
    # data
    tr_dataset = AudioDataset(train_dir, batch_size,
                              sample_rate=sample_rate, segment=segment)
    cv_dataset = AudioDataset(valid_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
                              sample_rate=sample_rate,
                              segment=-1, cv_maxlen=cv_maxlen)  # -1 -> use full audio
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=shuffle,
                                num_workers=num_workers)
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=0)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    # model
    model = ConvTasNet(N, L, B, H, P, X, R, C, 
                       norm_type=norm_type, causal=causal,
                       mask_nonlinear=mask_nonlinear)
    print(model)
    if use_cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()
    # optimizer
    if optimizer == 'sgd':
        optimizier = torch.optim.SGD(model.parameters(),
                                     lr=lr,
                                     momentum=momentum,
                                     weight_decay=l2)
    elif optimizer == 'adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=lr,
                                      weight_decay=l2)
    else:
        print("Not support optimizer")
        return

    # solver
    solver = Solver(data, model, optimizer, use_cuda,epochs,half_lr,early_stop,max_norm,save_folder,checkpoint,continue_from,model_path,print_freq,visdom,visdom_epoch,visdom_id)
    solver.train()





#%%
if __name__ == '__main__':
    main(train_dir,batch_size,sample_rate, segment,valid_dir,cv_maxlen,shuffle,num_workers,N, L, B, H, P, X, R, C,norm_type, causal, mask_nonlinear,use_cuda,optimizer,lr,momentum,l2)
    
#    args = parser.parse_args()
#    print(args)
#    main(args)
#    main(train_dir,batch_size,sample_rate, segment,valid_dir,cv_maxlen,shuffle,num_workers,N, L, B, H, P, X, R, C,norm_type, causal, mask_nonlinear,use_cuda,optimizer,lr,momentum,l2)
    
#%%
#fatemeh
#if __name__ == '__main__':
#    model = ConvTasNet(N, L, B, H, P, X, R, C,norm_type=norm_type, causal=causal,mask_nonlinear=mask_nonlinear)
    #criterion=nn.



##print(model)
#optimizier = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=l2)
