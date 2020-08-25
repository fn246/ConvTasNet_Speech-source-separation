# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 20:02:59 2020

@author: fatemeh nourian
"""

#my first attempt to train convTasNet - 99/06/04

import torch

from data import AudioDataLoader, AudioDataset
from solver import Solver
from conv_tasnet import ConvTasNet

# General config
# Task related
train_dir
valid_dir
sample_rate
segment
cv_maxlen
# Network architecture
N
L
B
H
P
X
R
C
norm_type
causal
mask_nonlinear
# Training config
use_cuda
epochs
half_lr
early_stop
max_norm
# minibatch
shuffle
batch_size
num_workers
# optimizer
optimizer
lr
momentum
l2
# save and load model
save_folder
checkpoint
continue_from
model_path
# logging
print_freq
visdom
visdom_epoch
visdom_id

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


if __name__ == '__main__':
#    args = parser.parse_args()
#    print(args)
#    main(args)
    main(train_dir,batch_size,sample_rate, segment,valid_dir,cv_maxlen,shuffle,num_workers,N, L, B, H, P, X, R, C,norm_type, causal, mask_nonlinear,use_cuda,optimizer,lr,momentum,l2)