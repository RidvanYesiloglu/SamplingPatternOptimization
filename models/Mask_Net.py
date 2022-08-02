# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 17:01:40 2019

@author: ridvan
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models import layers
import matplotlib.pyplot as plt
import h5py

class Mask_Net(nn.Module):
    def __init__(self, opt):
        super(Mask_Net,self).__init__()
        self.use_fixed_acc = opt.use_fixed_acc
        self.concat = layers.ConcatenateZero()
        self.fft = layers.FFT(opt)
        self.probmask = layers.ProbMask(opt)
        if self.use_fixed_acc:
            self.rescaleprobmask = layers.RescaleProbMap(opt)
        self.randommask = layers.RandomMask(opt)
        self.thresholdrandommask = layers.ThresholdRandomMask(opt)
        #ksp_mean = torch.from_numpy(np.load('ksp_mean.npy')).type(torch.FloatTensor).cuda(opt.gpu_id)
        #self.undersample = layers.UnderSample(ksp_mean)
        self.undersample = layers.UnderSample(opt.gpu_id) #
        print('Data loaded.')
        self.ifft = layers.IFFT(opt)
        self.use_jusr = opt.use_joint_us
        
    def forward(self,x):
        #self.x1 = self.concat(x) #magn im
    
#        self.x1 = x.permute(0,1,3,4,2) #45 deg
#        self.x1.retain_grad()
        self.x2 = self.fft(x)
#        self.x2.retain_grad()
        self.pr_mask = self.probmask(self.x2)
        self.pr_mask.retain_grad()
        if self.use_fixed_acc:
            if self.use_jusr:
                self.pr_mask2 = self.rescaleprobmask(self.pr_mask)
            else:
                self.pr_mask2_A = self.rescaleprobmask(self.pr_mask.narrow(0,0,1))
                self.pr_mask2_A.retain_grad()
                self.pr_mask2_B = self.rescaleprobmask(self.pr_mask.narrow(0,1,1))
                self.pr_mask2_B.retain_grad()
                self.pr_mask2 = torch.cat((self.pr_mask2_A,self.pr_mask2_B),dim=0)
            self.pr_mask2.retain_grad()
            self.rnd_thr = self.randommask(self.pr_mask2) 
            self.rnd_thr.retain_grad()
            self.stacked0 = torch.stack((self.pr_mask2, self.rnd_thr),dim=0)  
        else:
            self.rnd_thr = self.randommask(self.pr_mask) 
            self.rnd_thr.retain_grad()
            self.stacked0 = torch.stack((self.pr_mask, self.rnd_thr),dim=0)  
        self.stacked0.retain_grad()
        self.lst_pr_mask = self.thresholdrandommask(self.stacked0)        
        self.lst_pr_mask.retain_grad()
        self.x3 = self.x2 * self.lst_pr_mask[...,0]
#        self.cat_lst_pr_mask = torch.cat((self.lst_pr_mask,self.lst_pr_mask),dim=-1)
#        self.cat_lst_pr_mask.retain_grad()
#        self.x3 = self.undersample(self.x2, self.cat_lst_pr_mask)
        self.x3.retain_grad()
        self.us_im = self.ifft(self.x3)
        self.rsh_us_im = self.us_im.permute(0,1,2,5,3,4).reshape(-1,30,192,88)
        self.rsh_us_im.retain_grad()
        return self.rsh_us_im
        
    def find_masks(self,x):
        y = self.fft(x)
        pr_mask = self.probmask(y)
        if self.use_fixed_acc:
            pr_mask_rscl = self.rescaleprobmask(pr_mask)
            rnd_thr = self.randommask(pr_mask_rscl)
            bin_mask = self.thresholdrandommask(torch.stack((pr_mask_rscl, rnd_thr),dim=0))
            return pr_mask,pr_mask_rscl,bin_mask
           # return bin_mask,bin_mask,bin_mask
        else:
            rnd_thr = self.randommask(pr_mask)
            bin_mask = self.thresholdrandommask(torch.stack((pr_mask, rnd_thr),dim=0))
            return bin_mask,bin_mask

def init_weights(m):
    if type(m) == layers.ProbMask:
        m.mult = m.logit_slope_random_uniform()
        
        
"""
print('loading data...')
# load train data:
with h5py.File('../../mrs_gan/T1_1_multi_synth_recon_train.mat', 'r') as f:
    trn_data = f['data_fs'].value
    trn_data_shape = trn_data.shape
    trn_data = np.reshape(np.transpose(trn_data, (0, 2, 1)),(trn_data_shape[0],trn_data_shape[2],trn_data_shape[1],1))
with h5py.File('../../mrs_gan/T1_1_multi_synth_recon_val.mat', 'r') as f:
    val_data = f['data_fs'].value
    val_data_shape = val_data.shape
    val_data = np.reshape(np.transpose(val_data, (0, 2, 1)),(val_data_shape[0],val_data_shape[2],val_data_shape[1],1))
vol_size = trn_data.shape[1:-1]


inp = torch.from_numpy(trn_data[:,:,:,0]).cuda(0, async=True)

sh=(2726,256,152,2)

print('inp shape', inp.shape)
net = Mask_Net(sh)
for parameter in net.parameters():
    print('prmt', parameter)
print('for endd')
#net.apply(init_weights)
outp = net(torch.from_numpy(inp).float())

#print(inp)
print('******')
#print(outp)


"""