# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:51:39 2019

@author: ridvan
"""
import torch
from operator import mul
import numpy as np        
import time
import scipy.io
import functools
#from sklearn.cluster import KMeans
def cal_reg_cns(inp_sh,cal_reg_dia):
    x, y = np.indices((inp_sh[3], inp_sh[4]))
    return torch.reshape(torch.from_numpy(np.fft.fftshift((np.hypot(inp_sh[3]/2 - x, inp_sh[4]/2-y) > cal_reg_dia).astype(int))),inp_sh).type(torch.FloatTensor)


class ConcatenateZero(torch.nn.Module):
    def __init__(self):
        super(ConcatenateZero, self).__init__()

    def forward(self, x):
        return torch.stack((x,x*0),-1)
#
class FFT(torch.nn.Module):
    def __init__(self, opt):
        super(FFT, self).__init__()
        #assert (opt.inp_sh[-1] == 2), 'The FFT layer should take an input with shape (bs,?,?,2)'
        self.norm = opt.fftnorm

    def forward(self, x):
        if self.norm:
            return torch.fft.fft2(x[...,0]+1j*x[...,1], norm="ortho")
        else:
            return torch.fft.fft2(x[...,0]+1j*x[...,1])

class ProbMask(torch.nn.Module):
    def __init__(self, opt):
        super(ProbMask, self).__init__()
        if opt.lnbl_pr_sl:
            self.slope = torch.nn.Parameter(data=torch.tensor(opt.pr_mask_slope).float().requires_grad_().cuda(opt.gpu_id, async=True),requires_grad=True)
        else:
            self.slope = torch.tensor(opt.pr_mask_slope).float().cuda(opt.gpu_id, async=True)
        lst = list(opt.inp_sh)
        lst[-1] = 1
        input_shape_h = tuple(lst)
        eps = 0.01
        if opt.pr_init == 1:
            rand_x = (1-2*eps) * torch.rand(input_shape_h) + eps
        elif opt.pr_init == 2:
            rand_x = torch.ones(input_shape_h)*0.5
        elif opt.pr_init == 3:
            rand_x = torch.ones(input_shape_h)*0.1
        elif opt.pr_init == 4:
            rand_x = torch.ones(input_shape_h) - 0.01
        #self.mult = torch.nn.Parameter(data = (-torch.log(1. / rand_x - 1.) / opt.pr_mask_slope).requires_grad_().cuda(opt.gpu_id, async=True),requires_grad=True)
        
        if opt.pr_init == 5:
            cns_msk = scipy.io.loadmat('/auto/k2/ridvan/gaussmaskr16.mat')
            self.mult = torch.from_numpy(np.transpose(cns_msk['msk'],(2,0,1)).reshape(input_shape_h)).float().cuda(opt.gpu_id, async=True).requires_grad_()
        
        
        self.prob_mask_out = opt.prob_mask_out
        self.use_cal_reg = opt.use_cal_reg
        if self.use_cal_reg:
            self.msk_cns = cal_reg_cns(input_shape_h, opt.cal_reg_rad).cuda(opt.gpu_id, async=True)

    def forward(self, x):
        if self.prob_mask_out is None:
            if self.use_cal_reg:
                return self.msk_cns*torch.sigmoid(self.slope * self.mult)+(1-self.msk_cns)
            else:
                return torch.sigmoid(self.slope * self.mult)
        else:
            return self.prob_mask_out

class RescaleProbMap(torch.nn.Module):
    def __init__(self, opt):
        super(RescaleProbMap, self).__init__()
        inp_sh = list(opt.inp_sh)
        inp_sh[-1]=1
        no_of_pnts = functools.reduce(mul, inp_sh, 1)
        self.opt = opt
        self.use_cal_reg = opt.use_cal_reg
        if self.use_cal_reg:
            self.msk_cns = cal_reg_cns(inp_sh, opt.cal_reg_rad).cuda(opt.gpu_id, async=True)         
            self.ur_out_cal = (no_of_pnts*opt.ur-(1-self.msk_cns).sum())/no_of_pnts
    def forward(self, x):
        if self.use_cal_reg:
            xbar_out = torch.mean(x*self.msk_cns)
            r_out = self.ur_out_cal / xbar_out
            beta_out = (1-self.ur_out_cal) / (1-xbar_out)
            le = float(r_out<=1)
            return self.msk_cns*(le * x * r_out + (1-le) * (1 - (1 - x) * beta_out))+(1-self.msk_cns)
        else:
            xbar = torch.mean(x)
            ratio = self.opt.ur / xbar
            beta = (1-self.opt.ur) / (1-xbar)
            le = float(ratio<=1)
            return le * x * ratio + (1-le) * (1 - (1 - x) * beta)
class RandomMask(torch.nn.Module):
    def __init__(self, opt):
        super(RandomMask, self).__init__()
        self.gpu_id = opt.gpu_id
    def forward(self, x):
        input_shape = x.shape
        threshs = torch.rand(input_shape).cuda(self.gpu_id, async=True)
        return (0*x) + threshs

class ThresholdRandomMask(torch.nn.Module):
    def __init__(self, opt):
        super(ThresholdRandomMask, self).__init__()
        self.opt = opt
        inp_sh = list(opt.inp_sh)
        inp_sh[-1]=1
        if opt.lnbl_th_sl:
            self.slope = torch.nn.Parameter(data=torch.tensor(opt.thrding_slope).float().cuda(opt.gpu_id, async=True).requires_grad_(),requires_grad=True)
        else:
            self.slope = torch.tensor(opt.thrding_slope).float().cuda(opt.gpu_id, async=True)
        self.use_cal_reg = opt.use_cal_reg
        self.use_hrd_thr = opt.use_hrd_thr
        if self.use_cal_reg:
            self.msk_cns = cal_reg_cns(inp_sh, opt.cal_reg_rad).cuda(opt.gpu_id, async=True)
        
    def forward(self, x):
        #return self.cns_mask
        inputs = x[0]
        thresh = x[1]
        if (not self.use_hrd_thr):
            if self.use_cal_reg:
                return self.msk_cns*torch.nn.functional.sigmoid(self.slope * (inputs-thresh))+(1-self.msk_cns)
            else:
                return torch.nn.functional.sigmoid(self.opt.thrding_slope * (inputs-thresh))
        else:
            return torch.gt(inputs, thresh).type(torch.cuda.FloatTensor)

class UnderSample(torch.nn.Module):
    def __init__(self,gpu_id):
        super(UnderSample, self).__init__()
        #self.ksp_mean = tr_ksp_data
#        self.tr_ksp_data = tr_ksp_data
#        self.gpu_id = gpu_id
#        self.vars_mask = vars_mask.reshape((2,256,256,1))
#        self.vars_mask = np.concatenate((self.vars_mask,self.vars_mask),-1)
#        self.bin_mask = self.vars_mask
#        tr_ksp_data_mskd = self.tr_ksp_data[np.where(self.bin_mask, self.tr_ksp_data, self.tr_ksp_data.min()-1) != (self.tr_ksp_data.min()-1)].reshape((self.tr_ksp_data.shape[0],-1))
#        self.n_clusters = 10
#        self.kmeans = KMeans(n_clusters=self.n_clusters,random_state=0).fit(tr_ksp_data_mskd)
#        self.kspmeans = []
#        for i in range(self.n_clusters):
#            self.kspmeans.append(torch.from_numpy(self.tr_ksp_data[self.kmeans.labels_==i].mean(0)).reshape((2,1,1,256,256,2)).type(torch.FloatTensor).cuda(self.gpu_id))
            
    def forward(self, x, y):        
#        curr_mask = (y.data.cpu().numpy().squeeze())
#        #print(curr_mask.shape, self.bin_mask.shape, (y.data.cpu().numpy().squeeze() > 0.5).shape)
#        if ((curr_mask != self.bin_mask)).sum() != 0:
#            print('Mask changed.',((curr_mask != self.bin_mask)).sum(), curr_mask.sum(), self.bin_mask.sum())
#            self.bin_mask = curr_mask
#            tr_ksp_data_mskd = self.tr_ksp_data[np.where(self.bin_mask, self.tr_ksp_data, self.tr_ksp_data.min()-1) != (self.tr_ksp_data.min()-1)].reshape((self.tr_ksp_data.shape[0],-1))
#            self.kmeans = KMeans(n_clusters=self.n_clusters,random_state=0).fit(tr_ksp_data_mskd)
#            self.kspmeans = [None]*self.n_clusters
#            
#        curr = x.data.cpu().numpy().squeeze(2)
#        curr = curr[np.where(self.bin_mask.reshape((2,1,256,256,2)), curr, self.tr_ksp_data.min()-1) != (self.tr_ksp_data.min()-1)].reshape(curr.shape[1],-1)
#        #print(self.tr_ksp_data[kmeans.labels_==kmeans.predict(curr)].shape)
#        #print(curr.shape)        
#        for i in range(curr.shape[0]):
#            pred = self.kmeans.predict(curr)
#            if self.kspmeans[] is None:
#                print('None')
#                self.kspmeans[i] = torch.from_numpy(self.tr_ksp_data[self.kmeans.labels_==].mean(0)).reshape((2,1,1,256,256,2)).type(torch.FloatTensor).cuda(self.gpu_id)                
#            if i == 0:
#                ksp_mean = self.kspmeans[i]
#            else:
#                ksp_mean = torch.cat((ksp_mean,self.kspmeans[i]),dim=1)     
        #print('shape ', ksp_mean.shape, (x[..., 0] * y[..., 0]).shape, y[..., 0].shape)
#        print('shape_asil', x[..., 0].shape)

#        print('shape_asil 2', y[..., 0].shape)
    
        k_space_r = (x[..., 0] * y[..., 0]) #+ (ksp_mean[...,0] * (1-y[..., 0]))
        k_space_i = (x[..., 1] * y[..., 1]) #+ (ksp_mean[...,1] * (1-y[..., 1]))
        
        k_space = torch.stack((k_space_r, k_space_i), dim = -1)
        k_space.retain_grad()
        k_space = k_space.float()
        k_space.retain_grad()
        return k_space


class IFFT(torch.nn.Module):
    def __init__(self, opt):
        super(IFFT, self).__init__()
        self.norm=opt.ifftnorm

    def forward(self, x):
        if not self.norm:
            cmplx = torch.fft.ifft2(x)
            #return torch.ifft(x,2).permute(0,1,4,2,3)
        else:
            cmplx = torch.fft.ifft2(x,norm="ortho")
        return torch.stack((cmplx.real,cmplx.imag),-1)
            #return torch.ifft(x,2, normalized=True).permute(0,1,4,2,3)
 
