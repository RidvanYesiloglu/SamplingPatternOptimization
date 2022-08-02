# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 15:44:51 2019

@author: ridvan
"""
import numpy as np
import matplotlib.pyplot as plt
import os
net_name = 'RS_GAN_LAST'
print('Welcome to the RS GAN with Mask Optimization PSNR Plotter...')


thrding_slope = input('Enter the thresholding layer''s sigmoid''s slope: ')
use_fixed_acc = (input('Indicate whether to use a fixed acceleration rate(1) or not(2): ')==1)
if use_fixed_acc:
    r=input('Enter desired acceleration rate: ')
else:
    acc_loss_coe = input('Enter the coefficient of the loss component about acceleration: ')
lrm=input('Enter starting learning rate of the mask network: ')
lrg=input('Enter starting learning rate of the GAN network: ')
use_cal_reg = (input('Do you want to use a calibration region on the mask? ') == 1)
pr_mask_slope = input('Enter the probmask layer''s sigmoid''s slope: ')
pr_init = input('Do you want uniform initialization(1) or 0.5 initialization(2)? ')
prob_mask_out=None
if use_fixed_acc:
    run_name='r'+str(r)+'_lrm'+str(lrm)[2:]+'_lrg'+str(lrg)[2:]+'_prsl'+str(pr_mask_slope)+'_thrsl'+str(thrding_slope)+'_init'+str(pr_init)
else:
    run_name='alc'+str(acc_loss_coe)[2:]+'_lrm'+str(lrm)[2:]+'_lrg'+str(lrg)[2:]+'_prsl'+str(pr_mask_slope)+'_thrsl'+str(thrding_slope)+'_init'+str(pr_init)
if use_cal_reg:
    cal_reg_rad=input('Enter filled region radius: ')
    run_name = run_name + '_calrad' + str(cal_reg_rad)
else:
    cal_reg_rad = 0
run_no = input('Which run is this? ')
run_name = run_name + '_runno'+str(run_no)
psnr_meth = input('Do you want PSNR calc method 1 or 2? ')

import matplotlib.pyplot as plt
import numpy as np
import os
plt.close('all')
psnr_meth = 1
net_name = 'JRM_GAN_JACC_D2_TWOD'
run_name = 'r4_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_fn_nrml_runno1'
R=4
tr_psnrs_A = np.load(os.path.join('/auto/data2/ridvan/MSK_OPT_PRJ/JR_GANS/',net_name,'R'+str(R), run_name, 'tr_psnr_vs_ep'+str(psnr_meth)+'_A.npy'))
val_psnrs_A = np.load(os.path.join('/auto/data2/ridvan/MSK_OPT_PRJ/JR_GANS/',net_name,'R'+str(R), run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'_A.npy'))
tr_psnrs_B = np.load(os.path.join('/auto/data2/ridvan/MSK_OPT_PRJ/JR_GANS/',net_name,'R'+str(R), run_name, 'tr_psnr_vs_ep'+str(psnr_meth)+'_B.npy'))
val_psnrs_B = np.load(os.path.join('/auto/data2/ridvan/MSK_OPT_PRJ/JR_GANS/',net_name,'R'+str(R), run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'_B.npy'))
plt.figure()
plt.plot(tr_psnrs_B, label="Aver on Train Set, T1")
plt.plot(val_psnrs_B, label="Aver on Val Set, T1")
plt.plot(tr_psnrs_A, label="Aver on Train Set, T2")
plt.plot(val_psnrs_A, label="Aver on Val Set, T2")
plt.title('PSNR vs Epoch (JRMGAN with Conf: %s)' % run_name)
plt.legend(loc="best")
plt.ylabel('PSNR')
plt.xlabel('Epoch No')
opt_ep = np.argmax(val_psnrs_A+val_psnrs_B)+1
#opt_ep = np.argmax(val_psnrs_A+val_psnrs_B)+1
print('Use epoch ' + str(opt_ep) + ' which gives a sum of ' + '{:.3f}'.format(val_psnrs_A[opt_ep-1]+val_psnrs_B[opt_ep-1]) + ' (A: ' + '{:.3f}'.format(val_psnrs_A[opt_ep-1]) + ', B: ' + '{:.3f}'.format(val_psnrs_B[opt_ep-1]) + ')')

plt.savefig('/auto/k2/ridvan/Summaries/JR_GANS/JRMJACCD2TWOD/R' + str(R) + '/JRCNS/' + run_name + '/PSNRS_' + str(psnr_meth) + '.png', bbox_inches='tight')
plt.show()

#import numpy as np
#import matplotlib.pyplot as plt
#tr_psnrs = np.load('tr_psnr_vs_ep1.npy')
#val_psnrs = np.load('val_psnr_vs_ep1.npy')
#plt.plot(tr_psnrs, label="Aver on Train Set")
#plt.plot(val_psnrs, label="Aver on Val Set")
#plt.title('PSNR vs Epoch (rs GAN) (R=2)')
#plt.legend(loc="upper left")
#plt.ylabel('PSNR')
#plt.xlabel('Epoch No')
#plt.show()