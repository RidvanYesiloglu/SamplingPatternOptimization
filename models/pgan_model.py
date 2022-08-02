import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
from torchvision import models

#from Mask_Net 
from models import Mask_Net
import matplotlib.pyplot as plt

from ranger import Ranger  # this is from ranger.py
from ranger import RangerVA  # this is from ranger913A.py
from ranger import RangerQH  # this is from rangerqh.py

class pGAN(BaseModel):
    def name(self):
        return 'pGAN'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        #self.mask_net = Mask_Net((1,1,256,256,2),pmask_slope=5, pmask_init=None, sparsity=0.25, sample_slope=10, prob_mask_out=torch.from_numpy(np.load('maskP.npy')).cuda(1, async=True))
        self.mask_net = Mask_Net.Mask_Net(opt)
#        print('Mask_Net parameters are below:')
#        for param in self.mask_net.parameters():
#            print(param)
        self.netGT1 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.opt.gpu_id)
        self.netGT2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.opt.gpu_id)
        self.netGPD = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.norm, not opt.no_dropout, opt.init_type, self.opt.gpu_id)
        self.vgg=VGG16().cuda(self.opt.gpu_id)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netDT1 = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.opt.gpu_id)
            self.netDT2 = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.opt.gpu_id)
            self.netDPD = networks.define_D(opt.output_nc, opt.ndf,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.opt.gpu_id)
                                                                  
        if not self.isTrain or opt.continue_train:
            self.load_network(self.mask_net,'M',opt.which_epoch)
            self.load_network(self.netGT1, 'GT1', opt.which_epoch)
            self.load_network(self.netGT2, 'GT2', opt.which_epoch)
            self.load_network(self.netGPD, 'GPD', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netDT1, 'DT1', opt.which_epoch)
                self.load_network(self.netDT2, 'DT2', opt.which_epoch)
                self.load_network(self.netDPD, 'DPD', opt.which_epoch)
        if self.isTrain and opt.use_dt_cns:
            self.netGT1.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/'+opt.net_name+'/'+opt.run_name_wocns+'/'+str(opt.epfordtcns)+'_net_GT1.pth'))
            self.netGT2.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/'+opt.net_name+'/'+opt.run_name_wocns+'/'+str(opt.epfordtcns)+'_net_GT2.pth'))
            self.netGPD.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/'+opt.net_name+'/'+opt.run_name_wocns+'/'+str(opt.epfordtcns)+'_net_GPD.pth'))
            self.netDT1.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/'+opt.net_name+'/'+opt.run_name_wocns+'/'+str(opt.epfordtcns)+'_net_DT1.pth'))
            self.netDT2.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/'+opt.net_name+'/'+opt.run_name_wocns+'/'+str(opt.epfordtcns)+'_net_DT2.pth'))
            self.netDPD.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/'+opt.net_name+'/'+opt.run_name_wocns+'/'+str(opt.epfordtcns)+'_net_DPD.pth'))
            self.mask_net.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/'+opt.net_name+'/'+opt.run_name_wocns+'/'+str(opt.epfordtcns)+'_net_M.pth'))
            print('DTcnsload performed from:')
            print('/auto/data2/ridvan/MSK_OPT_PRJ/'+opt.net_name+'/'+opt.run_name_wocns+'/'+str(opt.epfordtcns)+'_net_....pth')
            
#        self.netGA.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRMTWOGTWOD/R64/r64_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss10000_fn_nrml_runnovgg3000_jusr/63_net_Ga.pth'))
#        self.netGB.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRMTWOGTWOD/R64/r64_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss10000_fn_nrml_runnovgg3000_jusr/63_net_Gb.pth'))
#        self.netDA.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRMTWOGTWOD/R64/r64_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss10000_fn_nrml_runnovgg3000_jusr/63_net_Da.pth'))
#        self.netDB.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRMTWOGTWOD/R64/r64_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss10000_fn_nrml_runnovgg3000_jusr/63_net_Db.pth'))
#        self.mask_net.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRMTWOGTWOD/R64/r64_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss10000_fn_nrml_runnovgg3000_jusr/63_net_M.pth'))
        
#        self.netG.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRM_GAN_JACC_D2_TWOD/R2/r2_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss1000_fn_nrml_runno1/50_net_G.pth'))
#        self.netDA.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRM_GAN_JACC_D2_TWOD/R2/r2_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss1000_fn_nrml_runno1/50_net_Da.pth'))
#        self.netDB.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRM_GAN_JACC_D2_TWOD/R2/r2_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss1000_fn_nrml_runno1/50_net_Db.pth'))
#        self.mask_net.load_state_dict(torch.load('/auto/data2/ridvan/MSK_OPT_PRJ/JRM_GAN_JACC_D2_TWOD/R2/r2_lrm0002_lrg0002_prsl100_thrsl100_init2_opt1_multloss1000_fn_nrml_runno1/50_net_M.pth'))
      
        self.fake_AB_pool = ImagePool(opt.pool_size)
        # define loss functions
        self.criterionGAN = networks.GANLoss(self.opt.gpu_id,use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
        self.criterionL1 = torch.nn.L1Loss()
        
        if self.isTrain:
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            if opt.opti_tech == 1:
                #self.optimizer_M = torch.optim.Adam(self.mask_net.parameters(),lr=opt.lrm, betas=(opt.beta1, 0.999))
                self.optimizer_GT1 = torch.optim.Adam(self.netGT1.parameters(),lr=opt.lrg, betas=(opt.beta1, 0.999))
                self.optimizer_GT2 = torch.optim.Adam(self.netGT2.parameters(),lr=opt.lrg, betas=(opt.beta1, 0.999))
                self.optimizer_GPD = torch.optim.Adam(self.netGPD.parameters(),lr=opt.lrg, betas=(opt.beta1, 0.999))
                self.optimizer_DT1 = torch.optim.Adam(self.netDT1.parameters(),lr=opt.lrg, betas=(opt.beta1, 0.999))
                self.optimizer_DT2 = torch.optim.Adam(self.netDT2.parameters(),lr=opt.lrg, betas=(opt.beta1, 0.999))
                self.optimizer_DPD = torch.optim.Adam(self.netDPD.parameters(),lr=opt.lrg, betas=(opt.beta1, 0.999))
            elif opt.opti_tech == 2:
                #self.optimizer_M = Ranger(self.mask_net.parameters())
                self.optimizer_GA = Ranger(self.netGA.parameters())
                self.optimizer_GB = Ranger(self.netGB.parameters())
                self.optimizer_D = Ranger(self.netD.parameters())
            #self.optimizers.append(self.optimizer_M)   
            self.optimizers.append(self.optimizer_GT1)
            self.optimizers.append(self.optimizer_GT2)
            self.optimizers.append(self.optimizer_GPD)
            self.optimizers.append(self.optimizer_DT1)
            self.optimizers.append(self.optimizer_DT2)
            self.optimizers.append(self.optimizer_DPD)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        #networks.print_network(self.netGT1)
#        if self.isTrain:
#            networks.print_network(self.netDA)
#        print('-----------------------------------------------')
        for param in self.mask_net.parameters():
            print(param.grad)

    def set_input(self, input):
        input_A = input['A'].to(torch.float32).contiguous()
        #input_B = input['B'].to(torch.float32).contiguous()
        self.input_A = input_A.cuda(self.opt.gpu_id, async=True)
        self.maps = input['mps'].to(torch.float32).contiguous().squeeze().cuda(self.opt.gpu_id, async=True)
        self.ref = input['ref'].to(torch.float32).contiguous().cuda(self.opt.gpu_id, async=True)
        self.ref = torch.sqrt(torch.pow(self.ref[...,0], 2)+torch.pow(self.ref[...,1], 2))
        #input_B = input_B.cuda(self.opt.gpu_id, async=True)
        #self.input_A = torch.reshape(input_A,(-1,1,192,88))
        #self.input_B = torch.reshape(input_B,(-1,1,192,88))
        self.image_paths = input['A_paths']

    def forward(self):
        self.real_A = self.input_A.permute(0,1,2,5,3,4).reshape(-1,30,192,88)
        self.concAB = self.mask_net(self.input_A)
        #self.concAB = torch.cat((self.concAB.narrow(0,0,1).squeeze(0), self.concAB.narrow(0,1,1).squeeze(0)),1)
        self.fake_T1 = self.netGT1(self.concAB)
        self.fake_T2 = self.netGT2(self.concAB)
        self.fake_PD = self.netGPD(self.concAB)
        
        if self.opt.use_dt_cns:
            pr_mask_AB, res_mask_AB, thr_mask_AB = self.mask_net.find_masks(self.input_A)
            bin_mask_T1 = thr_mask_AB.narrow(1,0,1).squeeze(0)
            bin_mask_T2 = thr_mask_AB.narrow(1,1,1).squeeze(0)
            bin_mask_PD = thr_mask_AB.narrow(1,2,1).squeeze(0)
            bin_mask_all = torch.stack((bin_mask_T1,bin_mask_T2,bin_mask_PD),1)
            
            real_A_k = torch.fft(self.input_A, 2, normalized=True)
            fake_all = torch.stack((self.fake_T1,self.fake_T2,self.fake_PD),1)
            fake_all = fake_all.reshape(1,3,5,2,192,88).permute(0,1,2,4,5,3)
            fake_all.retain_grad()
            fake_all = torch.fft(fake_all, 2, normalized=True)
            
            
            
            self.fake_A_k_cons = fake_all
            self.fake_A_k_cons = (self.fake_A_k_cons * (1-bin_mask_all)) + (real_A_k * bin_mask_all)
            self.fake_A_cons = torch.ifft(self.fake_A_k_cons, 2, normalized=True).permute(0,1,2,5,3,4).reshape(1,30,192,88)
            self.fake_A_cons.retain_grad()            
            self.fake_T1_cons = self.fake_A_cons.narrow(1,0,10)
            self.fake_T1_cons.retain_grad()    
            self.fake_T2_cons = self.fake_A_cons.narrow(1,10,10)
            self.fake_T2_cons.retain_grad()    
            self.fake_PD_cons = self.fake_A_cons.narrow(1,20,10)
            self.fake_PD_cons.retain_grad()    
            
            

    def test(self):
        self.real_A = self.input_A.permute(0,1,2,5,3,4).reshape(-1,30,192,88)
        self.concAB = self.mask_net(self.input_A)
        #self.concAB = torch.cat((self.concAB.narrow(0,0,1).squeeze(0), self.concAB.narrow(0,1,1).squeeze(0)),1)
        self.fake_T1 = self.netGT1(self.concAB)
        self.fake_T2 = self.netGT2(self.concAB)
        self.fake_PD = self.netGPD(self.concAB)
        
        self.fake_T1.detach()
        self.fake_T2.detach()
        self.fake_PD.detach()

        pr_mask_AB, res_mask_AB, thr_mask_AB = self.mask_net.find_masks(self.input_A)
        pr_mask_T1 = pr_mask_AB.narrow(1,0,1).squeeze(0)
        pr_mask_T2 = pr_mask_AB.narrow(1,1,1).squeeze(0)
        pr_mask_PD = pr_mask_AB.narrow(1,2,1).squeeze(0)
        res_mask_T1 = res_mask_AB.narrow(1,0,1).squeeze(0)
        res_mask_T2 = res_mask_AB.narrow(1,1,1).squeeze(0)
        res_mask_PD = res_mask_AB.narrow(1,2,1).squeeze(0)
        bin_mask_T1 = thr_mask_AB.narrow(1,0,1).squeeze(0)
        bin_mask_T2 = thr_mask_AB.narrow(1,1,1).squeeze(0)
        bin_mask_PD = thr_mask_AB.narrow(1,2,1).squeeze(0)
        bin_mask_all = torch.stack((bin_mask_T1,bin_mask_T2,bin_mask_PD),1)
        
        real_A_k = torch.fft.fft2(self.input_A[...,0]+1j*self.input_A[...,1], norm="ortho")
        real_A_k = torch.stack((real_A_k.real, real_A_k.imag),-1)
        fake_all = torch.stack((self.fake_T1,self.fake_T2,self.fake_PD),1)
        fake_all = fake_all.reshape(1,3,5,2,192,88).permute(0,1,2,4,5,3)
        fake_all.retain_grad()
        fake_all = torch.fft.fft2(fake_all[...,0]+1j*fake_all[...,1], norm="ortho")
        fake_all = torch.stack((fake_all.real, fake_all.imag),-1)
        
        
        
        self.fake_A_k_cons = fake_all
        self.fake_A_k_cons = (self.fake_A_k_cons * (1-bin_mask_all)) + (real_A_k * bin_mask_all)
        self.fake_A_cons = torch.fft.ifft2(self.fake_A_k_cons[...,0]+1j*self.fake_A_k_cons[...,1], norm="ortho")
        self.fake_A_cons = torch.stack((self.fake_A_cons.real, self.fake_A_cons.imag),-1).permute(0,1,2,5,3,4).reshape(1,30,192,88)
        self.fake_A_cons.detach()            
        self.fake_T1_cons = self.fake_A_cons.narrow(1,0,10)
        self.fake_T1_cons.detach()
        self.fake_T2_cons = self.fake_A_cons.narrow(1,10,10)
        self.fake_T2_cons.detach()    
        self.fake_PD_cons = self.fake_A_cons.narrow(1,20,10)
        self.fake_PD_cons.detach()  
            
#        print('real shape', self.real_A.shape)
#        print('fake shape', self.fake_A.shape)
#        print('fake_c shape', self.fake_A_cons.shape)
        return (self.real_A.narrow(1,0,10), self.fake_T1, self.fake_T1_cons, [pr_mask_T1,res_mask_T1,bin_mask_T1]), (self.real_A.narrow(1,10,10), self.fake_T2, self.fake_T2_cons, [pr_mask_T2,res_mask_T2,bin_mask_T2]),(self.real_A.narrow(1,20,10), self.fake_PD, self.fake_PD_cons, [pr_mask_PD,res_mask_PD,bin_mask_PD])
    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_DT1(self):
        # Fake
        if self.opt.use_dt_cns: 
        # stop backprop to the generator by detaching fake_B
            pred_fake_T1 = self.netDT1(self.fake_T1_cons.detach())        
        else:
            pred_fake_T1 = self.netDT1(self.fake_T1.detach())
        self.loss_D_fake_T1 = self.criterionGAN(pred_fake_T1, False)

        # Real
        
        pred_real_T1 = self.netDT1(self.real_A.narrow(1,0,10).detach())
        self.loss_D_real_T1 = self.criterionGAN(pred_real_T1, True)

        # Combined loss
        self.loss_D_T1 = (self.loss_D_fake_T1 + self.loss_D_real_T1) * 0.5*self.opt.lambda_adv

        self.loss_D_T1.backward()

    def backward_DT2(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.opt.use_dt_cns:         
            pred_fake_T2 = self.netDT2(self.fake_T2_cons.detach())        
        else:
            pred_fake_T2 = self.netDT2(self.fake_T2.detach())        
        self.loss_D_fake_T2 = self.criterionGAN(pred_fake_T2, False)

        # Real
        
        pred_real_T2 = self.netDT2(self.real_A.narrow(1,10,10).detach())
        self.loss_D_real_T2 = self.criterionGAN(pred_real_T2, True)

        # Combined loss
        self.loss_D_T2 = (self.loss_D_fake_T2 + self.loss_D_real_T2) * 0.5*self.opt.lambda_adv

        self.loss_D_T2.backward()  
    
    def backward_DPD(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.opt.use_dt_cns:         
            pred_fake_PD = self.netDPD(self.fake_PD_cons.detach())        
        else:
            pred_fake_PD = self.netDPD(self.fake_PD.detach())        
        self.loss_D_fake_PD = self.criterionGAN(pred_fake_PD, False)

        # Real
        
        pred_real_PD = self.netDPD(self.real_A.narrow(1,20,10).detach())
        self.loss_D_real_PD = self.criterionGAN(pred_real_PD, True)

        # Combined loss
        self.loss_D_PD = (self.loss_D_fake_PD + self.loss_D_real_PD) * 0.5*self.opt.lambda_adv

        self.loss_D_PD.backward()  
        
    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.use_dt_cns:  
            pred_fake_T1 = self.netDT1(self.fake_T1_cons)
        else:
            pred_fake_T1 = self.netDT1(self.fake_T1)
        self.loss_G_GAN_T1 = self.criterionGAN(pred_fake_T1, True)*self.opt.lambda_adv
        if self.opt.use_dt_cns:  
            pred_fake_T2 = self.netDT2(self.fake_T2_cons)
        else:
            pred_fake_T2 = self.netDT2(self.fake_T2)
        self.loss_G_GAN_T2 = self.criterionGAN(pred_fake_T2, True)*self.opt.lambda_adv
        if self.opt.use_dt_cns:  
            pred_fake_PD = self.netDPD(self.fake_PD_cons)
        else:
            pred_fake_PD = self.netDPD(self.fake_PD)
        self.loss_G_GAN_PD = self.criterionGAN(pred_fake_PD, True)*self.opt.lambda_adv
        
        #Perceptual loss
        #self.real_magn_B = (self.real_B.narrow(1,0,1).pow(2)+self.real_B.narrow(1,1,1).pow(2)).pow(0.5)
        self.VGG_real_T1=self.vgg(self.real_A.narrow(1,0,10).transpose(0,1).expand([10,3,192,88]))[0]
        if self.opt.use_dt_cns:
            #self.fake_c_magn_B = (self.fake_B_cons.narrow(1,0,1).pow(2)+self.fake_B_cons.narrow(1,1,1).pow(2)).pow(0.5)
            #self.fake_c_magn_B.retain_grad()
            self.VGG_fake_T1=self.vgg(self.fake_T1_cons.transpose(0,1).expand([10,3,192,88]))[0]
        else:
            #self.fake_magn_B = (self.fake_B.narrow(1,0,1).pow(2)+self.fake_B.narrow(1,1,1).pow(2)).pow(0.5)
            #self.fake_magn_B.retain_grad()
            self.VGG_fake_T1=self.vgg(self.fake_T1.transpose(0,1).expand([10,3,192,88]))[0]
        self.VGG_loss_T1=self.criterionL1(self.VGG_fake_T1,self.VGG_real_T1)* self.opt.lambda_vgg        
        #Perceptual loss
        #self.real_magn_A =  (self.real_A.narrow(1,0,1).pow(2)+self.real_A.narrow(1,1,1).pow(2)).pow(0.5)
        self.VGG_real_T2=self.vgg(self.real_A.narrow(1,10,10).transpose(0,1).expand([10,3,192,88]))[0]
        if self.opt.use_dt_cns: 
            #self.fake_c_magn_A = (self.fake_A_cons.narrow(1,0,1).pow(2)+self.fake_A_cons.narrow(1,1,1).pow(2)).pow(0.5)
            #self.fake_c_magn_A.retain_grad()
            self.VGG_fake_T2=self.vgg(self.fake_T2_cons.transpose(0,1).expand([10,3,192,88]))[0]
        else:
            #self.fake_magn_A = (self.fake_A.narrow(1,0,1).pow(2)+self.fake_A.narrow(1,1,1).pow(2)).pow(0.5)
            #self.fake_magn_A.retain_grad()
            self.VGG_fake_T2=self.vgg(self.fake_T2.transpose(0,1).expand([10,3,192,88]))[0]
        self.VGG_loss_T2=self.criterionL1(self.VGG_fake_T2,self.VGG_real_T2)* self.opt.lambda_vgg
        #Perceptual loss
        #self.real_magn_A =  (self.real_A.narrow(1,0,1).pow(2)+self.real_A.narrow(1,1,1).pow(2)).pow(0.5)
        self.VGG_real_PD=self.vgg(self.real_A.narrow(1,20,10).transpose(0,1).expand([10,3,192,88]))[0]
        if self.opt.use_dt_cns: 
            #self.fake_c_magn_A = (self.fake_A_cons.narrow(1,0,1).pow(2)+self.fake_A_cons.narrow(1,1,1).pow(2)).pow(0.5)
            #self.fake_c_magn_A.retain_grad()
            self.VGG_fake_PD=self.vgg(self.fake_PD_cons.transpose(0,1).expand([10,3,192,88]))[0]
        else:
            #self.fake_magn_A = (self.fake_A.narrow(1,0,1).pow(2)+self.fake_A.narrow(1,1,1).pow(2)).pow(0.5)
            #self.fake_magn_A.retain_grad()
            self.VGG_fake_PD=self.vgg(self.fake_PD.transpose(0,1).expand([10,3,192,88]))[0]
        self.VGG_loss_PD=self.criterionL1(self.VGG_fake_PD,self.VGG_real_PD)* self.opt.lambda_vgg
        
        # Second, G(A) = B
        if self.opt.use_dt_cns:          
            self.loss_G_L1_T1 = self.criterionL1(self.fake_T1_cons, self.real_A.narrow(1,0,10)) * self.opt.lambda_A
        else:
            self.loss_G_L1_T1 = self.criterionL1(self.fake_T1, self.real_A.narrow(1,0,10)) * self.opt.lambda_A
        # Second, G(A) = B
        if self.opt.use_dt_cns:          
            self.loss_G_L1_T2 = self.criterionL1(self.fake_T2_cons, self.real_A.narrow(1,10,10)) * self.opt.lambda_A
        else:
            self.loss_G_L1_T2 = self.criterionL1(self.fake_T2, self.real_A.narrow(1,10,10)) * self.opt.lambda_A
        # Second, G(A) = B
        if self.opt.use_dt_cns:          
            self.loss_G_L1_PD = self.criterionL1(self.fake_PD_cons, self.real_A.narrow(1,20,10)) * self.opt.lambda_A
        else:
            self.loss_G_L1_PD = self.criterionL1(self.fake_PD, self.real_A.narrow(1,20,10)) * self.opt.lambda_A
            
            
        if self.opt.loss_tech == 1:
            self.loss_G = self.loss_G_GAN_T1.pow(2) + self.loss_G_L1_T1.pow(2) + self.VGG_loss_T1.pow(2) + self.loss_G_GAN_T2.pow(2) + self.loss_G_L1_T2.pow(2) + self.VGG_loss_T2.pow(2) + self.loss_G_GAN_PD.pow(2) + self.loss_G_L1_PD.pow(2) + self.VGG_loss_PD.pow(2)
        elif self.opt.loss_tech == 2:
            ganAsoft = self.loss_G_GAN_A.pow(2)/(1+torch.exp(self.loss_G_GAN_B-self.loss_G_GAN_A))
            ganBsoft = self.loss_G_GAN_B.pow(2)/(1+torch.exp(self.loss_G_GAN_A-self.loss_G_GAN_B))
            l1Asoft  = self.loss_G_L1_A.pow(2) /(1+torch.exp(self.loss_G_L1_B-self.loss_G_L1_A))
            l1Bsoft  = self.loss_G_L1_B.pow(2) /(1+torch.exp(self.loss_G_L1_A-self.loss_G_L1_B))
            vggAsoft = self.VGG_loss_A.pow(2) / (1+torch.exp(self.VGG_loss_B-self.VGG_loss_A))
            vggBsoft = self.VGG_loss_B.pow(2) / (1+torch.exp(self.VGG_loss_A-self.VGG_loss_B))
            ganAsoft.retain_grad()
            ganBsoft.retain_grad()
            l1Asoft.retain_grad()
            l1Bsoft.retain_grad()
            vggAsoft.retain_grad()
            vggBsoft.retain_grad()
            self.loss_G = ganAsoft + ganBsoft + l1Asoft + l1Bsoft + vggAsoft + vggBsoft
        elif self.opt.loss_tech == 3:
            self.loss_G = self.loss_G_GAN_T1 + self.loss_G_L1_T1 + self.VGG_loss_T1 + self.loss_G_GAN_T2 + self.loss_G_L1_T2 + self.VGG_loss_T2 + self.loss_G_GAN_PD + self.loss_G_L1_PD + self.VGG_loss_PD
        if not self.opt.use_fixed_acc:
            self.us_loss = self.opt.acc_loss_coe * self.mask_net.find_masks(self.input_A)[1].sum()
            self.loss_G = self.loss_G + self.us_loss
        if self.opt.use_mult_const:
            self.mult_loss = 2*self.opt.mult_loss_coe * (torch.pow(self.mask_net.probmask.mult,2).mean())
            self.loss_G = self.loss_G + self.mult_loss
        else:
            self.mult_loss = torch.Tensor([0])
        self.loss_G.backward()
    def backward_M(self):
        self.loss_M_L1 = self.criterionL1(self.under_B.narrow(1,0,1),self.real_B)
        self.loss_M_L1.backward()
     
    def optimize_parameters(self):        
        self.forward()
        
        self.optimizer_DT1.zero_grad()
        self.backward_DT1()
        self.optimizer_DT1.step()
        
        self.optimizer_DT2.zero_grad()
        self.backward_DT2()
        self.optimizer_DT2.step()
        
        self.optimizer_DPD.zero_grad()
        self.backward_DPD()
        self.optimizer_DPD.step()
        
       # self.optimizer_M.zero_grad()
        self.optimizer_GT1.zero_grad()
        self.optimizer_GT2.zero_grad()
        self.optimizer_GPD.zero_grad()
        self.backward_G()
        self.optimizer_GT1.step()
        self.optimizer_GT2.step()
        self.optimizer_GPD.step()
        #self.optimizer_M.step()

    def get_current_errors(self):
        if self.opt.use_fixed_acc:
            return OrderedDict([('G_GAN_T1', self.loss_G_GAN_T1.item()),
                                ('G_GAN_T2', self.loss_G_GAN_T2.item()),
                                ('G_GAN_PD', self.loss_G_GAN_PD.item()),
                                ('G_L1_T1', self.loss_G_L1_T1.item()),
                                ('G_L1_T2', self.loss_G_L1_T2.item()),
                                ('G_L1_PD', self.loss_G_L1_PD.item()),
                                ('G_VGG_T1', self.VGG_loss_T1.item()),
                                ('G_VGG_T2', self.VGG_loss_T2.item()),
                                ('G_VGG_PD', self.VGG_loss_PD.item()),
                                ('G_SUM', self.loss_G.item()),
                                ('D_real_T1', self.loss_D_real_T1.item()),
                                ('D_fake_T1', self.loss_D_fake_T1.item()),
                                ('D_SUM_T1', self.loss_D_T1.item()),
                                ('D_real_T2', self.loss_D_real_T2.item()),
                                ('D_fake_T2', self.loss_D_fake_T2.item()),
                                ('D_SUM_T2', self.loss_D_T2.item()),
                                ('D_real_PD', self.loss_D_real_PD.item()),
                                ('D_fake_PD', self.loss_D_fake_PD.item()),
                                ('D_SUM_PD', self.loss_D_PD.item()),
                                ('Mult loss', self.mult_loss.item()),
                                ('Mult loss co', self.opt.mult_loss_coe),
                                ('R', 1.0/self.opt.ur),
                                ('prmask mean T1', self.mask_net.pr_mask.narrow(1,0,1).mean().item()),
                                ('rescl mean T1', self.mask_net.pr_mask2.narrow(1,0,1).mean().item()),
                                ('thrs rm mean T1', self.mask_net.lst_pr_mask.narrow(1,0,1).mean().item()),
                                ('prmask mean T2', self.mask_net.pr_mask.narrow(1,1,1).mean().item()),
                                ('rescl mean T2', self.mask_net.pr_mask2.narrow(1,1,1).mean().item()),
                                ('thrs rm mean T2', self.mask_net.lst_pr_mask.narrow(1,1,1).mean().item()),
                                ('prmask mean PD', self.mask_net.pr_mask.narrow(1,2,1).mean().item()),
                                ('rescl mean PD', self.mask_net.pr_mask2.narrow(1,2,1).mean().item()),
                                ('thrs rm mean PD', self.mask_net.lst_pr_mask.narrow(1,2,1).mean().item()),                                
                                ('pr mask slp', self.mask_net.probmask.slope.item()),
                                ('thr mask slp', self.opt.thrding_slope)])
        else:
            return OrderedDict([('G_GAN', self.loss_G_GAN.item()),
                                ('G_L1', self.loss_G_L1.item()),
                                ('G_VGG', self.VGG_loss.item()),
                                ('G_us_loss', self.us_loss.item()),
                                ('G_SUM', self.loss_G.item()),
                                ('D_real', self.loss_D_real.item()),
                                ('D_fake', self.loss_D_fake.item()),
                                ('D_SUM', self.loss_D.item()),
                                ('prmask mean', self.mask_net.pr_mask.mean().item()),
                                ('thrs rm mean', self.mask_net.lst_pr_mask.mean().item()),
                                ('pr slp', self.mask_net.probmask.slope.item()),
                                ('thr slp', self.mask_net.thresholdrandommask.slope.item())])

    def get_current_visuals(self):
        #real_A = util.tensor2im(self.real_magn_A.data)
        #real_B = util.tensor2im(self.real_magn_B.data)
        real_T1 = util.tensor2im(torch.sqrt(torch.pow(self.real_A.narrow(1,0,1).data,2)+torch.pow(self.real_A.narrow(1,1,1).data,2)))
        real_T2 = util.tensor2im(torch.sqrt(torch.pow(self.real_A.narrow(1,10,1).data,2)+torch.pow(self.real_A.narrow(1,11,1).data,2)))
        real_PD = util.tensor2im(torch.sqrt(torch.pow(self.real_A.narrow(1,20,1).data,2)+torch.pow(self.real_A.narrow(1,21,1).data,2)))
        if self.opt.use_dt_cns:
            #fake_A = util.tensor2im(normalize(self.fake_c_magn_A.data))
            #fake_B = util.tensor2im(normalize(self.fake_c_magn_B.data))
            fake_T1 = util.tensor2im(normalize(torch.sqrt(torch.pow(self.fake_T1_cons.narrow(1,0,1).data,2)+torch.pow(self.fake_T1_cons.narrow(1,1,1).data,2))))
            fake_T2 = util.tensor2im(normalize(torch.sqrt(torch.pow(self.fake_T2_cons.narrow(1,0,1).data,2)+torch.pow(self.fake_T2_cons.narrow(1,1,1).data,2))))
            fake_PD = util.tensor2im(normalize(torch.sqrt(torch.pow(self.fake_PD_cons.narrow(1,0,1).data,2)+torch.pow(self.fake_PD_cons.narrow(1,1,1).data,2))))
        else:
            #fake_A = util.tensor2im(self.fake_magn_A.data)
            #fake_B = util.tensor2im(self.fake_magn_B.data)
            fake_T1 = util.tensor2im(normalize(torch.sqrt(torch.pow(self.fake_T1.narrow(1,0,1).data,2)+torch.pow(self.fake_T1.narrow(1,1,1).data,2))))
            fake_T2 = util.tensor2im(normalize(torch.sqrt(torch.pow(self.fake_T2.narrow(1,0,1).data,2)+torch.pow(self.fake_T2.narrow(1,1,1).data,2))))
            fake_PD = util.tensor2im(normalize(torch.sqrt(torch.pow(self.fake_PD.narrow(1,0,1).data,2)+torch.pow(self.fake_PD.narrow(1,1,1).data,2))))
        #re_under_A_im =  util.tensor2im(normalize(self.concAB.narrow(1,0,1).data))
        #re_under_B_im =  util.tensor2im(normalize(self.concAB.narrow(1,2,1).data))
        #abs_fft_real_B_im = find_fft_im(self.real_B.permute(0,2,3,1).data)        
        #abs_fft_real_A_im = find_fft_im(self.real_A.permute(0,2,3,1).data)
        abs_fft_real_T1_im = find_fft_im(torch.stack((self.real_A.narrow(1,0,1).data,self.real_A.narrow(1,1,1).data),-1))        
        abs_fft_real_T2_im = find_fft_im(torch.stack((self.real_A.narrow(1,10,1).data,self.real_A.narrow(1,11,1).data),-1))
        abs_fft_real_PD_im = find_fft_im(torch.stack((self.real_A.narrow(1,20,1).data,self.real_A.narrow(1,21,1).data),-1))
        if self.opt.use_dt_cns:
            #abs_fft_fake_B_im = find_fft_im(self.fake_B_cons.permute(0,2,3,1).data)        
            #abs_fft_fake_A_im = find_fft_im(self.fake_A_cons.permute(0,2,3,1).data)
            abs_fft_fake_T1_im = find_fft_im(torch.stack((self.fake_T1_cons.narrow(1,0,1).data,self.fake_T1_cons.narrow(1,1,1).data),-1))
            abs_fft_fake_T2_im = find_fft_im(torch.stack((self.fake_T2_cons.narrow(1,0,1).data,self.fake_T2_cons.narrow(1,1,1).data),-1))
            abs_fft_fake_PD_im = find_fft_im(torch.stack((self.fake_PD_cons.narrow(1,0,1).data,self.fake_PD_cons.narrow(1,1,1).data),-1))
        else:
            #abs_fft_fake_B_im = find_fft_im(self.fake_B.permute(0,2,3,1).data)        
            #abs_fft_fake_A_im = find_fft_im(self.fake_A.permute(0,2,3,1).data)
            abs_fft_fake_T1_im = find_fft_im(torch.stack((self.fake_T1.narrow(1,0,1).data,self.fake_T1.narrow(1,1,1).data),-1))
            abs_fft_fake_T2_im = find_fft_im(torch.stack((self.fake_T2.narrow(1,0,1).data,self.fake_T2.narrow(1,1,1).data),-1))
            abs_fft_fake_PD_im = find_fft_im(torch.stack((self.fake_PD.narrow(1,0,1).data,self.fake_PD.narrow(1,1,1).data),-1))
#        mask = np.tile(np.fft.fftshift(self.mask_net.find_masks(self.real_B)[0].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1))
#        mask = ((1-(np.transpose(mask, (1, 2, 0)) + 1) / 2.0) * 255.0).astype(np.uint8)       
        if self.opt.use_fixed_acc: 
            pr_mask_AB, res_mask_AB, bin_mask_AB = self.mask_net.find_masks(self.input_A)
            pr_mask_T1 = pr_mask_AB.narrow(1,0,1).squeeze(0)
            pr_mask_T2 = pr_mask_AB.narrow(1,1,1).squeeze(0)
            pr_mask_PD = pr_mask_AB.narrow(1,2,1).squeeze(0)
            res_mask_T1 = res_mask_AB.narrow(1,0,1).squeeze(0)
            res_mask_T2 = res_mask_AB.narrow(1,1,1).squeeze(0)
            res_mask_PD = res_mask_AB.narrow(1,2,1).squeeze(0)
            bin_mask_T1 = bin_mask_AB.narrow(1,0,1).squeeze(0)
            bin_mask_T2 = bin_mask_AB.narrow(1,1,1).squeeze(0)
            bin_mask_PD = bin_mask_AB.narrow(1,2,1).squeeze(0)
        else:
            pr_mask_AB, bin_mask_AB = self.mask_net.find_masks(self.input_A)
            pr_mask_A = pr_mask_AB.narrow(0,0,1).squeeze(0)
            pr_mask_B = pr_mask_AB.narrow(0,1,1).squeeze(0)
            bin_mask_A = bin_mask_AB.narrow(0,0,1).squeeze(0)
            bin_mask_B = bin_mask_AB.narrow(0,1,1).squeeze(0)
        bin_no = 10
        bins = np.linspace(0,1,bin_no+1)
                
        pr_mask_A = np.fft.fftshift(pr_mask_T1.data.squeeze(-1).squeeze(1).cpu().float().numpy())
        freq, bins = np.histogram(pr_mask_A, histedges_equalN(pr_mask_A.flatten(), bin_no))
        fig,ax = plt.subplots(1,1)
        ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge') 
        ax.set_xticks(np.linspace(0,1,11))
        labels = []
        for i in range(bins.size):
            labels.append(('%.1e' % bins[i]))
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both',which='major',labelsize=6)
        ax.set_title('Prob Mask')
        fig.canvas.draw()   
        pr_mask_A_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        pr_mask_A_hist = pr_mask_A_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        if self.opt.use_fixed_acc:
            res_mask_A = np.fft.fftshift(res_mask_T1.data.squeeze(-1).squeeze(1).cpu().float().numpy())
            freq, bins = np.histogram(res_mask_A, histedges_equalN(pr_mask_A.flatten(), bin_no))
            fig,ax = plt.subplots(1,1)
            ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge')
            ax.set_xticks(np.linspace(0,1,11))
            labels = []
            for i in range(bins.size):
                labels.append(('%.1e' % bins[i]))
            ax.set_xticklabels(labels)
            ax.tick_params(axis='both',which='major',labelsize=6)
            ax.set_title('Rscld Mask')
            fig.canvas.draw()   
            res_mask_A_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            res_mask_A_hist = res_mask_A_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        bin_mask_A = np.fft.fftshift(bin_mask_T1.data.squeeze(-1).squeeze(1).cpu().float().numpy())
        freq, bins = np.histogram(bin_mask_A, histedges_equalN(pr_mask_A.flatten(), bin_no))
        fig,ax = plt.subplots(1,1)
        ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge')
        ax.set_xticks(np.linspace(0,1,11))
        labels = []
        for i in range(bins.size):
            labels.append(('%.1e' % bins[i]))
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both',which='major',labelsize=6)
        ax.set_title('Bin Mask')
        fig.canvas.draw()  
        bin_mask_A_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        bin_mask_A_hist = bin_mask_A_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
     
        pr_mask_A = (np.transpose(np.tile(pr_mask_A, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)  
        if self.opt.use_fixed_acc:
            res_mask_A = (np.transpose(np.tile(res_mask_A, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)
        bin_mask_A = (np.transpose(np.tile(bin_mask_A, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)
        plt.close('all')

        pr_mask_B = np.fft.fftshift(pr_mask_T2.data.squeeze(-1).squeeze(1).cpu().float().numpy())
        freq, bins = np.histogram(pr_mask_B, histedges_equalN(pr_mask_B.flatten(), bin_no))
        fig,ax = plt.subplots(1,1)
        ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge') 
        ax.set_xticks(np.linspace(0,1,11))
        labels = []
        for i in range(bins.size):
            labels.append(('%.1e' % bins[i]))
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both',which='major',labelsize=6)
        ax.set_title('Prob Mask')
        fig.canvas.draw()   
        pr_mask_B_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        pr_mask_B_hist = pr_mask_B_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        if self.opt.use_fixed_acc:
            res_mask_B = np.fft.fftshift(res_mask_T2.data.squeeze(-1).squeeze(1).cpu().float().numpy())
            freq, bins = np.histogram(res_mask_B, histedges_equalN(pr_mask_B.flatten(), bin_no))
            fig,ax = plt.subplots(1,1)
            ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge')
            ax.set_xticks(np.linspace(0,1,11))
            labels = []
            for i in range(bins.size):
                labels.append(('%.1e' % bins[i]))
            ax.set_xticklabels(labels)
            ax.tick_params(axis='both',which='major',labelsize=6)
            ax.set_title('Rscld Mask')
            fig.canvas.draw()   
            res_mask_B_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            res_mask_B_hist = res_mask_B_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        bin_mask_B = np.fft.fftshift(bin_mask_T2.data.squeeze(-1).squeeze(1).cpu().float().numpy())
        freq, bins = np.histogram(bin_mask_B, histedges_equalN(pr_mask_B.flatten(), bin_no))
        fig,ax = plt.subplots(1,1)
        ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge')
        ax.set_xticks(np.linspace(0,1,11))
        labels = []
        for i in range(bins.size):
            labels.append(('%.1e' % bins[i]))
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both',which='major',labelsize=6)
        ax.set_title('Bin Mask')
        fig.canvas.draw()  
        bin_mask_B_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        bin_mask_B_hist = bin_mask_B_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
     
        pr_mask_B = (np.transpose(np.tile(pr_mask_B, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)  
        if self.opt.use_fixed_acc:
            res_mask_B = (np.transpose(np.tile(res_mask_B, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)
        bin_mask_B = (np.transpose(np.tile(bin_mask_B, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)
        plt.close('all')        
        
        
        pr_mask_C = np.fft.fftshift(pr_mask_PD.data.squeeze(-1).squeeze(1).cpu().float().numpy())
        freq, bins = np.histogram(pr_mask_C, histedges_equalN(pr_mask_C.flatten(), bin_no))
        fig,ax = plt.subplots(1,1)
        ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge') 
        ax.set_xticks(np.linspace(0,1,11))
        labels = []
        for i in range(bins.size):
            labels.append(('%.1e' % bins[i]))
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both',which='major',labelsize=6)
        ax.set_title('Prob Mask')
        fig.canvas.draw()   
        pr_mask_C_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        pr_mask_C_hist = pr_mask_C_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        if self.opt.use_fixed_acc:
            res_mask_C = np.fft.fftshift(res_mask_PD.data.squeeze(-1).squeeze(1).cpu().float().numpy())
            freq, bins = np.histogram(res_mask_C, histedges_equalN(pr_mask_C.flatten(), bin_no))
            fig,ax = plt.subplots(1,1)
            ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge')
            ax.set_xticks(np.linspace(0,1,11))
            labels = []
            for i in range(bins.size):
                labels.append(('%.1e' % bins[i]))
            ax.set_xticklabels(labels)
            ax.tick_params(axis='both',which='major',labelsize=6)
            ax.set_title('Rscld Mask')
            fig.canvas.draw()   
            res_mask_C_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            res_mask_C_hist = res_mask_C_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        bin_mask_C = np.fft.fftshift(bin_mask_PD.data.squeeze(-1).squeeze(1).cpu().float().numpy())
        freq, bins = np.histogram(bin_mask_C, histedges_equalN(pr_mask_C.flatten(), bin_no))
        fig,ax = plt.subplots(1,1)
        ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge')
        ax.set_xticks(np.linspace(0,1,11))
        labels = []
        for i in range(bins.size):
            labels.append(('%.1e' % bins[i]))
        ax.set_xticklabels(labels)
        ax.tick_params(axis='both',which='major',labelsize=6)
        ax.set_title('Bin Mask')
        fig.canvas.draw()  
        bin_mask_C_hist = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        bin_mask_C_hist = bin_mask_C_hist.reshape(fig.canvas.get_width_height()[::-1] + (3,))        
     
        pr_mask_C = (np.transpose(np.tile(pr_mask_C, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)  
        if self.opt.use_fixed_acc:
            res_mask_C = (np.transpose(np.tile(res_mask_C, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)
        bin_mask_C = (np.transpose(np.tile(bin_mask_C, (3, 1, 1)), (1, 2, 0)) * 255.0).astype(np.uint8)
        plt.close('all')   
        
        if self.opt.use_fixed_acc:
            return OrderedDict([('real_T1', real_T1), ('real_T2', real_T2), ('real_PD', real_PD), ('fake_T1', fake_T1), ('fake_T2', fake_T2), ('fake_PD', fake_PD), ('abs_k_real_T1',abs_fft_real_T1_im), ('abs_k_real_T2',abs_fft_real_T2_im), ('abs_k_real_PD',abs_fft_real_PD_im), ('abs_k_fake_T1',abs_fft_fake_T1_im), ('abs_k_fake_T2',abs_fft_fake_T2_im), ('abs_k_fake_PD',abs_fft_fake_PD_im), ('pr_mask_T1', pr_mask_A), ('res_mask_T1', res_mask_A), ('bin_mask_T1', bin_mask_A),('pr_mask_T1_hist', pr_mask_A_hist),('res_mask_T1_hist', res_mask_A_hist),('bin_mask_T1_hist', bin_mask_A_hist), ('pr_mask_T2', pr_mask_B), ('res_mask_T2', res_mask_B), ('bin_mask_T2', bin_mask_B),('pr_mask_T2_hist', pr_mask_B_hist),('res_mask_T2_hist', res_mask_B_hist),('bin_mask_T2_hist', bin_mask_B_hist),('pr_mask_PD', pr_mask_C), ('res_mask_PD', res_mask_C), ('bin_mask_PD', bin_mask_C),('pr_mask_PD_hist', pr_mask_C_hist),('res_mask_PD_hist', res_mask_C_hist),('bin_mask_PD_hist', bin_mask_C_hist)])
        else:
            return OrderedDict([('real_A', real_A), ('real_A', real_B), ('Re[under_A]', re_under_B_im), ('fake_A', fake_B), ('abs_k_real_A',abs_fft_real_B_im), ('abs_k_fake_A',abs_fft_fake_B_im), ('pr_mask_A', pr_mask_A), ('bin_mask_A', bin_mask_A),('pr_mask_A_hist', pr_mask_A_hist),('bin_mask_A_hist', bin_mask_A_hist)])

       # return OrderedDict([('real_A', real_A), ('fake_A', fake_B), ('mask', mask)])
    def save(self, label):
        self.save_network(self.mask_net,'M',label, self.opt.gpu_id)
        self.save_network(self.netGT1, 'GT1', label, self.opt.gpu_id)
        self.save_network(self.netGT2, 'GT2', label, self.opt.gpu_id)
        self.save_network(self.netGPD, 'GPD', label, self.opt.gpu_id)
        self.save_network(self.netDT1, 'DT1', label, self.opt.gpu_id)
        self.save_network(self.netDT2, 'DT2', label, self.opt.gpu_id)
        self.save_network(self.netDPD, 'DPD', label, self.opt.gpu_id)

    def find_errors_in_testing(self):
        # Second, G(A) = B
        self.loss_G_L1_A_cns = self.criterionL1(self.fake_A_cons, self.input_A) * self.opt.lambda_A
        self.loss_G_L1_A = self.criterionL1(self.fake_A, self.input_A) * self.opt.lambda_A
        #Perceptual loss
        self.VGG_real_A=self.vgg(self.input_A.expand([int(self.input_A.size()[0]),3,int(self.input_A.size()[2]),int(self.input_A.size()[3])]))[0]
        self.VGG_fake_A_cns=self.vgg(self.fake_A_cons.expand([int(self.input_Aol.size()[0]),3,int(self.input_A.size()[2]),int(self.input_A.size()[3])]))[0]
        self.VGG_fake_A=self.vgg(self.fake_A.expand([int(self.input_A.size()[0]),3,int(self.input_A.size()[2]),int(self.input_A.size()[3])]))[0]
        self.VGG_loss_A_cns=self.criterionL1(self.VGG_fake_A_cns,self.VGG_real_A)* self.opt.lambda_vgg
        self.VGG_loss_A=self.criterionL1(self.VGG_fake_A,self.VGG_real_A)* self.opt.lambda_vgg
        self.loss_G_L1_B_cns = self.criterionL1(self.fake_B_cons, self.real_B) * self.opt.lambda_A  
        self.loss_G_L1_B = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_A  
        self.VGG_real_B=self.vgg(self.real_B.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG_fake_B_cns=self.vgg(self.fake_B_cons.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG_fake_B=self.vgg(self.fake_B.expand([int(self.real_B.size()[0]),3,int(self.real_B.size()[2]),int(self.real_B.size()[3])]))[0]
        self.VGG_loss_B_cns=self.criterionL1(self.VGG_fake_B_cns,self.VGG_real_B)* self.opt.lambda_vgg
        self.VGG_loss_B=self.criterionL1(self.VGG_fake_B,self.VGG_real_B)* self.opt.lambda_vgg
        if self.opt.use_fixed_acc:
            return (self.loss_G_L1_A_cns.item(), self.loss_G_L1_A.item(), self.VGG_loss_A_cns.item(), self.VGG_loss_A.item()),(self.loss_G_L1_B_cns.item(), self.loss_G_L1_B.item(), self.VGG_loss_B_cns.item(),self.VGG_loss_B.item())
        else:
            self.us_loss = self.opt.acc_loss_coe * self.mask_net.find_masks(self.input_A)[1].sum()
            return self.loss_G_L1.item(), self.VGG_loss.item(), self.us_loss.item()

def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))   
def normalize(inp):
    max_inp = inp.max()
    min_inp = inp.min()
    return (2*inp-min_inp-max_inp)/(max_inp-min_inp)
def find_fft_im(tns):
    #fft_tns = torch.fft(torch.stack((tns,tns*0),-1),2,normalized=True)
    fft_tns = torch.fft.fft2(tns[...,0]+1j*tns[...,1], norm="ortho")
    
    abs_fft_tns = fft_tns.abs().cpu().detach().numpy() #torch.sqrt(fft_tns[...,0]**2+fft_tns[...,1]**2)
    image_numpy = np.fft.fftshift((np.log(abs_fft_tns/abs_fft_tns.min())/np.log(abs_fft_tns/abs_fft_tns.min()).max())[0])
    image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    return image_numpy.astype(np.uint8)   
#Extracting VGG feature maps before the 2nd maxpooling layer  
class VGG16(torch.nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        for x in range(4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for param in self.parameters():
            param.requires_grad = False
    def forward(self, X):
        h_relu1 = self.stage1(X)
        h_relu2 = self.stage2(h_relu1)       
        return h_relu2
