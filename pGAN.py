import math
import util.util as util
from matplotlib.transforms import blended_transform_factory
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from skimage.measure import compare_ssim as ssim
import matplotlib.lines as lines
import numpy as np
import torch
def tensor2im_pgan(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor#[0]#.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) - image_numpy.min()) / (image_numpy.max() - image_numpy.min()) * 255.0
    return image_numpy.astype(imtype)
def find_ur_for_comp(desired_ur, thrd_slp):
    return (math.log((1+math.exp(-thrd_slp))/2)+desired_ur*thrd_slp)/(thrd_slp+2*math.log((1+math.exp(-thrd_slp))/2))
def init_perf_arr(opt, arr_name, load=True):
    perf_arr = np.zeros(opt.niter+opt.niter_decay)
    if opt.continue_train and load:
        perf_arr[:opt.which_epoch]=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, arr_name + '.npy'))[:opt.which_epoch]
    return perf_arr
def normalize(inp):
    max_inp = inp.max()
    min_inp = inp.min()
    return (inp-min_inp)/(max_inp-min_inp)  
def train(opt):
    #Firstly load val images to report psnr on val
    opt.phase = 'val'
    data_loader = CreateDataLoader(opt)
    val_dataset = data_loader.load_data()
    val_dataset_size = len(data_loader)
    print('Validation images = %d' % val_dataset_size)   
    opt.phase = 'train'
    model = create_model(opt)
    #Loading data
    data_loader = CreateDataLoader(opt)
    tr_dataset = data_loader.load_data()
    tr_dataset_size = len(data_loader)
    print('Training images = %d' % tr_dataset_size)    
    visualizer = Visualizer(opt)
    total_steps = 0
    val_psnr_vs_ep1_T1_c = init_perf_arr(opt, 'val_psnr_vs_ep1_T1_c')
    val_psnr_vs_ep1_T1 = init_perf_arr(opt, 'val_psnr_vs_ep1_T1')
    val_psnr_vs_ep1_T2_c = init_perf_arr(opt, 'val_psnr_vs_ep1_T2_c')
    val_psnr_vs_ep1_T2 = init_perf_arr(opt, 'val_psnr_vs_ep1_T2')
    val_psnr_vs_ep1_PD_c = init_perf_arr(opt, 'val_psnr_vs_ep1_PD_c')
    val_psnr_vs_ep1_PD = init_perf_arr(opt, 'val_psnr_vs_ep1_PD')
#    (tr_psnr_vs_ep1_A_c,tr_psnr_vs_ep1_A, tr_ssim_vs_ep_A_c, tr_ssim_vs_ep_A,tr_l1_vs_ep_A_c,tr_l1_vs_ep_A,tr_vgg_vs_ep_A_c,tr_vgg_vs_ep_A,tr_mmeans_vs_ep_A) = (init_perf_arr(opt, 'tr_psnr_vs_ep1_A_c'), init_perf_arr(opt, 'tr_psnr_vs_ep1_A'),\
#    init_perf_arr(opt, 'tr_ssim_vs_ep_A_c'),init_perf_arr(opt, 'tr_ssim_vs_ep_A'),init_perf_arr(opt, 'tr_l1_vs_ep_A_c'),init_perf_arr(opt, 'tr_l1_vs_ep_A'),init_perf_arr(opt, 'tr_vgg_vs_ep_A_c'),init_perf_arr(opt, 'tr_vgg_vs_ep_A'),init_perf_arr(opt, 'tr_mmeans_vs_ep_A'))
#    (tr_psnr_vs_ep1_B_c,tr_psnr_vs_ep1_B,tr_ssim_vs_ep_B_c, tr_ssim_vs_ep_B,tr_l1_vs_ep_B_c,tr_l1_vs_ep_B,tr_vgg_vs_ep_B_c,tr_vgg_vs_ep_B,tr_mmeans_vs_ep_B) = (init_perf_arr(opt, 'tr_psnr_vs_ep1_B_c'),init_perf_arr(opt, 'tr_psnr_vs_ep1_B'),\
#    init_perf_arr(opt, 'tr_ssim_vs_ep_B_c'),init_perf_arr(opt, 'tr_ssim_vs_ep_B'),init_perf_arr(opt, 'tr_l1_vs_ep_B_c'),init_perf_arr(opt, 'tr_l1_vs_ep_B'),init_perf_arr(opt, 'tr_vgg_vs_ep_B_c'),init_perf_arr(opt, 'tr_vgg_vs_ep_B'),init_perf_arr(opt, 'tr_mmeans_vs_ep_B'))
#    (val_psnr_vs_ep1_A_c,val_psnr_vs_ep1_A,val_ssim_vs_ep_A_c, val_ssim_vs_ep_A,val_l1_vs_ep_A_c,val_l1_vs_ep_A,val_vgg_vs_ep_A_c,val_vgg_vs_ep_A,val_mmeans_vs_ep_A) = (init_perf_arr(opt, 'val_psnr_vs_ep1_A_c'),init_perf_arr(opt, 'val_psnr_vs_ep1_A'),\
#    init_perf_arr(opt, 'val_ssim_vs_ep_A_c'),init_perf_arr(opt, 'val_ssim_vs_ep_A'), init_perf_arr(opt, 'val_l1_vs_ep_A_c'),init_perf_arr(opt, 'val_l1_vs_ep_A'),init_perf_arr(opt, 'val_vgg_vs_ep_A_c'),init_perf_arr(opt, 'val_vgg_vs_ep_A'),init_perf_arr(opt, 'val_mmeans_vs_ep_A'))
#    (val_psnr_vs_ep1_B_c,val_psnr_vs_ep1_B,val_ssim_vs_ep_B_c, val_ssim_vs_ep_B,val_l1_vs_ep_B_c,val_l1_vs_ep_B,val_vgg_vs_ep_B_c,val_vgg_vs_ep_B,val_mmeans_vs_ep_B) = (init_perf_arr(opt, 'val_psnr_vs_ep1_B_c'),init_perf_arr(opt, 'val_psnr_vs_ep1_B'), \
#    init_perf_arr(opt, 'val_ssim_vs_ep_B_c'),init_perf_arr(opt, 'val_ssim_vs_ep_B'), init_perf_arr(opt, 'val_l1_vs_ep_B_c'),init_perf_arr(opt, 'val_l1_vs_ep_B'),init_perf_arr(opt, 'val_vgg_vs_ep_B_c'),init_perf_arr(opt, 'val_vgg_vs_ep_B'),init_perf_arr(opt, 'val_mmeans_vs_ep_B'))
#    if not opt.use_fixed_acc:
#        tr_us_vs_ep = init_perf_arr(opt, 'tr_us_vs_ep')
#        val_us_vs_ep = init_perf_arr(opt, 'val_us_vs_ep')
    psnr_log_name = os.path.join(opt.checkpoints_dir, opt.run_name, 'psnr_log.txt')
    with open(psnr_log_name, 'a+') as psnr_log_file:
        psnr_log_file.write('Network: %s\nConfiguration: %s\n================ Performance on Train and Val Sets ================\n' % (opt.net_name, opt.run_name))
   #Starts training
    max_val_psnr1_sum = 0
    max_val_psnr1_sum_c = 0
    first_mult_coe = opt.mult_loss_coe
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):       
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0
        for i, data in enumerate(tr_dataset):
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()
            #Save current images (real_A, real_B, fake_B)
            if  epoch_iter % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch,epoch_iter, save_result)
            #Save current errors   
            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / tr_dataset_size, opt, errors)
            #Save model based on the number of iterations
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
    
            iter_data_time = time.time()
        #Save model based on the number of epochs
        print(opt.dataset_mode)
        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)
        
        if epoch % 1 == 0:
            # Performance on the train set
#            if opt.use_fixed_acc:
#                ([tr_psnr_vs_ep1_A_c[epoch-1],tr_psnr_vs_ep1_A[epoch-1], tr_ssim_vs_ep_A_c[epoch-1], tr_ssim_vs_ep_A[epoch-1],tr_l1_vs_ep_A_c[epoch-1], tr_l1_vs_ep_A[epoch-1],tr_vgg_vs_ep_A_c[epoch-1],tr_vgg_vs_ep_A[epoch-1],tr_mmeans_vs_ep_A[epoch-1]], \
#                [tr_psnr_vs_ep1_B_c[epoch-1],tr_psnr_vs_ep1_B[epoch-1],tr_ssim_vs_ep_B_c[epoch-1], tr_ssim_vs_ep_B[epoch-1], tr_l1_vs_ep_B_c[epoch-1],tr_l1_vs_ep_B[epoch-1],tr_vgg_vs_ep_B_c[epoch-1],tr_vgg_vs_ep_B[epoch-1],tr_mmeans_vs_ep_B[epoch-1]]) = aver_psnr_l1_vgg(model, opt, tr_dataset, tr_dataset_size) 
#                message = 'epoch: %d, time: %.3f, train_psnr_1_c(av): %.3f, train_psnr_1(av): %.3f, train_ssim_c(av): %.3f, train_ssim(av): %.3f, tr_l1_loss_c(av): %.3f, tr_l1_loss(av): %.3f, tr_vgg_loss_c(av) %.3f, tr_vgg_loss(av) %.3f, '\
#                % (epoch, time.time()-epoch_start_time,(tr_psnr_vs_ep1_A_c[epoch-1]+tr_psnr_vs_ep1_B_c[epoch-1])/2,(tr_psnr_vs_ep1_A[epoch-1]+tr_psnr_vs_ep1_B[epoch-1])/2,(tr_ssim_vs_ep_A_c[epoch-1]+tr_ssim_vs_ep_B_c[epoch-1])/2, (tr_ssim_vs_ep_A[epoch-1]+tr_ssim_vs_ep_B[epoch-1])/2, \
#                (tr_l1_vs_ep_A_c[epoch-1]+tr_l1_vs_ep_B_c[epoch-1])/2, (tr_l1_vs_ep_A[epoch-1]+tr_l1_vs_ep_B[epoch-1])/2, (tr_vgg_vs_ep_A_c[epoch-1]+tr_vgg_vs_ep_B_c[epoch-1])/2,(tr_vgg_vs_ep_A[epoch-1]+tr_vgg_vs_ep_B[epoch-1])/2)
#            else:
#                tr_psnr_vs_ep1[epoch-1], tr_psnr_vs_ep2[epoch-1], tr_l1_vs_ep[epoch-1], \
#                tr_vgg_vs_ep[epoch-1],tr_us_vs_ep[epoch-1] = aver_psnr_l1_vgg(model,opt,tr_dataset,tr_dataset_size)
#                message = '(epoch: %d, time: %.3f, train_psnr_1: %.3f, train_psnr_2: %.3f, tr_l1_loss: %.3f, tr_vgg_loss %.3f, tr_us_loss: %.3f, '\
#                % (epoch, time.time()-epoch_start_time,tr_psnr_vs_ep1[epoch-1], \
#                tr_psnr_vs_ep2[epoch-1], tr_l1_vs_ep[epoch-1],tr_vgg_vs_ep[epoch-1],tr_us_vs_ep[epoch-1])
#                np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_us_vs_ep'),tr_us_vs_ep)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep1_A'),tr_psnr_vs_ep1_A)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep1_A_c'),tr_psnr_vs_ep1_A_c)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep1_B'),tr_psnr_vs_ep1_B)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep1_B_c'),tr_psnr_vs_ep1_B_c)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_ssim_vs_ep_A_c'),tr_ssim_vs_ep_A_c)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_ssim_vs_ep_A'),tr_ssim_vs_ep_A)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_ssim_vs_ep_B_c'),tr_ssim_vs_ep_B_c)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_ssim_vs_ep_B'),tr_ssim_vs_ep_B)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_l1_vs_ep_A'),tr_l1_vs_ep_A)       
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_l1_vs_ep_B'),tr_l1_vs_ep_B) 
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_vgg_vs_ep_A'),tr_vgg_vs_ep_A)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_vgg_vs_ep_B'),tr_vgg_vs_ep_B)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_l1_vs_ep_A_c'),tr_l1_vs_ep_A_c)       
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_l1_vs_ep_B_c'),tr_l1_vs_ep_B_c) 
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_vgg_vs_ep_A_c'),tr_vgg_vs_ep_A_c)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_vgg_vs_ep_B_c'),tr_vgg_vs_ep_B_c)            
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_mmeans_vs_ep_A'),tr_mmeans_vs_ep_A)
#            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_mmeans_vs_ep_B'),tr_mmeans_vs_ep_B)           
            # Performance on the validation set
            if opt.use_fixed_acc:
                #([val_psnr_vs_ep1_A_c[epoch-1],val_psnr_vs_ep1_A[epoch-1],val_ssim_vs_ep_A_c[epoch-1], val_ssim_vs_ep_A[epoch-1], val_l1_vs_ep_A_c[epoch-1],val_l1_vs_ep_A[epoch-1],val_vgg_vs_ep_A_c[epoch-1],val_vgg_vs_ep_A[epoch-1],val_mmeans_vs_ep_A[epoch-1]], \
                #[val_psnr_vs_ep1_B_c[epoch-1],val_psnr_vs_ep1_B[epoch-1],val_ssim_vs_ep_B_c[epoch-1], val_ssim_vs_ep_B[epoch-1], val_l1_vs_ep_B_c[epoch-1], val_l1_vs_ep_B[epoch-1],val_vgg_vs_ep_B_c[epoch-1],val_vgg_vs_ep_B[epoch-1],val_mmeans_vs_ep_B[epoch-1]])= aver_psnr_l1_vgg(model, opt, val_dataset, val_dataset_size) 
                ([val_psnr_vs_ep1_T1_c[epoch-1],val_psnr_vs_ep1_T1[epoch-1]],[val_psnr_vs_ep1_T2_c[epoch-1],val_psnr_vs_ep1_T2[epoch-1]],[val_psnr_vs_ep1_PD_c[epoch-1],val_psnr_vs_ep1_PD[epoch-1]])= aver_psnr_l1_vgg_train(model, opt, val_dataset, val_dataset_size)
#                message = 'epoch: %d, time: %.3f, val_psnr_1_c(av): %.3f, val_psnr_1(av): %.3f, val_ssim_c(av): %.3f, val_ssim(av): %.3f, val_l1_loss_c(av): %.3f, val_l1_loss(av): %.3f, val_vgg_loss_c(av) %.3f, val_vgg_loss(av) %.3f, '\
#                % (epoch, time.time()-epoch_start_time,(val_psnr_vs_ep1_A_c[epoch-1]+val_psnr_vs_ep1_B_c[epoch-1])/2,(val_psnr_vs_ep1_A[epoch-1]+val_psnr_vs_ep1_B[epoch-1])/2,(val_ssim_vs_ep_A_c[epoch-1]+val_ssim_vs_ep_B_c[epoch-1])/2, (val_ssim_vs_ep_A[epoch-1]+val_ssim_vs_ep_B[epoch-1])/2, \
#                (val_l1_vs_ep_A_c[epoch-1]+val_l1_vs_ep_B_c[epoch-1])/2, (val_l1_vs_ep_A[epoch-1]+val_l1_vs_ep_B[epoch-1])/2, (val_vgg_vs_ep_A_c[epoch-1]+val_vgg_vs_ep_B_c[epoch-1])/2,(val_vgg_vs_ep_A[epoch-1]+val_vgg_vs_ep_B[epoch-1])/2)
                message = 'val_psnr_av (t1: %.3f,t2: %.3f,pd: %.3f): %.3f, val_psnr_c_av: (t1: %.3f,t2: %.3f,pd: %.3f) %.3f' % (val_psnr_vs_ep1_T1[epoch-1],val_psnr_vs_ep1_T2[epoch-1],val_psnr_vs_ep1_PD[epoch-1],(val_psnr_vs_ep1_T1[epoch-1]+val_psnr_vs_ep1_T2[epoch-1]+val_psnr_vs_ep1_PD[epoch-1])/3.0, val_psnr_vs_ep1_T1_c[epoch-1],val_psnr_vs_ep1_T2_c[epoch-1],val_psnr_vs_ep1_PD_c[epoch-1],(val_psnr_vs_ep1_T1_c[epoch-1]+val_psnr_vs_ep1_T2_c[epoch-1]+val_psnr_vs_ep1_PD_c[epoch-1])/3.0)
            else:
                val_psnr_vs_ep1[epoch-1], val_psnr_vs_ep2[epoch-1], val_l1_vs_ep[epoch-1], \
                val_vgg_vs_ep[epoch-1],val_us_vs_ep[epoch-1] = aver_psnr_l1_vgg(model,opt,val_dataset,val_dataset_size)
                message += 'val_psnr_1: %.3f, val_psnr_2: %.3f, val_l1_loss: %.3f, val_vgg_loss %.3f, val_us_loss: %.3f'\
                % (val_psnr_vs_ep1[epoch-1], val_psnr_vs_ep2[epoch-1], \
                val_l1_vs_ep[epoch-1],val_vgg_vs_ep[epoch-1],val_us_vs_ep[epoch-1])
                np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_us_vs_ep'),val_us_vs_ep)
            message += ', pr slp: %.3f, thr slp: %.3f, ur: %.2f)' % (model.mask_net.probmask.slope.item(), model.mask_net.thresholdrandommask.slope.item(),opt.ur)
            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep1_T1'),val_psnr_vs_ep1_T1)    
            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep1_T1_c'),val_psnr_vs_ep1_T1_c)
            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep1_T2'),val_psnr_vs_ep1_T2)
            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep1_T2_c'),val_psnr_vs_ep1_T2_c)
            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep1_PD'),val_psnr_vs_ep1_PD)
            np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep1_PD_c'),val_psnr_vs_ep1_PD_c)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_ssim_vs_ep_A_c'),val_ssim_vs_ep_A_c)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_ssim_vs_ep_A'),val_ssim_vs_ep_A)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_ssim_vs_ep_B_c'),val_ssim_vs_ep_B_c)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_ssim_vs_ep_B'),val_ssim_vs_ep_B)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_l1_vs_ep_A'),val_l1_vs_ep_A)    
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_l1_vs_ep_A_c'),val_l1_vs_ep_A_c)    
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_l1_vs_ep_B'),val_l1_vs_ep_B) 
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_l1_vs_ep_B_c'),val_l1_vs_ep_B_c) 
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_vgg_vs_ep_A'),val_vgg_vs_ep_A)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_vgg_vs_ep_B'),val_vgg_vs_ep_B)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_vgg_vs_ep_A_c'),val_vgg_vs_ep_A_c)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_vgg_vs_ep_B_c'),val_vgg_vs_ep_B_c) 
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_mmeans_vs_ep_A'),val_mmeans_vs_ep_A)
           # np.save(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_mmeans_vs_ep_B'),val_mmeans_vs_ep_B) 
            with open(psnr_log_name, "a") as psnr_log_file:
                psnr_log_file.write('%s\n' % message) 
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
        if (not opt.use_dt_cns) and (opt.use_mult_const) and epoch <= 49:
            opt.ur = math.pow(1.0/float(opt.r),epoch/49.0)
            opt.mult_loss_coe = first_mult_coe - first_mult_coe*epoch/49.0
            model.opt = opt
        else:
            new_max_val_psnr1_sum_c = max(max_val_psnr1_sum_c, val_psnr_vs_ep1_T1_c[epoch-1]+val_psnr_vs_ep1_T2_c[epoch-1]+val_psnr_vs_ep1_PD_c[epoch-1])
            new_max_val_psnr1_sum = max(max_val_psnr1_sum, val_psnr_vs_ep1_T1[epoch-1]+val_psnr_vs_ep1_T2[epoch-1]+val_psnr_vs_ep1_PD[epoch-1])
            if (new_max_val_psnr1_sum_c != max_val_psnr1_sum_c) or (new_max_val_psnr1_sum != max_val_psnr1_sum): # or new_max_val_psnr1_sum_c != max_val_psnr1_sum_c) or (epoch % 5 == 0):
                model.save(epoch)
                print('Model saved.')
            max_val_psnr1_sum= new_max_val_psnr1_sum
            max_val_psnr1_sum_c= new_max_val_psnr1_sum_c

def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise  
def test(opt):
    plt.close('all')    
    psnr_meth = 1
    fig,axes = plt.subplots(1,4,figsize=(15,4.3))
    #tr_psnr_vs_ep_A=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep'+str(psnr_meth)+'_A.npy'))
    val_psnr_vs_ep_T1=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'_T1.npy'))
    #tr_psnr_vs_ep_B=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep'+str(psnr_meth)+'_B.npy'))
    val_psnr_vs_ep_T2=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'_T2.npy'))
    val_psnr_vs_ep_PD=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'_PD.npy'))
    #axes[0].plot(tr_psnr_vs_ep_B, label="Aver. on Train Set, T1")
    #axes[0].plot(val_psnr_vs_ep_B, label="Aver. on Val Set, T1")
    #axes[0].plot(tr_psnr_vs_ep_A, label="Aver. on Train Set, T2")
    #axes[0].plot(val_psnr_vs_ep_A, label="Aver. on Val Set, T2")
    axes[0].set_title('PSNR vs Epoch')
    axes[0].legend(loc="best")
    axes[0].set_ylabel('PSNR')
    axes[0].set_xlabel('Epoch No')
    
    #tr_l1_vs_ep_A=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_l1_vs_ep_A.npy'))
#    val_l1_vs_ep_A=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_l1_vs_ep_A.npy'))
    #tr_l1_vs_ep_B=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_l1_vs_ep_B.npy'))
#    val_l1_vs_ep_B=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_l1_vs_ep_B.npy'))
    #axes[1].plot(tr_l1_vs_ep_B, label="Aver on Train Set, T1")
#    axes[1].plot(val_l1_vs_ep_B, label="Aver on Val Set, T1")
    #axes[1].plot(tr_l1_vs_ep_A, label="Aver on Train Set, T2")
#    axes[1].plot(val_l1_vs_ep_A, label="Aver on Val Set, T2")
    axes[1].set_title('L1 Loss vs Epoch')
    axes[1].legend(loc="best")
    axes[1].set_ylabel('L1 Loss')
    axes[1].set_xlabel('Epoch No')
   
    #tr_vgg_vs_ep_A=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_vgg_vs_ep_A.npy'))
#    val_vgg_vs_ep_A=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_vgg_vs_ep_A.npy'))
    #tr_vgg_vs_ep_B=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_vgg_vs_ep_B.npy'))
 #   val_vgg_vs_ep_B=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_vgg_vs_ep_B.npy'))
    #axes[2].plot(tr_vgg_vs_ep_B, label="Aver on Train Set, T1")
  #  axes[2].plot(val_vgg_vs_ep_B, label="Aver on Val Set, T1")
    #axes[2].plot(tr_vgg_vs_ep_A, label="Aver on Train Set, T2")
   # axes[2].plot(val_vgg_vs_ep_A, label="Aver on Val Set, T2")
    axes[2].set_title('VGG Loss vs Epoch')
    axes[2].legend(loc="best")
    axes[2].set_ylabel('VGG Loss')
    axes[2].set_xlabel('Epoch No')
    
    #tr_psnr_vs_ep_A_c=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep'+str(psnr_meth)+'_A_c.npy'))
    val_psnr_vs_ep_T1_c=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'_T1_c.npy'))
    #tr_psnr_vs_ep_B_c=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep'+str(psnr_meth)+'_B_c.npy'))
    val_psnr_vs_ep_T2_c=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'_T2_c.npy'))
    val_psnr_vs_ep_PD_c=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'_PD_c.npy'))
    #axes[3].plot(tr_psnr_vs_ep_B_c, label="Aver on Train Set, T1")
    axes[3].plot(val_psnr_vs_ep_T1_c, label="Aver on Val Set, T1")
    #axes[3].plot(tr_psnr_vs_ep_A_c, label="Aver on Train Set, T2")
    axes[3].plot(val_psnr_vs_ep_T2_c, label="Aver on Val Set, T2")
    axes[3].plot(val_psnr_vs_ep_PD_c, label="Aver on Val Set, PD")
    axes[3].set_title('PSNR vs Epoch (with Data Consistency)')
    axes[3].legend(loc="best")
    axes[3].set_ylabel('PSNR')
    axes[3].set_xlabel('Epoch No')
    plt.suptitle('PSNR vs Epoch (JRGAN with Mask Optimization with R = %s)' % str(opt.r) + opt.use_mult_const * ' with additional loss')
    output_dir = '/auto/k2/ridvan/Last_Summary/' + str(opt.net_name) + '/' + opt.run_name
    mkdir_p(output_dir)
    plt.savefig(output_dir + '/PSNRS.png', bbox_inches='tight')
    print('PSNR vs Epoch figure saved.')
    plt.show()
    #print(val_psnr_vs_ep_A_c[:50]+val_psnr_vs_ep_B_c[:50])
    if opt.use_mult_const and (not opt.use_dt_cns):
        opt.which_epoch = np.argmax(val_psnr_vs_ep_T1_c[49:]+val_psnr_vs_ep_T2_c[49:]+val_psnr_vs_ep_PD_c[49:]) + 49 + 1
        #opt.which_epoch = 63
    else:
        opt.which_epoch = np.argmax(val_psnr_vs_ep_T1_c+val_psnr_vs_ep_T2_c+val_psnr_vs_ep_PD_c) + 1
    #opt.which_epoch = 74
    # np.max(val_psnr_vs_ep_A_c[49:]+val_psnr_vs_ep_B_c[49:])
#    print('Optimal epoch is ' + str(opt.which_epoch) + ' which gives a val av psnr of {:.3f} (av), {:.3f} (T1), {:.3f} (T2) on dtcns; and {:.3f} (av), {:.3f} (T1), {:.3f} (T2).'.format((val_psnr_vs_ep_B_c[opt.which_epoch-1]+val_psnr_vs_ep_A_c[opt.which_epoch-1])/2,val_psnr_vs_ep_B_c[opt.which_epoch-1],val_psnr_vs_ep_A_c[opt.which_epoch-1],(val_psnr_vs_ep_B[opt.which_epoch-1]+val_psnr_vs_ep_A[opt.which_epoch-1])/2,val_psnr_vs_ep_B[opt.which_epoch-1],val_psnr_vs_ep_A[opt.which_epoch-1]))
    print('Optimal epoch is ' + str(opt.which_epoch) + ' which gives a val av psnr of {:.3f} (av), {:.3f} (T1), {:.3f} (T2), {:.3f} (PD) on dtcns.'.format((val_psnr_vs_ep_T1_c[opt.which_epoch-1]+val_psnr_vs_ep_T2_c[opt.which_epoch-1]+val_psnr_vs_ep_PD_c[opt.which_epoch-1])/3,val_psnr_vs_ep_T1_c[opt.which_epoch-1],val_psnr_vs_ep_T2_c[opt.which_epoch-1],val_psnr_vs_ep_PD_c[opt.which_epoch-1]))
    print('No of eps: ', np.argmax((val_psnr_vs_ep_T1_c==0)))    
    #if opt.use_mult_const:
    	#noncnsopt = np.argmax(val_psnr_vs_ep_A[49:]+val_psnr_vs_ep_B[49:]) + 1 + 49
    #else:
	#noncnsopt = np.argmax(val_psnr_vs_ep_A+val_psnr_vs_ep_B) + 1
    #print('For nondatacons, opt ep is ' + str(noncnsopt) + ' which gives a val av psnr of {:.3f} (av), {:.3f} (T1), {:.3f} (T2) on dtcns; and {:.3f} (av), {:.3f} (T1), {:.3f} (T2).'.format((val_psnr_vs_ep_B_c[noncnsopt-1]+val_psnr_vs_ep_A_c[noncnsopt-1])/2,val_psnr_vs_ep_B_c[noncnsopt-1],val_psnr_vs_ep_A_c[noncnsopt-1],(val_psnr_vs_ep_B[noncnsopt-1]+val_psnr_vs_ep_A[noncnsopt-1])/2,val_psnr_vs_ep_B[noncnsopt-1],val_psnr_vs_ep_A[noncnsopt-1]))
    #opt.which_epoch = int(math.ceil(opt.which_epoch/25.0)) * 25
    #print('Epoch ' + str(opt.which_epoch) + ' will be used.')
    opt.epoch_count=opt.which_epoch + 1 
    opt.use_hrd_thr = True#(input('Do you want to use hard thresholding (1) or sigmoid (2)? ') == 1)
    #binarize_mask = (input('Do you want to binarize mask (1) or not (0)?') == 1) 
    #opt.use_hrd_thr = True
    binarize_mask = False    
    opt.binarize_mask = False
    model = create_model(opt)
#    opt.phase = 'train'
#    data_loader = CreateDataLoader(opt)
#    tr_dataset = data_loader.load_data()
#    tr_dataset_size = len(data_loader)
#    if binarize_mask:
#        opt.bin_thr = find_thr_for_bin(model, opt, tr_dataset)
#        opt.binarize_mask = True
#        model = create_model(opt)
#        print('Threshold was made ', str(opt.bin_thr), ' .')
#    print('Training images = %d' % tr_dataset_size)
#    if opt.use_fixed_acc:
#        [tr_av_psnr1_A_c,tr_av_psnr1_A,tr_av_psnr2_A,tr_av_l1_A_c,tr_av_l1_A,tr_av_vgg_l_A_c,tr_av_vgg_l_A,tr_msk_mean_A],[tr_av_psnr1_B_c,tr_av_psnr1_B,tr_av_psnr2_B,tr_av_l1_B_c,tr_av_l1_B,tr_av_vgg_l_B_c,tr_av_vgg_l_B,tr_msk_mean_B] = aver_psnr_l1_vgg(model, opt, tr_dataset, tr_dataset_size)
#    else:
#        tr_av_psnr1, tr_av_psnr2, tr_av_l1, tr_av_vgg_l, tr_av_us_l = aver_psnr_l1_vgg(model, opt, tr_dataset, tr_dataset_size)
#    opt.phase = 'val'
#    data_loader = CreateDataLoader(opt)
#    val_dataset = data_loader.load_data()
#    val_dataset_size = len(data_loader)
#    print('Validation images = %d' % val_dataset_size)    
#    if opt.use_fixed_acc:
#        [val_av_psnr1_A_c,val_av_psnr1_A,val_av_psnr2_A,val_av_l1_A_c,val_av_l1_A,val_av_vgg_l_A_c,val_av_vgg_l_A,val_msk_mean_A],[val_av_psnr1_B_c,val_av_psnr1_B,val_av_psnr2_B,val_av_l1_B_c,val_av_l1_B,val_av_vgg_l_B_c,val_av_vgg_l_B,val_msk_mean_B] = aver_psnr_l1_vgg(model, opt, val_dataset, val_dataset_size)
#    else:
#        val_av_psnr1, val_av_psnr2, val_av_l1, val_av_vgg_l, val_av_us_l = aver_psnr_l1_vgg(model, opt, val_dataset, val_dataset_size)
    opt.phase = 'val'
    data_loader = CreateDataLoader(opt)
    test_dataset = data_loader.load_data()
    test_dataset_size = len(data_loader)
    print('Test images = %d' % test_dataset_size) 
    #analyze_sym(model, opt, test_dataset)
    display_remafa_coil_comb(model, opt, test_dataset, output_dir)
    #[tst_av_psnr1_T1_c,tst_av_psnr1_T1, tst_av_ssim_T1_c, tst_av_ssim_T1, tst_msk_mean_T1], [tst_av_psnr1_T2_c,tst_av_psnr1_T2, tst_av_ssim_T2_c, tst_av_ssim_T2, tst_msk_mean_T2], [tst_av_psnr1_PD_c,tst_av_psnr1_PD,tst_av_ssim_PD_c, tst_av_ssim_PD,tst_msk_mean_PD], masks_T1, masks_T2, masks_PD = aver_psnr_l1_vgg_glo_range(model, opt, test_dataset, test_dataset_size, return_also_masks=True)
    print('----------------------------Results----------------------------------')   
    #print(masks_T1[2].mean().item()*64.0, masks_T2[2].mean().item()*64.0, masks_PD[2].mean().item()*64.0, (masks_T1[2].mean().item()+masks_T2[2].mean().item()+masks_PD[2].mean().item())*64.0/3.0)
    print('Net: %s, Conf: %s, Epoch: %i' % (opt.net_name, opt.run_name, opt.which_epoch))
    #print('Test PSNR1: %.4f (T1: %.3f, T2: %.3f, PD: %.3f), Test PSNR1_c: %.4f (T1: %.3f, T2: %.3f, PD: %.3f), Test SSIM_c: %.5f (T1: %.5f, T2: %.5f, PD: %.3f), Test SSIM: %.5f (T1: %.5f, T2: %.5f, PD: %.3f), Tst US rate: %.3f (T1: %.3f, T2: %.3f, PD: %.3f)' % ((tst_av_psnr1_T1+tst_av_psnr1_T2+tst_av_psnr1_PD)/3, tst_av_psnr1_T1, tst_av_psnr1_T2, tst_av_psnr1_PD, (tst_av_psnr1_T1_c+tst_av_psnr1_T2_c+tst_av_psnr1_PD_c)/3, tst_av_psnr1_T1_c, tst_av_psnr1_T2_c, tst_av_psnr1_PD_c, (tst_av_ssim_T1_c+tst_av_ssim_T2_c+tst_av_ssim_PD_c)/3, tst_av_ssim_T1_c, tst_av_ssim_T2_c, tst_av_ssim_PD_c, (tst_av_ssim_T1+tst_av_ssim_T2+tst_av_ssim_PD)/3, tst_av_ssim_T1, tst_av_ssim_T2, tst_av_ssim_PD, (tst_msk_mean_T1+tst_msk_mean_T2+tst_msk_mean_PD)/3, tst_msk_mean_T1, tst_msk_mean_T2, tst_msk_mean_PD))
    [tst_av_psnr1_T1_c,tst_av_psnr1_T1, tst_av_ssim_T1_c, tst_av_ssim_T1, tst_msk_mean_T1], [tst_av_psnr1_T2_c,tst_av_psnr1_T2, tst_av_ssim_T2_c, tst_av_ssim_T2, tst_msk_mean_T2], [tst_av_psnr1_PD_c,tst_av_psnr1_PD,tst_av_ssim_PD_c, tst_av_ssim_PD,tst_msk_mean_PD], masks_T1, masks_T2, masks_PD = aver_psnr_l1_vgg_loc_range(model, opt, test_dataset, test_dataset_size, return_also_masks=True)
    print('----------------------------------------------------------------------')
    print('Test PSNR1: %.4f (T1: %.3f, T2: %.3f, PD: %.3f), Test PSNR1_c: %.4f (T1: %.3f, T2: %.3f, PD: %.3f), Test SSIM_c: %.5f (T1: %.5f, T2: %.5f, PD: %.3f), Test SSIM: %.5f (T1: %.5f, T2: %.5f, PD: %.3f), Tst US rate: %.3f (T1: %.3f, T2: %.3f, PD: %.3f)' % ((tst_av_psnr1_T1+tst_av_psnr1_T2+tst_av_psnr1_PD)/3, tst_av_psnr1_T1, tst_av_psnr1_T2, tst_av_psnr1_PD, (tst_av_psnr1_T1_c+tst_av_psnr1_T2_c+tst_av_psnr1_PD_c)/3, tst_av_psnr1_T1_c, tst_av_psnr1_T2_c, tst_av_psnr1_PD_c, (tst_av_ssim_T1_c+tst_av_ssim_T2_c+tst_av_ssim_PD_c)/3, tst_av_ssim_T1_c, tst_av_ssim_T2_c, tst_av_ssim_PD_c, (tst_av_ssim_T1+tst_av_ssim_T2+tst_av_ssim_PD)/3, tst_av_ssim_T1, tst_av_ssim_T2, tst_av_ssim_PD, (tst_msk_mean_T1+tst_msk_mean_T2+tst_msk_mean_PD)/3, tst_msk_mean_T1, tst_msk_mean_T2, tst_msk_mean_PD))
    print('Number of taken pixels are ' + str(masks_T1[2].sum().cpu().detach().numpy()[()]) + ' on T1 and ' + str(masks_T2[2].sum().cpu().detach().numpy()[()]) + ' on T2 and ' + str(masks_PD[2].sum().cpu().detach().numpy()[()]) + ' on PD.')
    print('----------------------------------------------------------------------')
    #display_remafa(model, opt, test_dataset, output_dir)
    print('ReMaFa image was saved.')
    #display_mask_hists(opt, [masks_B,masks_A], output_dir)
    print('Mask hists image was saved.')  
def display_mask_hists(opt, masks, output_dir):
    fig,axes = plt.subplots(1,4+2*opt.use_fixed_acc,figsize=(36,4.3))
    plot_mask_hist(masks[0][0], axes[0], 'Prb Mask T1')
    if opt.use_fixed_acc:
        plot_mask_hist(masks[0][1], axes[1], 'Rescd Mask T1')
        plot_mask_hist(masks[0][2], axes[2], 'Thrdd Mask T1')   
    else:
        plot_mask_hist(masks[0][1], axes[1], 'Thrdd Mask T1') 
    if opt.use_fixed_acc:
        plot_mask_hist(masks[1][0], axes[3], 'Prb Mask T2')
        plot_mask_hist(masks[1][1], axes[4], 'Rescd Mask T2')
        plot_mask_hist(masks[1][2], axes[5], 'Thrdd Mask T2')   
    else:
        plot_mask_hist(masks[1][0], axes[2], 'Prb Mask T2')
        plot_mask_hist(masks[1][1], axes[3], 'Thrdd Mask T2') 
    plt.suptitle('Mask Hists with Conf.: %s' % opt.run_name)
    plt.show()
    plt.savefig(output_dir + '/mask_hists_hrd' + str(opt.use_hrd_thr) + '.png', bbox_inches='tight')
def plot_mask_hist(mask, plot_ax, title):
    mask = np.fft.fftshift(mask.data.squeeze(-1).squeeze(1).cpu().float().numpy())
    freq, bins = np.histogram(mask, histedges_equalN(mask.flatten(), 10))
    plot_ax.bar(np.linspace(0,1,11)[:-1], freq,width=0.1,align='edge') 
    plot_ax.set_xticks(np.linspace(0,1,11))
    labels = []
    for i in range(bins.size):
        labels.append(('%.1e' % bins[i]))
    plot_ax.set_xticklabels(labels)
    plot_ax.tick_params(axis='both',which='major',labelsize=6)
    plot_ax.set_title(title)

#def test(opt):
#    opt.use_hrd_thr = (input('Do you want to use hard thresholding (1) or sigmoid (2)? ') == 1)
#    model = create_model(opt)
#    opt.phase = 'train'
#    data_loader = CreateDataLoader(opt)
#    tr_dataset = data_loader.load_data()
#    tr_dataset_size = len(data_loader)
#    print('Training images = %d' % tr_dataset_size)
#    if opt.use_fixed_acc:
#        tr_av_psnr1, tr_av_psnr2, tr_av_l1, tr_av_vgg_l = aver_psnr_l1_vgg(model, opt, tr_dataset, tr_dataset_size)
#    else:
#        tr_av_psnr1, tr_av_psnr2, tr_av_l1, tr_av_vgg_l, tr_av_us_l = aver_psnr_l1_vgg(model, opt, tr_dataset, tr_dataset_size)
#    display_remafa(model, opt, tr_dataset)
#    opt.phase = 'val'
#    data_loader = CreateDataLoader(opt)
#    val_dataset = data_loader.load_data()
#    val_dataset_size = len(data_loader)
#    print('Validation images = %d' % val_dataset_size)    
#    if opt.use_fixed_acc:
#        val_av_psnr1, val_av_psnr2, val_av_l1, val_av_vgg_l = aver_psnr_l1_vgg(model, opt, val_dataset, val_dataset_size)
#    else:
#        val_av_psnr1, val_av_psnr2, val_av_l1, val_av_vgg_l, val_av_us_l = aver_psnr_l1_vgg(model, opt, val_dataset, val_dataset_size)
#    opt.phase = 'test'
#    data_loader = CreateDataLoader(opt)
#    test_dataset = data_loader.load_data()
#    test_dataset_size = len(data_loader)
#    print('Test images = %d' % test_dataset_size) 
#    if opt.use_fixed_acc:
#        tst_av_psnr1, tst_av_psnr2, tst_av_l1, tst_av_vgg_l, masks = aver_psnr_l1_vgg(model, opt, test_dataset, test_dataset_size, return_also_masks=True)
#    else:
#        tst_av_psnr1, tst_av_psnr2, tst_av_l1, tst_av_vgg_l, tst_av_us_l, masks = aver_psnr_l1_vgg(model, opt, test_dataset, test_dataset_size, return_also_masks=True)
#    print('----------------------------Results----------------------------------')   
#    print('Net: %s, Conf: %s, Epoch: %i' % (opt.net_name, opt.run_name, opt.which_epoch))
#    if opt.use_fixed_acc:
#        print('Train PSNR1: %.3f, Train PSNR2: %.3f, Train L1 Loss: %.3f, Train VGG Loss: %.3f' % (tr_av_psnr1, tr_av_psnr2, tr_av_l1, tr_av_vgg_l))
#        print('Val PSNR1: %.3f, Val PSNR2: %.3f, Val L1 Loss: %.3f, Val VGG Loss: %.3f' % (val_av_psnr1, val_av_psnr2, val_av_l1, val_av_vgg_l))
#        print('Test PSNR1: %.3f, Test PSNR2: %.3f, Tst L1 Loss: %.3f, Tst VGG Loss: %.3f' % (tst_av_psnr1, tst_av_psnr2, tst_av_l1, tst_av_vgg_l))
#    else:
#        print('Train PSNR1: %.3f, Train PSNR2: %.3f, Train L1 Loss: %.3f, Train VGG Loss: %.3f, Train US Loss: %.3f' % (tr_av_psnr1, tr_av_psnr2, tr_av_l1, tr_av_vgg_l, tr_av_us_l))
#        print('Val PSNR1: %.3f, Val PSNR2: %.3f, Val L1 Loss: %.3f, Val VGG Loss: %.3f, Val US Loss: %.3f' % (val_av_psnr1, val_av_psnr2, val_av_l1, val_av_vgg_l, val_av_us_l))
#        print('Test PSNR1: %.3f, Test PSNR2: %.3f, Tst L1 Loss: %.3f, Tst VGG Loss: %.3f, Tst US Loss: %.3f' % (tst_av_psnr1, tst_av_psnr2, tst_av_l1, tst_av_vgg_l, tst_av_us_l))
#    print('----------------------------------------------------------------------')
#    display_masks(opt, masks)
#    display_remafa(model, opt, tr_dataset)
#    fig,axes = plt.subplots(1,2+opt.use_fixed_acc,figsize=(18,4.3))
#    plot_mask_hist(masks[0], axes[0], 'Prb Mask')
#    if opt.use_fixed_acc:
#        plot_mask_hist(masks[1], axes[1], 'Rescd Mask')
#        plot_mask_hist(masks[2], axes[2], 'Thrdd Mask')   
#    else:
#        plot_mask_hist(masks[1], axes[1], 'Thrdd Mask') 
#    plt.suptitle('RSGAN with Conf.: %s' % opt.run_name)
#    plt.show()
#    plotOrNot = (input('Do you want to plot the psnr as a function of epochs (1) or not(0)? ') == 1)
#    if plotOrNot:
#        fig,axes = plt.subplots(1,3,figsize=(15,4.3))
#        psnr_meth = input('Do you want PSNR calc method 1 or 2? ')
#        tr_psnr_vs_ep=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_psnr_vs_ep'+str(psnr_meth)+'.npy'))
#        val_psnr_vs_ep=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_psnr_vs_ep'+str(psnr_meth)+'.npy'))
#        axes[0].plot(tr_psnr_vs_ep, label="Aver on Train Set")
#        axes[0].plot(val_psnr_vs_ep, label="Aver on Val Set")
#        axes[0].set_title('PSNR vs Epoch')
#        axes[0].legend(loc="best")
#        axes[0].set_ylabel('PSNR')
#        axes[0].set_xlabel('Epoch No')
#        
#        tr_l1_vs_ep=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_l1_vs_ep.npy'))
#        val_l1_vs_ep=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_l1_vs_ep.npy'))
#        axes[1].plot(tr_l1_vs_ep, label="Aver on Train Set")
#        axes[1].plot(val_l1_vs_ep, label="Aver on Val Set")
#        axes[1].set_title('L1 Loss vs Epoch')
#        axes[1].legend(loc="best")
#        axes[1].set_ylabel('L1 Loss')
#        axes[1].set_xlabel('Epoch No')
#   
#        tr_vgg_vs_ep=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'tr_vgg_vs_ep.npy'))
#        val_vgg_vs_ep=np.load(os.path.join(opt.checkpoints_dir, opt.run_name, 'val_vgg_vs_ep.npy'))
#        axes[2].plot(tr_vgg_vs_ep, label="Aver on Train Set")
#        axes[2].plot(val_vgg_vs_ep, label="Aver on Val Set")
#        axes[2].set_title('VGG Loss vs Epoch')
#        axes[2].legend(loc="best")
#        axes[2].set_ylabel('VGG Loss')
#        axes[2].set_xlabel('Epoch No')
#        plt.suptitle('RSGAN with Conf.: %s' % opt.run_name)
#        plt.show()
def analyze_sym(model, opt, dataset):
    for i, data in enumerate(dataset):
        model.set_input(data)
        takens = 0
        while takens != round(opt.ur*256*256*2):
            (real_A,fake_A,fake_A_c,masks_A),(real_B,fake_B,fake_B_c,masks_B) = model.test()
            takens = masks_B[2].sum().cpu().detach().numpy()[()] + masks_A[2].sum().cpu().detach().numpy()[()]
            print(takens)
        print('Number of taken pixels in display_remafa are ' + str(masks_B[2].sum().cpu().detach().numpy()[()]) + ' on T1 and ' + str(masks_A[2].sum().cpu().detach().numpy()[()]) + ' on T2.')
        mask_bin_A = np.transpose(np.tile(np.fft.fftshift(masks_A[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1)), (1, 2, 0))  
        mask_bin_B = np.transpose(np.tile(np.fft.fftshift(masks_B[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1)), (1, 2, 0)) 
        break
    conj_mask = np.zeros((256,256))
    conj_mask[1:128,:]=1
    conj_mask[0,0]=1
    conj_mask[0,128:]=1
    conj_mask[128,0]=1
    conj_mask[128,128:]=1
    no_sym_B, no_not_sym_B, sym_mask_B, not_sym_mask_B = analyze_mask_symmetry(mask_bin_B[:,:,0],conj_mask)
    no_sym_A, no_not_sym_A, sym_mask_A, not_sym_mask_A = analyze_mask_symmetry(mask_bin_A[:,:,0],conj_mask)
    fig,ax=plt.subplots(1,4,figsize=(12,3))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
    ax[0].imshow(sym_mask_B)
    ax[0].axis('off')
    ax[1].imshow(not_sym_mask_B)
    ax[1].axis('off')
    ax[2].imshow(sym_mask_A)
    ax[2].axis('off')
    ax[3].imshow(not_sym_mask_A)
    ax[3].axis('off')
    plt.show()
    print('SYMS: ', no_sym_B, no_not_sym_B, no_sym_A, no_not_sym_A)
        
        
def aver_psnr_l1_vgg_loc_range(model, opt, dataset, dataset_size, return_also_masks=False):
    #psnr_calc_cons = 10*math.log10((2**2)*256*256)
    tot_psnr1_T1 = 0
    tot_psnr1_T1_c = 0
    tot_ssim_T1 = 0
    tot_ssim_T1_c = 0
#    tot_vgg_l_T1 = 0
#    tot_l1_l_T1 = 0
#    tot_vgg_l_T1_c = 0
#    tot_l1_l_T1_c = 0
    tot_psnr1_T2 = 0
    tot_psnr1_T2_c = 0
    tot_ssim_T2 = 0
    tot_ssim_T2_c = 0
#    tot_vgg_l_T2 = 0
#    tot_l1_l_T2 = 0
#    tot_vgg_l_T2_c = 0
#    tot_l1_l_T2_c = 0
    tot_psnr1_PD = 0
    tot_psnr1_PD_c = 0
    tot_ssim_PD = 0
    tot_ssim_PD_c = 0
#    tot_vgg_l_PD = 0
#    tot_l1_l_PD = 0
#    tot_vgg_l_PD_c = 0
#    tot_l1_l_PD_c = 0
    min_ss = 1
    max_ss = 0
    for i, data in enumerate(dataset):
        model.set_input(data)
        (real_t1,fake_t1,fake_t1_c,masks_t1),(real_t2,fake_t2,fake_t2_c,masks_t2),(real_pd,fake_pd,fake_pd_c,masks_pd) = model.test() #masks[0].shape=1,1,192,88,1
        fake_t1_c[fake_t1_c<-1] = -1
        fake_t1_c[fake_t1_c>1] = 1
        fake_t2_c[fake_t2_c<-1] = -1
        fake_t2_c[fake_t2_c>1] = 1
        fake_pd_c[fake_pd_c<-1] = -1
        fake_pd_c[fake_pd_c>1] = 1
        
        all_reals = torch.cat((real_t1,real_t2,real_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        all_reals = ((all_reals[...,0] ** 2) + (all_reals[...,1] ** 2)).sum(1).sqrt().detach().cpu().numpy()


#        all_reals = torch.cat((real_t1,real_t2,real_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
#        all_reals = torch.stack(((all_reals[...,0]*model.maps[...,0] + all_reals[...,1]*model.maps[...,1]).sum(1), (all_reals[...,1]*model.maps[...,0] - all_reals[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
#        all_reals = np.hypot(all_reals[...,0],all_reals[...,1])
        
        all_fakes = torch.cat((fake_t1,fake_t2,fake_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        #all_fakes = torch.stack(((all_fakes[...,0]*model.maps[...,0] + all_fakes[...,1]*model.maps[...,1]).sum(1), (all_fakes[...,1]*model.maps[...,0] - all_fakes[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
        all_fakes = ((all_fakes[...,0] ** 2) + (all_fakes[...,1] ** 2)).sum(1).sqrt().detach().cpu().numpy()
        #all_fakes = np.hypot(all_fakes[...,0],all_fakes[...,1])
        
        all_fakes_c = torch.cat((fake_t1_c,fake_t2_c,fake_pd_c),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        all_fakes_c = torch.stack(((all_fakes_c[...,0]*model.maps[...,0] + all_fakes_c[...,1]*model.maps[...,1]).sum(1), (all_fakes_c[...,1]*model.maps[...,0] - all_fakes_c[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
        all_fakes_c = np.hypot(all_fakes_c[...,0],all_fakes_c[...,1])        
        
        psnr_calc_cons = 10*math.log10(((all_reals[0,:,:].max()-all_reals[0,:,:].min())**2)*1*192*88)
        tot_psnr1_T1 += psnr_calc_cons-10*math.log10(np.sum((all_reals[0,:,:]-all_fakes[0,:,:])**2))
        tot_psnr1_T1_c += psnr_calc_cons-10*math.log10(np.sum((all_reals[0,:,:]-all_fakes_c[0,:,:])**2))
        psnr_calc_cons = 10*math.log10(((all_reals[1,:,:].max()-all_reals[1,:,:].min())**2)*1*192*88)
        tot_psnr1_T2 += psnr_calc_cons-10*math.log10(np.sum((all_reals[1,:,:]-all_fakes[1,:,:])**2))
        tot_psnr1_T2_c += psnr_calc_cons-10*math.log10(np.sum((all_reals[1,:,:]-all_fakes_c[1,:,:])**2))
        psnr_calc_cons = 10*math.log10(((all_reals[2,:,:].max()-all_reals[2,:,:].min())**2)*1*192*88)
        tot_psnr1_PD += psnr_calc_cons-10*math.log10(np.sum((all_reals[2,:,:]-all_fakes[2,:,:])**2))
        tot_psnr1_PD_c += psnr_calc_cons-10*math.log10(np.sum((all_reals[2,:,:]-all_fakes_c[2,:,:])**2))
        tot_ssim_T1 += ssim(all_reals[0,:,:], all_fakes[0,:,:], data_range=(all_reals[0,:,:].max()-all_reals[0,:,:].min()), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_T1_c += ssim(all_reals[0,:,:], all_fakes_c[0,:,:], data_range=(all_reals[0,:,:].max()-all_reals[0,:,:].min()), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_T2 += ssim(all_reals[1,:,:], all_fakes[1,:,:], data_range=(all_reals[1,:,:].max()-all_reals[1,:,:].min()), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_T2_c += ssim(all_reals[1,:,:], all_fakes_c[1,:,:], data_range=(all_reals[1,:,:].max()-all_reals[1,:,:].min()), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_PD += ssim(all_reals[2,:,:], all_fakes[2,:,:], data_range=(all_reals[2,:,:].max()-all_reals[2,:,:].min()), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_PD_c += ssim(all_reals[2,:,:], all_fakes_c[2,:,:], data_range=(all_reals[2,:,:].max()-all_reals[2,:,:].min()), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        
#        real_t1 = (real_t1.cpu().detach().numpy()[0,:,:,:])
#        fake_t1 = (fake_t1.cpu().detach().numpy()[0,:,:,:])
#        fake_t1_c = (fake_t1_c.cpu().detach().numpy()[0,:,:,:])        
#        abs_real_t1 = np.zeros((5,192,88))
#        abs_fake_t1 = np.zeros((5,192,88))
#        abs_fake_t1_c = np.zeros((5,192,88))
#       
#        real_t2 = (real_t2.cpu().detach().numpy()[0,:,:,:])
#        fake_t2 = (fake_t2.cpu().detach().numpy()[0,:,:,:])
#        fake_t2_c = (fake_t2_c.cpu().detach().numpy()[0,:,:,:])
#        abs_real_t2 = np.zeros((5,192,88))
#        abs_fake_t2 = np.zeros((5,192,88))
#        abs_fake_t2_c = np.zeros((5,192,88))
#
#        real_pd = (real_pd.cpu().detach().numpy()[0,:,:,:])
#        fake_pd = (fake_pd.cpu().detach().numpy()[0,:,:,:])
#        fake_pd_c = (fake_pd_c.cpu().detach().numpy()[0,:,:,:])
#        abs_real_pd = np.zeros((5,192,88))
#        abs_fake_pd = np.zeros((5,192,88))
#        abs_fake_pd_c = np.zeros((5,192,88))
#        
#        for i in range(5):
#            abs_real_t1[i,:,:] = np.hypot(real_t1[2*i,:,:], real_t1[2*i+1,:,:])
#            abs_fake_t1[i,:,:] = np.hypot(fake_t1[2*i,:,:], fake_t1[2*i+1,:,:])
#            abs_fake_t1_c[i,:,:] = np.hypot(fake_t1_c[2*i,:,:], fake_t1_c[2*i+1,:,:])
#            
#            abs_real_t2[i,:,:] = np.hypot(real_t2[2*i,:,:], real_t2[2*i+1,:,:])
#            abs_fake_t2[i,:,:] = np.hypot(fake_t2[2*i,:,:], fake_t2[2*i+1,:,:])
#            abs_fake_t2_c[i,:,:] = np.hypot(fake_t2_c[2*i,:,:], fake_t2_c[2*i+1,:,:])
#            
#            abs_real_pd[i,:,:] = np.hypot(real_pd[2*i,:,:], real_pd[2*i+1,:,:])
#            abs_fake_pd[i,:,:] = np.hypot(fake_pd[2*i,:,:], fake_pd[2*i+1,:,:])
#            abs_fake_pd_c[i,:,:] = np.hypot(fake_pd_c[2*i,:,:], fake_pd_c[2*i+1,:,:])
#        
#        
#        
#        for i in range(5):
#            psnr_calc_cons = 10*math.log10(((abs_real_t1[i,:,:].max()-abs_real_t1[i,:,:].min())**2)*1*192*88*opt.batchSize)
#            tot_psnr1_T1 += psnr_calc_cons-10*math.log10(np.sum((abs_real_t1[i,:,:]-abs_fake_t1[i,:,:])**2))
#            tot_psnr1_T1_c += psnr_calc_cons-10*math.log10(np.sum((abs_real_t1[i,:,:]-abs_fake_t1_c[i,:,:])**2))
#            psnr_calc_cons = 10*math.log10(((abs_real_t2[i,:,:].max()-abs_real_t2[i,:,:].min())**2)*1*192*88*opt.batchSize)
#            tot_psnr1_T2 += psnr_calc_cons-10*math.log10(np.sum((abs_real_t2[i,:,:]-abs_fake_t2[i,:,:])**2))
#            tot_psnr1_T2_c += psnr_calc_cons-10*math.log10(np.sum((abs_real_t2[i,:,:]-abs_fake_t2_c[i,:,:])**2))
#            psnr_calc_cons = 10*math.log10(((abs_real_pd[i,:,:].max()-abs_real_pd[i,:,:].min())**2)*1*192*88*opt.batchSize)
#            tot_psnr1_PD += psnr_calc_cons-10*math.log10(np.sum((abs_real_pd[i,:,:]-abs_fake_pd[i,:,:])**2))
#            tot_psnr1_PD_c += psnr_calc_cons-10*math.log10(np.sum((abs_real_pd[i,:,:]-abs_fake_pd_c[i,:,:])**2))
#            tot_ssim_T1 += ssim(abs_real_t1[i,:,:], abs_fake_t1[i,:,:], data_range=abs_real_t1[i,:,:].max()-abs_real_t1[i,:,:].min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_T1_c += ssim(abs_real_t1[i,:,:], abs_fake_t1_c[i,:,:], data_range=abs_real_t1[i,:,:].max()-abs_real_t1[i,:,:].min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_T2 += ssim(abs_real_t2[i,:,:], abs_fake_t2[i,:,:], data_range=abs_real_t2[i,:,:].max()-abs_real_t2[i,:,:].min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_T2_c += ssim(abs_real_t2[i,:,:], abs_fake_t2_c[i,:,:], data_range=abs_real_t2[i,:,:].max()-abs_real_t2[i,:,:].min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_PD += ssim(abs_real_pd[i,:,:], abs_fake_pd[i,:,:], data_range=abs_real_pd[i,:,:].max()-abs_real_pd[i,:,:].min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_PD_c += ssim(abs_real_pd[i,:,:], abs_fake_pd_c[i,:,:], data_range=abs_real_pd[i,:,:].max()-abs_real_pd[i,:,:].min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#        
##        if min_ss > ssim(real_A, fake_A, data_range=real_A.max()-real_A.min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False):#, K1=0.001, K2=0.003):
##            min_ss = ssim(real_A, fake_A, data_range=real_A.max()-real_A.min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)#, K1=0.001, K2=0.003)
##        if max_ss < ssim(real_A, fake_A, data_range=real_A.max()-real_A.min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False):#, K1=0.001, K2=0.003):
##            max_ss = ssim(real_A, fake_A, data_range=real_A.max()-real_A.min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)#, K1=0.001, K2=0.003)
##
##        (l1_l_A_c,l1_l_A,vgg_l_A_c,vgg_l_A),(l1_l_B_c,l1_l_B,vgg_l_B_c,vgg_l_B) = model.find_errors_in_testing()
##        tot_l1_l_A += l1_l_A
##        tot_l1_l_A_c += l1_l_A_c
##        tot_vgg_l_A += vgg_l_A
##        tot_vgg_l_A_c += vgg_l_A_c
##        tot_l1_l_B += l1_l_B
##        tot_l1_l_B_c += l1_l_B_c
##        tot_vgg_l_B += vgg_l_B
##        tot_vgg_l_B_c += vgg_l_B_c
#    dataset_size *= 5
    to_return = ([tot_psnr1_T1_c/dataset_size,tot_psnr1_T1/dataset_size,tot_ssim_T1_c/dataset_size,tot_ssim_T1/dataset_size,masks_t1[2].mean().cpu().detach().numpy()[()]],\
    [tot_psnr1_T1_c/dataset_size,tot_psnr1_T2/dataset_size,tot_ssim_T2_c/dataset_size,tot_ssim_T2/dataset_size,masks_t2[2].mean().cpu().detach().numpy()[()]],\
    [tot_psnr1_PD_c/dataset_size,tot_psnr1_PD/dataset_size,tot_ssim_PD_c/dataset_size,tot_ssim_PD/dataset_size,masks_pd[2].mean().cpu().detach().numpy()[()]])
    if return_also_masks:
        to_return += (masks_t1,masks_t2,masks_pd,)
    #print(min_ss, max_ss)
    return to_return
   
   
def aver_psnr_l1_vgg_glo_range(model, opt, dataset, dataset_size, return_also_masks=False):
    psnr_calc_cons = 10*math.log10((2**2)*1*192*88)
    tot_psnr1_T1 = 0
    tot_psnr1_T1_c = 0
    tot_ssim_T1 = 0
    tot_ssim_T1_c = 0
#    tot_vgg_l_T1 = 0
#    tot_l1_l_T1 = 0
#    tot_vgg_l_T1_c = 0
#    tot_l1_l_T1_c = 0
    tot_psnr1_T2 = 0
    tot_psnr1_T2_c = 0
    tot_ssim_T2 = 0
    tot_ssim_T2_c = 0
#    tot_vgg_l_T2 = 0
#    tot_l1_l_T2 = 0
#    tot_vgg_l_T2_c = 0
#    tot_l1_l_T2_c = 0
    tot_psnr1_PD = 0
    tot_psnr1_PD_c = 0
    tot_ssim_PD = 0
    tot_ssim_PD_c = 0
#    tot_vgg_l_PD = 0
#    tot_l1_l_PD = 0
#    tot_vgg_l_PD_c = 0
#    tot_l1_l_PD_c = 0
    min_ss = 1
    max_ss = 0
    for i, data in enumerate(dataset):
        model.set_input(data)
        
        (real_t1,fake_t1,fake_t1_c,masks_t1),(real_t2,fake_t2,fake_t2_c,masks_t2),(real_pd,fake_pd,fake_pd_c,masks_pd) = model.test() #masks[0].shape=1,1,192,88,1
        fake_t1_c[fake_t1_c<-1] = -1
        fake_t1_c[fake_t1_c>1] = 1
        fake_t2_c[fake_t2_c<-1] = -1
        fake_t2_c[fake_t2_c>1] = 1
        fake_pd_c[fake_pd_c<-1] = -1
        fake_pd_c[fake_pd_c>1] = 1
        
        all_reals = torch.cat((real_t1,real_t2,real_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        all_reals = ((all_reals[...,0] ** 2) + (all_reals[...,1] ** 2)).sum(1).sqrt().detach().cpu().numpy()
#        all_reals = torch.cat((real_t1,real_t2,real_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
#        all_reals = torch.stack(((all_reals[...,0]*model.maps[...,0] + all_reals[...,1]*model.maps[...,1]).sum(1), (all_reals[...,1]*model.maps[...,0] - all_reals[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
#        all_reals = np.hypot(all_reals[...,0],all_reals[...,1])
        
        all_fakes = torch.cat((fake_t1,fake_t2,fake_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        #all_fakes = torch.stack(((all_fakes[...,0]*model.maps[...,0] + all_fakes[...,1]*model.maps[...,1]).sum(1), (all_fakes[...,1]*model.maps[...,0] - all_fakes[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
        all_fakes = ((all_fakes[...,0] ** 2) + (all_fakes[...,1] ** 2)).sum(1).sqrt().detach().cpu().numpy()
        #all_fakes = np.hypot(all_fakes[...,0],all_fakes[...,1])
        
        all_fakes_c = torch.cat((fake_t1_c,fake_t2_c,fake_pd_c),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        all_fakes_c = torch.stack(((all_fakes_c[...,0]*model.maps[...,0] + all_fakes_c[...,1]*model.maps[...,1]).sum(1), (all_fakes_c[...,1]*model.maps[...,0] - all_fakes_c[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
        all_fakes_c = np.hypot(all_fakes_c[...,0],all_fakes_c[...,1])        
        
        
        tot_psnr1_T1 += psnr_calc_cons-10*math.log10(np.sum((all_reals[0,:,:]-all_fakes[0,:,:])**2))
        tot_psnr1_T1_c += psnr_calc_cons-10*math.log10(np.sum((all_reals[0,:,:]-all_fakes_c[0,:,:])**2))
        tot_psnr1_T2 += psnr_calc_cons-10*math.log10(np.sum((all_reals[1,:,:]-all_fakes[1,:,:])**2))
        tot_psnr1_T2_c += psnr_calc_cons-10*math.log10(np.sum((all_reals[1,:,:]-all_fakes_c[1,:,:])**2))
        tot_psnr1_PD += psnr_calc_cons-10*math.log10(np.sum((all_reals[2,:,:]-all_fakes[2,:,:])**2))
        tot_psnr1_PD_c += psnr_calc_cons-10*math.log10(np.sum((all_reals[2,:,:]-all_fakes_c[2,:,:])**2))
        tot_ssim_T1 += ssim(all_reals[0,:,:], all_fakes[0,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_T1_c += ssim(all_reals[0,:,:], all_fakes_c[0,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_T2 += ssim(all_reals[1,:,:], all_fakes[1,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_T2_c += ssim(all_reals[1,:,:], all_fakes_c[1,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_PD += ssim(all_reals[2,:,:], all_fakes[2,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        tot_ssim_PD_c += ssim(all_reals[2,:,:], all_fakes_c[2,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        
        
#        real_t1 = (real_t1.cpu().detach().numpy()[0,:,:,:])
#        fake_t1 = (fake_t1.cpu().detach().numpy()[0,:,:,:])
#        fake_t1_c = (fake_t1_c.cpu().detach().numpy()[0,:,:,:])        
#        abs_real_t1 = np.zeros((5,192,88))
#        abs_fake_t1 = np.zeros((5,192,88))
#        abs_fake_t1_c = np.zeros((5,192,88))
#       
#        real_t2 = (real_t2.cpu().detach().numpy()[0,:,:,:])
#        fake_t2 = (fake_t2.cpu().detach().numpy()[0,:,:,:])
#        fake_t2_c = (fake_t2_c.cpu().detach().numpy()[0,:,:,:])
#        abs_real_t2 = np.zeros((5,192,88))
#        abs_fake_t2 = np.zeros((5,192,88))
#        abs_fake_t2_c = np.zeros((5,192,88))
#
#        real_pd = (real_pd.cpu().detach().numpy()[0,:,:,:])
#        fake_pd = (fake_pd.cpu().detach().numpy()[0,:,:,:])
#        fake_pd_c = (fake_pd_c.cpu().detach().numpy()[0,:,:,:])
#        abs_real_pd = np.zeros((5,192,88))
#        abs_fake_pd = np.zeros((5,192,88))
#        abs_fake_pd_c = np.zeros((5,192,88))
#        
#        for i in range(5):
#            abs_real_t1[i,:,:] = np.hypot(real_t1[2*i,:,:], real_t1[2*i+1,:,:])
#            abs_fake_t1[i,:,:] = np.hypot(fake_t1[2*i,:,:], fake_t1[2*i+1,:,:])
#            abs_fake_t1_c[i,:,:] = np.hypot(fake_t1_c[2*i,:,:], fake_t1_c[2*i+1,:,:])
#            
#            abs_real_t2[i,:,:] = np.hypot(real_t2[2*i,:,:], real_t2[2*i+1,:,:])
#            abs_fake_t2[i,:,:] = np.hypot(fake_t2[2*i,:,:], fake_t2[2*i+1,:,:])
#            abs_fake_t2_c[i,:,:] = np.hypot(fake_t2_c[2*i,:,:], fake_t2_c[2*i+1,:,:])
#            
#            abs_real_pd[i,:,:] = np.hypot(real_pd[2*i,:,:], real_pd[2*i+1,:,:])
#            abs_fake_pd[i,:,:] = np.hypot(fake_pd[2*i,:,:], fake_pd[2*i+1,:,:])
#            abs_fake_pd_c[i,:,:] = np.hypot(fake_pd_c[2*i,:,:], fake_pd_c[2*i+1,:,:])
#        
        
#        for i in range(5):
#            tot_psnr1_T1 += psnr_calc_cons-10*math.log10(np.sum((abs_real_t1[i,:,:]-abs_fake_t1[i,:,:])**2))
#            tot_psnr1_T1_c += psnr_calc_cons-10*math.log10(np.sum((abs_real_t1[i,:,:]-abs_fake_t1_c[i,:,:])**2))
#            tot_psnr1_T2 += psnr_calc_cons-10*math.log10(np.sum((abs_real_t2[i,:,:]-abs_fake_t2[i,:,:])**2))
#            tot_psnr1_T2_c += psnr_calc_cons-10*math.log10(np.sum((abs_real_t2[i,:,:]-abs_fake_t2_c[i,:,:])**2))
#            tot_psnr1_PD += psnr_calc_cons-10*math.log10(np.sum((abs_real_pd[i,:,:]-abs_fake_pd[i,:,:])**2))
#            tot_psnr1_PD_c += psnr_calc_cons-10*math.log10(np.sum((abs_real_pd[i,:,:]-abs_fake_pd_c[i,:,:])**2))
#            tot_ssim_T1 += ssim(abs_real_t1[i,:,:], abs_fake_t1[i,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_T1_c += ssim(abs_real_t1[i,:,:], abs_fake_t1_c[i,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_T2 += ssim(abs_real_t2[i,:,:], abs_fake_t2[i,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_T2_c += ssim(abs_real_t2[i,:,:], abs_fake_t2_c[i,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_PD += ssim(abs_real_pd[i,:,:], abs_fake_pd[i,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
#            tot_ssim_PD_c += ssim(abs_real_pd[i,:,:], abs_fake_pd_c[i,:,:], data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
        
#        if min_ss > ssim(real_A, fake_A, data_range=real_A.max()-real_A.min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False):#, K1=0.001, K2=0.003):
#            min_ss = ssim(real_A, fake_A, data_range=real_A.max()-real_A.min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)#, K1=0.001, K2=0.003)
#        if max_ss < ssim(real_A, fake_A, data_range=real_A.max()-real_A.min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False):#, K1=0.001, K2=0.003):
#            max_ss = ssim(real_A, fake_A, data_range=real_A.max()-real_A.min(), gaussian_weights=True, sigma=1.5, use_sample_covariance=False)#, K1=0.001, K2=0.003)
#
#        (l1_l_A_c,l1_l_A,vgg_l_A_c,vgg_l_A),(l1_l_B_c,l1_l_B,vgg_l_B_c,vgg_l_B) = model.find_errors_in_testing()
#        tot_l1_l_A += l1_l_A
#        tot_l1_l_A_c += l1_l_A_c
#        tot_vgg_l_A += vgg_l_A
#        tot_vgg_l_A_c += vgg_l_A_c
#        tot_l1_l_B += l1_l_B
#        tot_l1_l_B_c += l1_l_B_c
#        tot_vgg_l_B += vgg_l_B
#        tot_vgg_l_B_c += vgg_l_B_c
    #dataset_size *= 5
        
    to_return = ([tot_psnr1_T1_c/dataset_size,tot_psnr1_T1/dataset_size,tot_ssim_T1_c/dataset_size,tot_ssim_T1/dataset_size,masks_t1[2].mean().cpu().detach().numpy()[()]],\
    [tot_psnr1_T2_c/dataset_size,tot_psnr1_T2/dataset_size,tot_ssim_T2_c/dataset_size,tot_ssim_T2/dataset_size,masks_t2[2].mean().cpu().detach().numpy()[()]],\
    [tot_psnr1_PD_c/dataset_size,tot_psnr1_PD/dataset_size,tot_ssim_PD_c/dataset_size,tot_ssim_PD/dataset_size,masks_pd[2].mean().cpu().detach().numpy()[()]])
    if return_also_masks:
        to_return += (masks_t1,masks_t2,masks_pd,)
    print(min_ss, max_ss)
    return to_return
   
#def aver_psnr_l1_vgg_glo_range(model, opt, dataset, dataset_size, return_also_masks=False):
#    psnr_calc_cons = 10*math.log10((2**2)*256*256)
#    tot_psnr1_A = 0
#    tot_psnr1_A_c = 0
#    tot_ssim_A = 0
#    tot_ssim_A_c = 0
#    tot_vgg_l_A = 0
#    tot_l1_l_A = 0
#    tot_vgg_l_A_c = 0
#    tot_l1_l_A_c = 0
#    tot_psnr1_B = 0
#    tot_psnr1_B_c = 0
#    tot_ssim_B = 0
#    tot_ssim_B_c = 0
#    tot_vgg_l_B = 0
#    tot_l1_l_B = 0
#    tot_vgg_l_B_c = 0
#    tot_l1_l_B_c = 0
#    for i, data in enumerate(dataset):
#        model.set_input(data)
#        (real_A,fake_A,fake_A_c,masks_A),(real_B,fake_B,fake_B_c,masks_B) = model.test()
#        fake_A_c[fake_A_c<-1] = -1
#        fake_A_c[fake_A_c>1] = 1
#        fake_B_c[fake_B_c<-1] = -1
#        fake_B_c[fake_B_c>1] = 1
#        
#        real_A = (real_A.cpu().detach().numpy()[0,0,:,:])
#        fake_A = (fake_A.cpu().detach().numpy()[0,0,:,:])
#        fake_A_c = (fake_A_c.cpu().detach().numpy()[0,0,:,:])
#       
#        real_B = (real_B.cpu().detach().numpy()[0,0,:,:])
#        fake_B = (fake_B.cpu().detach().numpy()[0,0,:,:])
#        fake_B_c = (fake_B_c.cpu().detach().numpy()[0,0,:,:])
#        
#        if (i % 100) == 0:
#            print(real_A.mean(), fake_A.mean(),fake_A_c.mean())
#            print(real_B.mean(), fake_B.mean(),fake_B_c.mean())
#        tot_psnr1_A += psnr_calc_cons-10*math.log10(np.sum((real_A-fake_A)**2))
#        tot_psnr1_A_c += psnr_calc_cons-10*math.log10(np.sum((real_A-fake_A_c)**2))
#        tot_ssim_A += ssim(real_A, fake_A, data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)#, K1=0.1, K2=0.03)
#        tot_ssim_A_c += ssim(real_A, fake_A_c, data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)#, K1=0.01, K2=0.03)
#        tot_psnr1_B += psnr_calc_cons-10*math.log10(np.sum((real_B-fake_B)**2))
#        tot_psnr1_B_c += psnr_calc_cons-10*math.log10(np.sum((real_B-fake_B_c)**2))
#        tot_ssim_B += ssim(real_B, fake_B, data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)#, K1=0.01, K2=0.03)
#        tot_ssim_B_c += ssim(real_B, fake_B_c, data_range=2, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)#, K1=0.01, K2=0.03)
#        (l1_l_A_c,l1_l_A,vgg_l_A_c,vgg_l_A),(l1_l_B_c,l1_l_B,vgg_l_B_c,vgg_l_B) = model.find_errors_in_testing()
#        tot_l1_l_A += l1_l_A
#        tot_l1_l_A_c += l1_l_A_c
#        tot_vgg_l_A += vgg_l_A
#        tot_vgg_l_A_c += vgg_l_A_c
#        tot_l1_l_B += l1_l_B
#        tot_l1_l_B_c += l1_l_B_c
#        tot_vgg_l_B += vgg_l_B
#        tot_vgg_l_B_c += vgg_l_B_c
#    to_return = ([tot_psnr1_A_c/dataset_size,tot_psnr1_A/dataset_size,tot_ssim_A_c/dataset_size,tot_ssim_A/dataset_size,tot_l1_l_A_c/dataset_size,tot_l1_l_A/dataset_size,tot_vgg_l_A_c/dataset_size,tot_vgg_l_A/dataset_size,masks_A[2].mean().cpu().detach().numpy()[()]],\
#    [tot_psnr1_B_c/dataset_size,tot_psnr1_B/dataset_size,tot_ssim_B_c/dataset_size,tot_ssim_B/dataset_size,tot_l1_l_B_c/dataset_size,tot_l1_l_B/dataset_size,tot_vgg_l_B_c/dataset_size,tot_vgg_l_B/dataset_size,masks_B[2].mean().cpu().detach().numpy()[()]])
#    if return_also_masks:
#        to_return += (masks_A,masks_B,)
#    return to_return
#    
def aver_psnr_l1_vgg_train(model, opt, dataset, dataset_size, return_also_masks=False):
    psnr_calc_cons = 10*math.log10((2**2)*10*192*88)
    tot_psnr1_t1 = 0
    tot_psnr1_t1_c = 0
    tot_psnr1_t2 = 0
    tot_psnr1_t2_c = 0
    tot_psnr1_pd = 0
    tot_psnr1_pd_c = 0
    for i, data in enumerate(dataset):
        model.set_input(data)
        (real_t1,fake_t1,fake_t1_c,masks_t1),(real_t2,fake_t2,fake_t2_c,masks_t2),(real_pd,fake_pd,fake_pd_c,masks_pd) = model.test()
        fake_t1_c[fake_t1_c<-1] = -1
        fake_t1_c[fake_t1_c>1] = 1
        fake_t2_c[fake_t2_c<-1] = -1
        fake_t2_c[fake_t2_c>1] = 1
        fake_pd_c[fake_pd_c<-1] = -1
        fake_pd_c[fake_pd_c>1] = 1
        
        
        real_t1 = (real_t1.cpu().detach().numpy()[0,:,:,:])
        fake_t1 = (fake_t1.cpu().detach().numpy()[0,:,:,:])
        fake_t1_c = (fake_t1_c.cpu().detach().numpy()[0,:,:,:])        
       
        real_t2 = (real_t2.cpu().detach().numpy()[0,:,:,:])
        fake_t2 = (fake_t2.cpu().detach().numpy()[0,:,:,:])
        fake_t2_c = (fake_t2_c.cpu().detach().numpy()[0,:,:,:])

        real_pd = (real_pd.cpu().detach().numpy()[0,:,:,:])
        fake_pd = (fake_pd.cpu().detach().numpy()[0,:,:,:])
        fake_pd_c = (fake_pd_c.cpu().detach().numpy()[0,:,:,:])
        
        tot_psnr1_t1 += psnr_calc_cons-10*math.log10(np.sum((real_t1-fake_t1)**2))
        tot_psnr1_t1_c += psnr_calc_cons-10*math.log10(np.sum((real_t1-fake_t1_c)**2))
        tot_psnr1_t2 += psnr_calc_cons-10*math.log10(np.sum((real_t2-fake_t2)**2))
        tot_psnr1_t2_c += psnr_calc_cons-10*math.log10(np.sum((real_t2-fake_t2_c)**2))
        tot_psnr1_pd += psnr_calc_cons-10*math.log10(np.sum((real_pd-fake_pd)**2))
        tot_psnr1_pd_c += psnr_calc_cons-10*math.log10(np.sum((real_pd-fake_pd_c)**2))
    to_return = ([tot_psnr1_t1_c/dataset_size,tot_psnr1_t1/dataset_size],\
    [tot_psnr1_t2_c/dataset_size,tot_psnr1_t2/dataset_size],[tot_psnr1_pd_c/dataset_size,tot_psnr1_pd/dataset_size])
    return to_return
def display_remafa_coil_comb(model, opt, dataset, output_dir):
    print(output_dir)
    print('Remafa will be displayed...')
    datasetind = 90
    for i, data in enumerate(dataset):
        if i != datasetind:
            continue
        #datasetind += 10
        model.set_input(data)
        takens = 0
        while takens != round(opt.ur*192*88*3):
            (real_t1,fake_t1,fake_t1_c,masks_t1),(real_t2,fake_t2,fake_t2_c,masks_t2),(real_pd,fake_pd,fake_pd_c,masks_pd) = model.test()
            takens = masks_t1[2].sum().cpu().detach().numpy()[()] + masks_t2[2].sum().cpu().detach().numpy()[()] + masks_pd[2].sum().cpu().detach().numpy()[()]
            #print(takens)
        print('Number of taken pixels in display_remafa are ' + str(masks_t1[2].sum().cpu().detach().numpy()[()]) + ' on T1 and ' + str(masks_t2[2].sum().cpu().detach().numpy()[()]) + ' on T2 and ' + str(masks_pd[2].sum().cpu().detach().numpy()[()]) + ' on PD.')
        
        fake_t1_c[fake_t1_c<-1] = -1
        fake_t1_c[fake_t1_c>1] = 1
        fake_t2_c[fake_t2_c<-1] = -1
        fake_t2_c[fake_t2_c>1] = 1
        fake_pd_c[fake_pd_c<-1] = -1
        fake_pd_c[fake_pd_c>1] = 1
        
        all_reals = torch.cat((real_t1,real_t2,real_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        all_reals = ((all_reals[...,0] ** 2) + (all_reals[...,1] ** 2)).sum(1).sqrt().detach().cpu().numpy()

#        all_reals = torch.stack(((all_reals[...,0]*model.maps[...,0] + all_reals[...,1]*model.maps[...,1]).sum(1), (all_reals[...,1]*model.maps[...,0] - all_reals[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
#        all_reals = np.hypot(all_reals[...,0],all_reals[...,1])
        
        all_fakes = torch.cat((fake_t1,fake_t2,fake_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        #all_fakes = torch.stack(((all_fakes[...,0]*model.maps[...,0] + all_fakes[...,1]*model.maps[...,1]).sum(1), (all_fakes[...,1]*model.maps[...,0] - all_fakes[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
        all_fakes = ((all_fakes[...,0] ** 2) + (all_fakes[...,1] ** 2)).sum(1).sqrt().detach().cpu().numpy()
        #all_fakes = np.hypot(all_fakes[...,0],all_fakes[...,1])
        
        all_fakes_c = torch.cat((fake_t1_c,fake_t2_c,fake_pd_c),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        all_fakes_c = torch.stack(((all_fakes_c[...,0]*model.maps[...,0] + all_fakes_c[...,1]*model.maps[...,1]).sum(1), (all_fakes_c[...,1]*model.maps[...,0] - all_fakes_c[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
        all_fakes_c = np.hypot(all_fakes_c[...,0],all_fakes_c[...,1])
        
        #all_fakes = model.ref.squeeze().detach().cpu().numpy()

        real_t1 = (real_t1.cpu().detach().numpy()[0,:,:,:])
        fake_t1 = (fake_t1.cpu().detach().numpy()[0,:,:,:])
        fake_t1_c = (fake_t1_c.cpu().detach().numpy()[0,:,:,:])        
       
        real_t2 = (real_t2.cpu().detach().numpy()[0,:,:,:])
        fake_t2 = (fake_t2.cpu().detach().numpy()[0,:,:,:])
        fake_t2_c = (fake_t2_c.cpu().detach().numpy()[0,:,:,:])

        real_pd = (real_pd.cpu().detach().numpy()[0,:,:,:])
        fake_pd = (fake_pd.cpu().detach().numpy()[0,:,:,:])
        fake_pd_c = (fake_pd_c.cpu().detach().numpy()[0,:,:,:])
        abs_real_t1 = np.zeros((5,192,88))
        abs_fake_t1 = np.zeros((5,192,88))
        abs_fake_t1_c = np.zeros((5,192,88))
        abs_real_t2 = np.zeros((5,192,88))
        abs_fake_t2 = np.zeros((5,192,88))
        abs_fake_t2_c = np.zeros((5,192,88))
        abs_real_pd = np.zeros((5,192,88))
        abs_fake_pd = np.zeros((5,192,88))
        abs_fake_pd_c = np.zeros((5,192,88))
        for ind in range(5):
            abs_real_t1[ind,:,:] = np.hypot(real_t1[2*ind,:,:], real_t1[2*ind+1,:,:]).reshape((1,192,88))
            abs_fake_t1[ind,:,:] = np.hypot(fake_t1[2*ind,:,:], fake_t1[2*ind+1,:,:]).reshape((1,192,88))
            abs_fake_t1_c[ind,:,:] = np.hypot(fake_t1_c[2*ind,:,:], fake_t1_c[2*ind+1,:,:]).reshape((1,192,88))
            
            abs_real_t2[ind,:,:] = np.hypot(real_t2[2*ind,:,:], real_t2[2*ind+1,:,:]).reshape((1,192,88))
            abs_fake_t2[ind,:,:] = np.hypot(fake_t2[2*ind,:,:], fake_t2[2*ind+1,:,:]).reshape((1,192,88))
            abs_fake_t2_c[ind,:,:] = np.hypot(fake_t2_c[2*ind,:,:], fake_t2_c[2*ind+1,:,:]).reshape((1,192,88))
            
            abs_real_pd[ind,:,:] = np.hypot(real_pd[2*ind,:,:], real_pd[2*ind+1,:,:]).reshape((1,192,88))
            abs_fake_pd[ind,:,:] = np.hypot(fake_pd[2*ind,:,:], fake_pd[2*ind+1,:,:]).reshape((1,192,88))
            abs_fake_pd_c[ind,:,:] = np.hypot(fake_pd_c[2*ind,:,:], fake_pd_c[2*ind+1,:,:]).reshape((1,192,88))
            
#        fig,ax = plt.subplots(1,3,figsize=(88*3/(24.0),192/(24.0)))
#        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
#        ax[0].imshow(abs_real_t1[3,:,:],cmap='gray')
#        ax[0].axis('off')
#        ax[1].imshow(abs_real_t2[3,:,:],cmap='gray')
#        ax[1].axis('off')
#        ax[2].imshow(abs_real_pd[3,:,:],cmap='gray')
#        ax[2].axis('off')
##        for i in range(3,4):
##            ax[3*i].imshow(abs_real_t1[i,:,:],cmap='gray')
##            ax[3*i].axis('off')
##            ax[3*i+1].imshow(abs_real_t2[i,:,:],cmap='gray')
##            ax[3*i+1].axis('off')
##            ax[3*i+2].imshow(abs_real_pd[i,:,:],cmap='gray')
##            ax[3*i+2].axis('off')
#        plt.show()
#        print('bitti')
        #real_A = util.tensor2im((real_A).data)
        #use = (input('Use or not?') == 1)
            
            
        t1_ref = tensor2im_pgan(all_reals[0:1,:,:])
        t1_c1_ref = tensor2im_pgan(abs_real_t1[0:1])
        t1_c2_ref = tensor2im_pgan(abs_real_t1[1:2])
        t1_c3_ref = tensor2im_pgan(abs_real_t1[2:3])
        t1_c4_ref = tensor2im_pgan(abs_real_t1[3:4])
        t1_c5_ref = tensor2im_pgan(abs_real_t1[4:5])
        
        t2_ref = tensor2im_pgan(all_reals[1:2,:,:])        
        t2_c1_ref = tensor2im_pgan(abs_real_t2[0:1])
        t2_c2_ref = tensor2im_pgan(abs_real_t2[1:2])
        t2_c3_ref = tensor2im_pgan(abs_real_t2[2:3])
        t2_c4_ref = tensor2im_pgan(abs_real_t2[3:4])
        t2_c5_ref = tensor2im_pgan(abs_real_t2[4:5])
        
        pd_ref = tensor2im_pgan(all_reals[2:3,:,:])        
        pd_c1_ref = tensor2im_pgan(abs_real_pd[0:1])
        pd_c2_ref = tensor2im_pgan(abs_real_pd[1:2])
        pd_c3_ref = tensor2im_pgan(abs_real_pd[2:3])
        pd_c4_ref = tensor2im_pgan(abs_real_pd[3:4])
        pd_c5_ref = tensor2im_pgan(abs_real_pd[4:5])
        #use = (input('Use or not?') == 1)
        if opt.use_dt_cns:
            t1_rec = tensor2im_pgan(all_fakes_c[0:1,:,:])
            t1_c1_rec = tensor2im_pgan(abs_fake_t1_c[0:1])
            t1_c2_rec = tensor2im_pgan(abs_fake_t1_c[1:2])
            t1_c3_rec = tensor2im_pgan(abs_fake_t1_c[2:3])
            t1_c4_rec = tensor2im_pgan(abs_fake_t1_c[3:4])
            t1_c5_rec = tensor2im_pgan(abs_fake_t1_c[4:5])
            
            t2_rec = tensor2im_pgan(all_fakes_c[1:2,:,:])
            t2_c1_rec = tensor2im_pgan(abs_fake_t2_c[0:1])
            t2_c2_rec = tensor2im_pgan(abs_fake_t2_c[1:2])
            t2_c3_rec = tensor2im_pgan(abs_fake_t2_c[2:3])
            t2_c4_rec = tensor2im_pgan(abs_fake_t2_c[3:4])
            t2_c5_rec = tensor2im_pgan(abs_fake_t2_c[4:5])
            
            pd_rec = tensor2im_pgan(all_fakes_c[2:3,:,:])
            pd_c1_rec = tensor2im_pgan(abs_fake_pd_c[0:1])
            pd_c2_rec = tensor2im_pgan(abs_fake_pd_c[1:2])
            pd_c3_rec = tensor2im_pgan(abs_fake_pd_c[2:3])
            pd_c4_rec = tensor2im_pgan(abs_fake_pd_c[3:4])
            pd_c5_rec = tensor2im_pgan(abs_fake_pd_c[4:5])
        else:
            t1_rec = tensor2im_pgan(all_fakes[0:1,:,:])
            t1_c1_rec = tensor2im_pgan(abs_fake_t1[0:1])
            t1_c2_rec = tensor2im_pgan(abs_fake_t1[1:2])
            t1_c3_rec = tensor2im_pgan(abs_fake_t1[2:3])
            t1_c4_rec = tensor2im_pgan(abs_fake_t1[3:4])
            t1_c5_rec = tensor2im_pgan(abs_fake_t1[4:5])
            
            t2_rec = tensor2im_pgan(all_fakes[1:2,:,:])
            t2_c1_rec = tensor2im_pgan(abs_fake_t2[0:1])
            t2_c2_rec = tensor2im_pgan(abs_fake_t2[1:2])
            t2_c3_rec = tensor2im_pgan(abs_fake_t2[2:3])
            t2_c4_rec = tensor2im_pgan(abs_fake_t2[3:4])
            t2_c5_rec = tensor2im_pgan(abs_fake_t2[4:5])
            
            pd_rec = tensor2im_pgan(all_fakes[2:3,:,:])
            pd_c1_rec = tensor2im_pgan(abs_fake_pd[0:1])
            pd_c2_rec = tensor2im_pgan(abs_fake_pd[1:2])
            pd_c3_rec = tensor2im_pgan(abs_fake_pd[2:3])
            pd_c4_rec = tensor2im_pgan(abs_fake_pd[3:4])
            pd_c5_rec = tensor2im_pgan(abs_fake_pd[4:5])
            
            
        #x1, x2, y1, y2 = 96, 159, 130, 193
        x1, x2, y1, y2 = 23, 66, 75, 118
        #print(np.mean(np.abs(rec_A[y1:y2+1, x1:x2+1,:]+rec_B[y1:y2+1, x1:x2+1,:]-real_A[y1:y2+1, x1:x2+1,:]-real_B[y1:y2+1, x1:x2+1,:])))
        mask_bin_T1 = np.transpose(np.tile(np.fft.fftshift(masks_t1[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1)), (1, 2, 0))  
        mask_bin_T2 = np.transpose(np.tile(np.fft.fftshift(masks_t2[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1)), (1, 2, 0)) 
        mask_bin_PD = np.transpose(np.tile(np.fft.fftshift(masks_pd[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1)), (1, 2, 0)) 
        #print(analyze_mask_symmetry(mask_bin_B[:,:,0]))
        #print(analyze_mask_symmetry(mask_bin_A[:,:,0]))
        #no_rows = 3
        no_cols = 9
        fig, axes = plt.subplots(nrows=1, ncols=no_cols, figsize=(14,8.5))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
        listoflabels = ['T1 Mask', 'T1 Coil Combination\n From References','T1 Coil Combination\n From Reconstructions', 'T2 Mask', 'T2 Coil Combination\n From References','T2 Coil Combination\n From Reconstructions', 'PD Mask', 'PD Coil Combination\n From References','PD Coil Combination\n From Reconstructions']
        #listoflabels = ['Coil 1 Ref.', 'Coil 1 Rec.', 'Coil 2 Ref.', 'Coil 2 Rec.', 'Coil 3 Ref.', 'Coil 3 Rec.', 'Coil 4 Ref.', 'Coil 4 Rec.', 'Coil 5 Ref.', 'Coil 5 Rec.', 'Coil Combination\n From Ref.', 'Coil Combination\n From Rec.','Mask']
        #listofims = [[t1_c1_ref, t1_c1_rec, t1_c2_ref, t1_c2_rec, t1_c3_ref, t1_c3_rec,  t1_c4_ref, t1_c4_rec, t1_c5_ref, t1_c5_rec, t1_ref, t1_rec, mask_bin_T1], \
         #            [t2_c1_ref, t2_c1_rec, t2_c2_ref, t2_c2_rec, t2_c3_ref, t2_c3_rec,  t2_c4_ref, t2_c4_rec, t2_c5_ref, t2_c5_rec, t2_ref, t2_rec, mask_bin_T2], \
         #            [pd_c1_ref, pd_c1_rec, pd_c2_ref, pd_c2_rec, pd_c3_ref, pd_c3_rec,  pd_c4_ref, pd_c4_rec, pd_c5_ref, pd_c5_rec, pd_ref, pd_rec, mask_bin_PD]]
        listofims = [mask_bin_T1, t1_ref, t1_rec, mask_bin_T2, t2_ref, t2_rec, mask_bin_PD, pd_ref, pd_rec]
#        no_sym_B, no_not_sym_B = analyze_mask_symmetry(mask_bin_B[:,:,0])
 #       no_sym_A, no_not_sym_A = analyze_mask_symmetry(mask_bin_A[:,:,0])
        t1msksm = mask_bin_T1[:,:,0].sum()
        t2msksm = mask_bin_T2[:,:,0].sum()
        pdmsksm = mask_bin_PD[:,:,0].sum()
        for cl_ind in range(no_cols): 
            axes[cl_ind].imshow(listofims[cl_ind])
            axes[cl_ind].axis('off')
            axes[cl_ind].set_title(listoflabels[cl_ind],{'fontsize': 10})
            line = lines.Line2D([-0.125, -0.125], [0, 192],lw=0.25, color='white', axes=axes[cl_ind])
            axes[cl_ind].add_line(line)
            if cl_ind != 0 and cl_ind != 3 and cl_ind != 6:
                transform = blended_transform_factory(fig.transFigure, axes[cl_ind].transAxes)
                axins = zoomed_inset_axes(axes[cl_ind],1.5,bbox_to_anchor=((cl_ind)/float(no_cols), -0.37, 1/float(no_cols), 0.5), bbox_transform=transform, loc=8, borderpad=0) #-0.6
                #x1, x2, y1, y2 = 96, 159, 130, 193 # specify the limits                
                axins_im = np.zeros((192,88,3),dtype=np.uint8)
                axins_im[y1:y2+1, x1:x2+1,:] = np.flipud(listofims[cl_ind])[191-y2:192-y1,x1:x2+1,:]
                
                axins.imshow(axins_im)
                
                axins.set_xlim(x1, x2) # apply the x-limits
                axins.set_ylim(y1, y2) # apply the y-limits
                axins.axis('off')
                mark_inset(axes[cl_ind], axins, loc1=1, loc2=2, fc="none", ec="0.5")
            elif cl_ind == 0:
                t = ("T1 Samplings: %i (%.1f%%)\nTotal samplings: %i\nRealized overall acce.: %.2f") % (t1msksm, 100.0*(t1msksm)/float(t1msksm+t2msksm+pdmsksm), t1msksm+t2msksm+pdmsksm, (192*88*3.0)/float(t1msksm+t2msksm+pdmsksm))
                axes[cl_ind].text(44, 240, t, ha='center', rotation=0, wrap=True, fontsize=8.7)
            elif cl_ind == 3:
                t = ("T2 Samplings: %i (%.1f%%)\nTotal samplings: %i\nRealized overall acce.: %.2f") % (t2msksm, 100.0*(t2msksm)/float(t1msksm+t2msksm+pdmsksm), t1msksm+t2msksm+pdmsksm, (192*88*3.0)/float(t1msksm+t2msksm+pdmsksm))
                axes[cl_ind].text(44, 240, t, ha='center', rotation=0, wrap=True, fontsize=8.7)
            elif cl_ind == 6:
                t = ("PD Samplings: %i (%.1f%%)\nTotal samplings: %i\nRealized overall acce.: %.2f") % (pdmsksm, 100.0*(pdmsksm)/float(t1msksm+t2msksm+pdmsksm), t1msksm+t2msksm+pdmsksm, (192*88*3.0)/float(t1msksm+t2msksm+pdmsksm))
                axes[cl_ind].text(44, 240, t, ha='center', rotation=0, wrap=True, fontsize=8.7)    
#        gs1 = gridspec.GridSpec(2,4)
#        gs1.update(wspace=0, hspace=0.05) # set the spacing between axes. 
#        plt.axis('off')
#        for i in range(8):
#           # i = i + 1 # grid spec indexes from 0
#            ax1 = plt.subplot(gs1[i])
#            ax1.imshow(listofims[i])
#            ax1.axis('off')
#            ax1.set_title(listoflabels[i])
 #       plt.suptitle('Visual Results for JRGAN with Mask Optimization with R = %s' % str(opt.r) + opt.use_mult_const * ' with additional loss')
        plt.savefig(output_dir + '/ReMaFa' + '_hrd' + str(opt.use_hrd_thr) + '.svg', bbox_inches='tight')
        plt.show()
        print('break')
        #break
    
def display_remafa(model, opt, dataset, output_dir):
    print(output_dir)
    print('Remafa will be displayed...')
    datasetind = 90
    #50
    for i, data in enumerate(dataset):
        if i != datasetind:
            continue
        #datasetind += 10
        #print('i235')
        model.set_input(data)
        takens = 0
        while takens != round(opt.ur*192*88*3):
            (real_t1,fake_t1,fake_t1_c,masks_t1),(real_t2,fake_t2,fake_t2_c,masks_t2),(real_pd,fake_pd,fake_pd_c,masks_pd) = model.test()
            takens = masks_t1[2].sum().cpu().detach().numpy()[()] + masks_t2[2].sum().cpu().detach().numpy()[()] + masks_pd[2].sum().cpu().detach().numpy()[()]
            #print(takens)
        print('Number of taken pixels in display_remafa are ' + str(masks_t1[2].sum().cpu().detach().numpy()[()]) + ' on T1 and ' + str(masks_t2[2].sum().cpu().detach().numpy()[()]) + ' on T2 and ' + str(masks_pd[2].sum().cpu().detach().numpy()[()]) + ' on PD.')
        
        fake_t1_c[fake_t1_c<-1] = -1
        fake_t1_c[fake_t1_c>1] = 1
        fake_t2_c[fake_t2_c<-1] = -1
        fake_t2_c[fake_t2_c>1] = 1
        fake_pd_c[fake_pd_c<-1] = -1
        fake_pd_c[fake_pd_c>1] = 1
        
        all_fakes = torch.cat((fake_t1,fake_t2,fake_pd),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        all_fakes = torch.stack(((all_fakes[...,0]*model.maps[...,0] + all_fakes[...,1]*model.maps[...,1]).sum(1), (all_fakes[...,1]*model.maps[...,0] - all_fakes[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
        all_fakes = np.hypot(all_fakes[...,0],all_fakes[...,1])
        
        all_fakes_c = torch.cat((fake_t1_c,fake_t2_c,fake_pd_c),0).reshape(3,5,2,192,88).permute(0,1,3,4,2)
        all_fakes_c = torch.stack(((all_fakes_c[...,0]*model.maps[...,0] + all_fakes_c[...,1]*model.maps[...,1]).sum(1), (all_fakes_c[...,1]*model.maps[...,0] - all_fakes_c[...,0]*model.maps[...,1]).sum(1)),-1).detach().cpu().numpy()
        all_fakes_c = np.hypot(all_fakes_c[...,0],all_fakes_c[...,1])
        
        real_t1 = (real_t1.cpu().detach().numpy()[0,:,:,:])
        fake_t1 = (fake_t1.cpu().detach().numpy()[0,:,:,:])
        fake_t1_c = (fake_t1_c.cpu().detach().numpy()[0,:,:,:])        
       
        real_t2 = (real_t2.cpu().detach().numpy()[0,:,:,:])
        fake_t2 = (fake_t2.cpu().detach().numpy()[0,:,:,:])
        fake_t2_c = (fake_t2_c.cpu().detach().numpy()[0,:,:,:])

        real_pd = (real_pd.cpu().detach().numpy()[0,:,:,:])
        fake_pd = (fake_pd.cpu().detach().numpy()[0,:,:,:])
        fake_pd_c = (fake_pd_c.cpu().detach().numpy()[0,:,:,:])

        ind = 3
        abs_real_t1 = np.hypot(real_t1[2*ind,:,:], real_t1[2*ind+1,:,:]).reshape((1,192,88))
        abs_fake_t1 = np.hypot(fake_t1[2*ind,:,:], fake_t1[2*ind+1,:,:]).reshape((1,192,88))
        abs_fake_t1_c = np.hypot(fake_t1_c[2*ind,:,:], fake_t1_c[2*ind+1,:,:]).reshape((1,192,88))
        
        abs_real_t2 = np.hypot(real_t2[2*ind,:,:], real_t2[2*ind+1,:,:]).reshape((1,192,88))
        abs_fake_t2 = np.hypot(fake_t2[2*ind,:,:], fake_t2[2*ind+1,:,:]).reshape((1,192,88))
        abs_fake_t2_c = np.hypot(fake_t2_c[2*ind,:,:], fake_t2_c[2*ind+1,:,:]).reshape((1,192,88))
        
        abs_real_pd = np.hypot(real_pd[2*ind,:,:], real_pd[2*ind+1,:,:]).reshape((1,192,88))
        abs_fake_pd = np.hypot(fake_pd[2*ind,:,:], fake_pd[2*ind+1,:,:]).reshape((1,192,88))
        abs_fake_pd_c = np.hypot(fake_pd_c[2*ind,:,:], fake_pd_c[2*ind+1,:,:]).reshape((1,192,88))
            
#        fig,ax = plt.subplots(1,3,figsize=(88*3/(24.0),192/(24.0)))
#        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
#        ax[0].imshow(abs_real_t1[3,:,:],cmap='gray')
#        ax[0].axis('off')
#        ax[1].imshow(abs_real_t2[3,:,:],cmap='gray')
#        ax[1].axis('off')
#        ax[2].imshow(abs_real_pd[3,:,:],cmap='gray')
#        ax[2].axis('off')
##        for i in range(3,4):
##            ax[3*i].imshow(abs_real_t1[i,:,:],cmap='gray')
##            ax[3*i].axis('off')
##            ax[3*i+1].imshow(abs_real_t2[i,:,:],cmap='gray')
##            ax[3*i+1].axis('off')
##            ax[3*i+2].imshow(abs_real_pd[i,:,:],cmap='gray')
##            ax[3*i+2].axis('off')
#        plt.show()
#        print('bitti')
        #real_A = util.tensor2im((real_A).data)
        real_t1_im = tensor2im_pgan(abs_real_t1)
        real_t2_im = tensor2im_pgan(abs_real_t2)
        real_pd_im = tensor2im_pgan(abs_real_pd)
        #use = (input('Use or not?') == 1)
        if opt.use_dt_cns:    
            rec_T1_im = tensor2im_pgan(abs_fake_t1_c)
            rec_T2_im = tensor2im_pgan(abs_fake_t2_c)
            rec_PD_im = tensor2im_pgan(abs_fake_pd_c)
        else:
            rec_T1_im = tensor2im_pgan(abs_fake_t1)
            rec_T2_im = tensor2im_pgan(abs_fake_t2)
            rec_PD_im = tensor2im_pgan(abs_fake_pd)
        #x1, x2, y1, y2 = 96, 159, 130, 193
        x1, x2, y1, y2 = 20, 83, 40, 103
        #print(np.mean(np.abs(rec_A[y1:y2+1, x1:x2+1,:]+rec_B[y1:y2+1, x1:x2+1,:]-real_A[y1:y2+1, x1:x2+1,:]-real_B[y1:y2+1, x1:x2+1,:])))
        mask_bin_T1 = np.transpose(np.tile(np.fft.fftshift(masks_t1[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1)), (1, 2, 0))  
        mask_bin_T2 = np.transpose(np.tile(np.fft.fftshift(masks_t2[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1)), (1, 2, 0)) 
        mask_bin_PD = np.transpose(np.tile(np.fft.fftshift(masks_pd[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1)), (1, 2, 0)) 
        #print(analyze_mask_symmetry(mask_bin_B[:,:,0]))
        #print(analyze_mask_symmetry(mask_bin_A[:,:,0]))
        fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(12,2.91))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
        listoflabels = ['T1 Mask', 'T1 Reference', 'T1 Reconstruction','T2 Mask', 'T2 Reference', 'T2 Reconstruction', 'PD Mask', 'PD Reference', 'PD Reconstruction']
        listofims = [mask_bin_T1, real_t1_im, rec_T1_im, mask_bin_T1, real_t2_im, rec_T2_im, mask_bin_PD, real_pd_im, rec_PD_im]
#        no_sym_B, no_not_sym_B = analyze_mask_symmetry(mask_bin_B[:,:,0])
 #       no_sym_A, no_not_sym_A = analyze_mask_symmetry(mask_bin_A[:,:,0])
        t1msksm = mask_bin_T1[:,:,0].sum()
        t2msksm = mask_bin_T2[:,:,0].sum()
        pdmsksm = mask_bin_PD[:,:,0].sum()
        for no,ax in enumerate(axes):
            ax.imshow(listofims[no])
            ax.axis('off')
            ax.set_title(listoflabels[no])
            
            line = lines.Line2D([-0.125, -0.125], [0, 192],lw=0.25, color='white', axes=ax)
            ax.add_line(line)
            
            
            if no != 0 and no != 3 and no != 6:
                transform = blended_transform_factory(fig.transFigure, ax.transAxes)
                axins = zoomed_inset_axes(ax,1.75,bbox_to_anchor=((no)/9.0, -0.6, 1/9.0, 0.5), bbox_transform=transform, loc=8, borderpad=0)
                #x1, x2, y1, y2 = 96, 159, 130, 193 # specify the limits                
                axins_im = np.zeros((192,88,3),dtype=np.uint8)
                axins_im[y1:y2+1, x1:x2+1,:] = np.flipud(listofims[no])[191-y2:192-y1,x1:x2+1,:]
                
                axins.imshow(axins_im)
                
                axins.set_xlim(x1, x2) # apply the x-limits
                axins.set_ylim(y1, y2) # apply the y-limits
                axins.axis('off')
                mark_inset(ax, axins, loc1=1, loc2=2, fc="none", ec="0.5")
            elif no == 0:
                t = ("T1 Samplings: %i (%.1f%%)\nTotal samplings: %i\nRealized overall acce.: %.2f") % (t1msksm, 100.0*(t1msksm)/float(t1msksm+t2msksm+pdmsksm), t1msksm+t2msksm+pdmsksm, (192*88*2.0)/float(t1msksm+t2msksm+pdmsksm))
                ax.text(44, 300, t, ha='center', rotation=0, wrap=True)
            elif no == 3:
                t = ("T2 Samplings: %i (%.1f%%)\nTotal samplings: %i\nRealized overall acce.: %.2f") % (t2msksm, 100.0*(t2msksm)/float(t1msksm+t2msksm+pdmsksm), t1msksm+t2msksm+pdmsksm, (192*88*2.0)/float(t1msksm+t2msksm+pdmsksm))
                ax.text(44, 300, t, ha='center', rotation=0, wrap=True)
            elif no == 6:
                t = ("PD Samplings: %i (%.1f%%)\nTotal samplings: %i\nRealized overall acce.: %.2f") % (pdmsksm, 100.0*(pdmsksm)/float(t1msksm+t2msksm+pdmsksm), t1msksm+t2msksm+pdmsksm, (192*88*2.0)/float(t1msksm+t2msksm+pdmsksm))
                ax.text(44, 300, t, ha='center', rotation=0, wrap=True)    
#        gs1 = gridspec.GridSpec(2,4)
#        gs1.update(wspace=0, hspace=0.05) # set the spacing between axes. 
#        plt.axis('off')
#        for i in range(8):
#           # i = i + 1 # grid spec indexes from 0
#            ax1 = plt.subplot(gs1[i])
#            ax1.imshow(listofims[i])
#            ax1.axis('off')
#            ax1.set_title(listoflabels[i])
 #       plt.suptitle('Visual Results for JRGAN with Mask Optimization with R = %s' % str(opt.r) + opt.use_mult_const * ' with additional loss')
        plt.savefig(output_dir + '/ReMaFa' + '_hrd' + str(opt.use_hrd_thr) + '.svg', bbox_inches='tight')
        plt.show()
        print('break')
        break
def display_remafa2(model, opt, dataset, output_dir):
    print(output_dir)
    print('Remafa will be displayed...')
    nos = 5
    abs_real_t1 = np.zeros((nos,5,192,88))
    abs_fake_t1 = np.zeros((nos,5,192,88))
    abs_fake_t1_c = np.zeros((nos,5,192,88))
    abs_real_t2 = np.zeros((nos,5,192,88))
    abs_fake_t2 = np.zeros((nos,5,192,88))
    abs_fake_t2_c = np.zeros((nos,5,192,88))
    abs_real_pd = np.zeros((nos,5,192,88))
    abs_fake_pd = np.zeros((nos,5,192,88))
    abs_fake_pd_c = np.zeros((nos,5,192,88))
    wh_ind = 0
    for i, data in enumerate(dataset):
        
        #print('i235')
        model.set_input(data)
        (real_t1,fake_t1,fake_t1_c,masks_t1),(real_t2,fake_t2,fake_t2_c,masks_t2),(real_pd,fake_pd,fake_pd_c,masks_pd) = model.test()
        #takens = masks_t1[2].sum().cpu().detach().numpy()[()] + masks_t2[2].sum().cpu().detach().numpy()[()] + masks_pd[2].sum().cpu().detach().numpy()[()]
            #print(takens)
        print('Number of taken pixels in display_remafa are ' + str(masks_t1[2].sum().cpu().detach().numpy()[()]) + ' on T1 and ' + str(masks_t2[2].sum().cpu().detach().numpy()[()]) + ' on T2 and ' + str(masks_pd[2].sum().cpu().detach().numpy()[()]) + ' on PD.')
        
        fake_t1_c[fake_t1_c<-1] = -1
        fake_t1_c[fake_t1_c>1] = 1
        fake_t2_c[fake_t2_c<-1] = -1
        fake_t2_c[fake_t2_c>1] = 1
        fake_pd_c[fake_pd_c<-1] = -1
        fake_pd_c[fake_pd_c>1] = 1
        
        real_t1 = (real_t1.cpu().detach().numpy()[0,:,:,:])
        fake_t1 = (fake_t1.cpu().detach().numpy()[0,:,:,:])
        fake_t1_c = (fake_t1_c.cpu().detach().numpy()[0,:,:,:])        
        
       
        real_t2 = (real_t2.cpu().detach().numpy()[0,:,:,:])
        fake_t2 = (fake_t2.cpu().detach().numpy()[0,:,:,:])
        fake_t2_c = (fake_t2_c.cpu().detach().numpy()[0,:,:,:])
        

        real_pd = (real_pd.cpu().detach().numpy()[0,:,:,:])
        fake_pd = (fake_pd.cpu().detach().numpy()[0,:,:,:])
        fake_pd_c = (fake_pd_c.cpu().detach().numpy()[0,:,:,:])
       
        
        for ind in range(5):
            abs_real_t1[wh_ind,ind,:,:] = np.hypot(real_t1[2*ind,:,:], real_t1[2*ind+1,:,:])
            abs_fake_t1[wh_ind,ind,:,:] = np.hypot(fake_t1[2*ind,:,:], fake_t1[2*ind+1,:,:])
            abs_fake_t1_c[wh_ind,ind,:,:] = np.hypot(fake_t1_c[2*ind,:,:], fake_t1_c[2*ind+1,:,:])
            
            abs_real_t2[wh_ind,ind,:,:] = np.hypot(real_t2[2*ind,:,:], real_t2[2*ind+1,:,:])
            abs_fake_t2[wh_ind,ind,:,:] = np.hypot(fake_t2[2*ind,:,:], fake_t2[2*ind+1,:,:])
            abs_fake_t2_c[wh_ind,ind,:,:] = np.hypot(fake_t2_c[2*ind,:,:], fake_t2_c[2*ind+1,:,:])
            
            abs_real_pd[wh_ind,ind,:,:] = np.hypot(real_pd[2*ind,:,:], real_pd[2*ind+1,:,:])
            abs_fake_pd[wh_ind,ind,:,:] = np.hypot(fake_pd[2*ind,:,:], fake_pd[2*ind+1,:,:])
            abs_fake_pd_c[wh_ind,ind,:,:] = np.hypot(fake_pd_c[2*ind,:,:], fake_pd_c[2*ind+1,:,:])
        
        if wh_ind == 0:
            fig,ax = plt.subplots(nos,15,figsize=(88*15/(60.0),nos*192/(60.0)))
        fig.subplots_adjust(left=0,right=1,bottom=0,top=1,wspace=0,hspace=0)
        for ind in range(5):
            ax[wh_ind,3*ind].imshow(abs_real_t1[wh_ind,ind,:,:],cmap='gray')
            ax[wh_ind,3*ind].axis('off')
            ax[wh_ind,3*ind+1].imshow(abs_real_t2[wh_ind,ind,:,:],cmap='gray')
            ax[wh_ind,3*ind+1].axis('off')
            ax[wh_ind,3*ind+2].imshow(abs_real_pd[wh_ind,ind,:,:],cmap='gray')
            ax[wh_ind,3*ind+2].axis('off')
        
        
        wh_ind += 1
        if wh_ind == nos:
            wh_ind = 0
            plt.show()
            print(i , ' = dtstind')
#def analyze_mask_symmetry(mask):
#    (xs,ys) = np.nonzero(mask)
#    no_sym = 0
#    no_not_sym = 0
#    sym_mask = np.zeros_like(mask)
#    not_sym_mask = np.zeros_like(mask)
#    for (x,y) in zip(xs,ys):
#        if x != 0 and y != 0 and x!=128 and y!=128 and mask[256-x,256-y] != 0:
#            no_sym += 1
#            sym_mask[x,y] = 1
#        else:        
#            no_not_sym += 1
#            if x<=128:
#                not_sym_mask[x,y] = 1
#            else:
#                not_sym_mask[256-x,256-y] = 1
#    return no_sym, no_not_sym, sym_mask, not_sym_mask
def analyze_mask_symmetry(mask, conjs):
    (xs,ys) = np.nonzero(mask)
    no_sym = 0
    no_not_sym = 0
    sym_mask = np.zeros_like(mask)
    not_sym_mask = np.zeros_like(mask)
    print('analyze')
    for (x,y) in zip(xs,ys):
        if (not ((x%128)==0 and (y%128)==0)) and (mask[(256-x)%256,(256-y)%256] != 0):
            no_sym += 1
            sym_mask[x,y] = 1
            #print(x,y)
        else:        
            no_not_sym += 1
            if conjs[x,y] == 1:
                not_sym_mask[x,y] = 1
            else:
                not_sym_mask[(256-x)%256,(256-y)%256] = 1
            #if x== 128 or y==128:
             #   print(x,y)
    return no_sym, no_not_sym, sym_mask, not_sym_mask

def find_thr_for_bin(model, opt, dataset):
    for i, data in enumerate(dataset):
        if i == 0:
            model.set_input(data)
            (real,fake,masks,real_A) = model.test()
            mask = masks[2].cpu().detach().numpy()
            req = mask.size / opt.r
            thr = 0.5
            step = 0.5
            taken = (mask>thr).sum()
            while taken != req:
                taken = (mask>thr).sum()
                isless = taken < req
                ismore = taken > req
                while (isless and taken < req) or (ismore and taken > req):
                    thr += -step * isless + step * ismore
                    taken = (mask>thr).sum()     
                step /= 2
                print('thr :' + str(thr) + ', taken: ' + str(taken) + ', step was made: ' + str(step))
            break
    return thr   
def display_masks(opt, masks):
    if opt.use_fixed_acc:
        mask_prob = np.tile(np.fft.fftshift(masks[0].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1))
        mask_prob = ((np.transpose(mask_prob, (1, 2, 0))))
        mask_rsc = np.tile(np.fft.fftshift(masks[1].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1))
        mask_rsc = ((np.transpose(mask_rsc, (1, 2, 0))))
        mask_bin = np.tile(np.fft.fftshift(masks[2].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1))
        mask_bin = ((np.transpose(mask_bin, (1, 2, 0))))
        f, (ax1,ax2,ax3) = plt.subplots(1, 3, sharey=True, figsize=(6,2.3))
        ax1.set_title('Mask Prob')
        ax1.imshow(mask_prob,cmap='Greys')
        ax2.set_title('Mask Rscl')
        ax2.imshow(mask_rsc, cmap='Greys')
        ax3.set_title('Mask Bin')
        ax3.imshow(mask_bin, cmap='Greys')
        plt.suptitle('RSGAN with Conf.: %s' % opt.run_name)
        plt.show()
    else:
        mask_prob = np.tile(np.fft.fftshift(masks[0].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1))
        mask_prob = ((np.transpose(mask_prob, (1, 2, 0))))
        mask_bin = np.tile(np.fft.fftshift(masks[1].data.squeeze(-1).squeeze(1).cpu().float().numpy()), (3, 1, 1))
        mask_bin = ((np.transpose(mask_bin, (1, 2, 0))))
        f, (ax1,ax2) = plt.subplots(1, 2, sharey=True)
        ax1.set_title('Mask Prob')
        ax1.imshow(mask_prob,cmap='Greys')
        ax2.set_title('Mask Bin')
        ax2.imshow(mask_bin,cmap='Greys')
        plt.suptitle('RSGAN with Conf.: %s' % opt.run_name)
        plt.show()   
def histedges_equalN(x, nbin):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbin + 1),
                     np.arange(npt),
                     np.sort(x))

import os
import numpy as np
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
import Options
import matplotlib
matplotlib.use('Agg') #to block terminal display in deep pc
import matplotlib.pyplot as plt
import time
#import skimage
#print(skimage.__version__)
plt.close('all')
print('Welcome to JRMTHREEGTHREED trainer/tester...')
gpu_id = int(input('Enter gpu_id: '))
trainOrTest = (int(input('Would you like to train(1) or test(0)? '))==1)
#print('train test ' + trainOrTest)
opt = Options.Options('train') if trainOrTest==1 else Options.Options('test')
opt.set_use_midas(False)
opt.gpu_id = gpu_id
opt.lnbl_th_sl = 0
if opt.lnbl_th_sl:
    opt.thrding_slope = input('Enter the thresholding layer''s sigmoid''s initial slope: ')
else:
    opt.thrding_slope=100 #/ math.pow(1.185,49)

        
opt.use_fixed_acc = 1
if opt.use_fixed_acc:
    opt.r=input('Enter desired acceleration rate: ')
    opt.net_name = 'JRMTHREEGTHREED_l10000_R'+str(opt.r)
    if trainOrTest:
        opt.ur = 1.0/float(opt.r)
    else:
        opt.ur = 1.0/float(opt.r)
    #opt.ur= find_ur_for_comp(1.0/r, opt.thrding_slope) #undersampling rate
    print('US rate for compensation was made ' + str(opt.ur))
else:
    opt.acc_loss_coe = input('Enter the coefficient of the loss component about acceleration: ')
    opt.net_name = 'JRMTHREEGTHREED/acc'+str(opt.acc_loss_coe)

opt.lrm=0.0002
opt.lrg=0.0002
opt.use_hrd_thr = False
opt.use_cal_reg = 0
opt.lnbl_pr_sl = 0
if opt.lnbl_pr_sl:
    opt.pr_mask_slope = input('Enter the probmask layer''s sigmoid''s initial slope: ')
else:
    opt.pr_mask_slope = 100
opt.pr_init = 5
opt.prob_mask_out=None
opt.opti_tech = 1
if opt.use_fixed_acc:
    opt.set_dir(opt.net_name,run_name='r'+str(opt.r)+'_lrm'+str(opt.lrm)[2:]+'_lrg'+str(opt.lrg)[2:]+'_prsl'+('1_'*opt.lnbl_pr_sl)+str(opt.pr_mask_slope)+'_thrsl'+('1_'*opt.lnbl_th_sl)+str(opt.thrding_slope)+'_init'+str(opt.pr_init)+'_opt'+str(opt.opti_tech))
else:
    opt.set_dir(opt.net_name,run_name='alc'+str(opt.acc_loss_coe)[2:]+'_lrm'+str(opt.lrm)[2:]+'_lrg'+str(opt.lrg)[2:]+'_prsl'+('1_'*opt.lnbl_pr_sl)+str(opt.pr_mask_slope)+'_thrsl'+('1_'*opt.lnbl_th_sl)+str(opt.thrding_slope)+'_init'+str(opt.pr_init)+'_opt'+str(opt.opti_tech)  )
if opt.use_cal_reg:
    opt.cal_reg_rad=input('Enter filled region radius: ')
    opt.run_name = opt.run_name + '_calrad' + str(opt.cal_reg_rad)
else:
    opt.cal_reg_rad = 0
opt.use_mult_const = (input('Do you want to utilize a loss component to constrain the parameter mult (1) or not(0)?')==1)
opt.mult_loss_coe = 0
if opt.use_mult_const:
    opt.mult_loss_coe = 10000
    opt.run_name = opt.run_name + '_multloss'+str(opt.mult_loss_coe)
opt.load_rcn_net = False
opt.fftnorm = True #default is one
opt.ifftnorm = False #default is zero
run_no = '10000_500_5'# 'notnrm' #_newdtst'#'cnj_ind_masks'#'vgg3000' # '452likmeans' #'56likmeans' #'wholemeanli_b5'#
opt.loss_tech =3
opt.use_joint_us = True
opt.use_dt_cns = (input('Do you want to use data consistency?')==1)
if opt.use_dt_cns:
    if trainOrTest:
        opt.epfordtcns = (input('Epoch no for data consistency: '))
    opt.use_mult_const = False
    opt.mult_loss_coe = 0
    print('Mult loss coeff was made zero.')
    opt.ur = 1.0/float(opt.r)
if opt.loss_tech == 1:
    opt.run_name = opt.run_name + opt.load_rcn_net * '_ldrcn' + opt.fftnorm * '_fn' + opt.ifftnorm * 'ifn' + '_sq' + '_runno'+str(run_no)+(opt.use_joint_us*'_jusr')
elif opt.loss_tech == 2:
    opt.run_name = opt.run_name + opt.load_rcn_net * '_ldrcn' + opt.fftnorm * '_fn' + opt.ifftnorm * 'ifn' + '_sftmx' + '_runno'+str(run_no)+(opt.use_joint_us*'_jusr')
elif opt.loss_tech == 3:
    opt.run_name = opt.run_name + opt.load_rcn_net * '_ldrcn' + opt.fftnorm * '_fn' + opt.ifftnorm * 'ifn' + '_nrml' + '_runno'+str(run_no)+(opt.use_joint_us*'_jusr')
opt.run_name_wocns = opt.run_name
opt.run_name += (opt.use_dt_cns*'_dtcns')
#opt.run_name = opt.run_name + '_fr5050500e75'
opt.epoch_count=1
#if trainOrTest == 0:
#    opt.which_epoch = input('Which epoch do you want to load for testing? ')
#    opt.epoch_count=opt.which_epoch + 1
if trainOrTest == 1:
    opt.continue_train = False
    if opt.continue_train:
        opt.which_epoch = input('From which epoch do you want to continue training? ')
        opt.epoch_count=opt.which_epoch + 1  
 
print('Net name is ' + opt.net_name)      
print('Run name is ' + opt.run_name)
#print('train test ' + trainOrTest + 'bool' + str(trainOrTest == 1))
if trainOrTest == 1:
    train(opt)
else:
    test(opt)
#plt.close('all')
