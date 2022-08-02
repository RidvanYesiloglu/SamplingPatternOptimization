class Options:
    def __init__(self,opt_set):
        self.inp_sh=(1,3,1,192,88,2)
        if opt_set == 'test':
            self.niter=125
            self.niter_decay=125
            self.save_epoch_freq=25
            self.update_html_freq=2275;
            self.display_freq = 250
            self.print_freq=455
            self.no_html=False
            self.display_single_pane_ncols = 0
            self.save_latest_freq=1000000000            
            self.phase='test'
            self.beta1=0.5
            self.no_lsgan=True
            self.lambda_A=200
            self.lambda_identity=0
            self.pool_size=0
            self.lr_policy='lambda'
            self.lr_decay_iters=50
            self.isTrain=False
            self.lambda_vgg=100
            self.vgg_layer=2
            self.lambda_adv=20
            self.dataroot='/auto/data2/ridvan/DATASETS/umram_dataset/numpy_data/'
            self.batchSize=1
            self.input_nc=30
            self.output_nc=10
            self.ngf=64
            self.ndf=64
            self.n_layers_D=3
            self.dataset_mode='aligned_mat'
            self.model='pGAN'
            self.which_direction='BtoA'
            self.nThreads=2
            
            self.norm='instance'
            self.serial_batches=True
            self.training=True
            self.display_winsize=256
            self.display_id=0
            self.display_server="http://localhost"
            self.display_port=8097
            self.no_dropout=True
            self.max_dataset_size=float("inf")
            self.init_type='normal'
        elif opt_set == 'train': 
            self.niter=175
            self.niter_decay=75
            self.save_epoch_freq=25
            self.update_html_freq=2275;
            self.display_freq = 250
            self.print_freq=455
            self.no_html=False
            self.display_single_pane_ncols = 0
            self.save_latest_freq=1000000000            
            self.phase='train'
            self.beta1=0.5
            self.no_lsgan=True
            self.lambda_A=10000#200
            self.lambda_identity=0
            self.pool_size=0
            self.lr_policy='lambda'
            self.lr_decay_iters=50
            self.isTrain=True
            self.lambda_vgg=500#100
            self.vgg_layer=2
            self.lambda_adv=5#20
            self.dataroot='/auto/data2/ridvan/DATASETS/umram_dataset/numpy_data/'
            self.batchSize=1
            self.input_nc=30
            self.output_nc=10
            self.ngf=64
            self.ndf=64
            self.n_layers_D=3
            self.dataset_mode='aligned_mat'
            self.model='pGAN'
            self.which_direction='BtoA'
            self.nThreads=2
            
            self.norm='instance'
            self.serial_batches=False
            self.training=True
            self.display_winsize=256
            self.display_id=0
            self.display_server="http://localhost"
            self.display_port=8097
            self.no_dropout=True
            self.max_dataset_size=float("inf")
            self.init_type='normal'
        elif opt_set == 'val':
            print('entered val opt creation')
            self.phase='val'
            self.dataroot='/auto/data2/ridvan/DATASETS/umram_dataset/numpy_data/'
            self.which_direction='BtoA'
            self.dataset_mode='aligned_mat'
            self.batchSize=1
            self.serial_batches=False
            self.nThreads=2
            self.max_dataset_size=float("inf")
            self.model='pGAN'
            self.lr_gd=0.0005
            self.lr_m= 0.001
    def set_dir(self, net_name, run_name):
        self.net_name=net_name
        self.run_name=run_name
        self.checkpoints_dir='/auto/data2/ridvan/MSK_OPT_PRJ/'+net_name+'/'
    def set_use_midas(self, use_midas):
        self.use_midas = use_midas
        if use_midas:
            self.dataroot='/auto/data2/ridvan/DATASETS/MIDAS'
            self.inp_sh=(1,1,176,256,2)