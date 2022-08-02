import torch.utils.data
from data.base_data_loader import BaseDataLoader
import numpy as np, h5py
import random
def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreateDataset(opt):
    #data directory
    target_file=opt.dataroot+opt.phase+'/data.npy'
    print('phase: ', opt.phase, ', dataroot: ', opt.dataroot)
    data = np.load(target_file) # with shape (3,5,1085,192,88,2) 
    #data=np.transpose(data,(0,1,3,2,4,5))
    target_file=opt.dataroot+opt.phase+'/mps.npy'    
    mps = np.load(target_file)
    #mps = np.transpose(mps ,(0,1,3,2,4,5))
    target_file=opt.dataroot+opt.phase+'/sng_ref.npy'    
    ref= np.load(target_file)
    #ref = np.transpose(np.load(ref) ,(0,2,1,3,4))
    print('loaded data shape: ', data.shape, ', map shape: ', mps.shape, ', ref shape: ', ref.shape)
    dataset = []
    #making range of each image -1 to 1 and converting to torch tensor
    for train_sample in range(data.shape[2]):
        # each sample has shape (3,5,192,88,2)
        dataset.append({'A': torch.from_numpy(data[:,:,train_sample,:,:,:]),'mps':torch.from_numpy(mps[:,:,train_sample,:,:,:]),'ref':torch.from_numpy(ref[:,train_sample,:,:]),'A_paths':opt.dataroot}) 
    return dataset 



class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self


    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:
                break
            yield data