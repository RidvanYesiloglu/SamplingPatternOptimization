import h5py
import numpy as np
import matplotlib.pyplot as plt
target_file='datasets/IXI/train/data.mat'
f = h5py.File(target_file,'r') 
data_y=np.array(f['data_y'])[:,:,:,1]
data_x=np.array(f['data_x'])[:,:,:,1]

data_x_t = np.transpose(data_x,(2,0,1))
print(np.fft.fftshift(np.fft.fft2(data_x_t)).shape)
plt.imshow(np.abs(np.fft.fftshift(np.fft.fft2(data_x_t)))[0,:,:])

data_x_fu=mask*np.fft.fftshift(np.fft.fft2(data_x_t))
print(data_x_fu.shape)
data_x_u = np.real(np.fft.ifft2(np.fft.ifftshift(data_x_fu)))
print(data_x_u.shape)

data_u = np.transpose(data_x_u,(1,2,0))

fig = plt.figure()
axes = fig.subplots(nrows=1, ncols=2)
axes[0].imshow(data_x_u[0,:,:])
axes[1].imshow(data_x_t[0,:,:])
plt.show()




import h5py
import numpy as np
import matplotlib.pyplot as plt
target_file='datasets/IXI/train/data.mat'
f = h5py.File(target_file,'r') 
data_y=np.array(f['data_y'])[:,:,:,1]
data_x=np.array(f['data_x'])[:,:,:,1]
data_x_t = np.transpose(data_x,(2,0,1))
data_x_fu=mask*np.fft.fftshift(np.fft.fft2(data_x_t))
data_x_u = np.real(np.fft.ifft2(np.fft.ifftshift(data_x_fu)))
data_u = np.transpose(data_x_u,(1,2,0))
