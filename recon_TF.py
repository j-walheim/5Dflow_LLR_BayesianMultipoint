import sys
import os
os.environ['TOOLBOX_PATH'] = '/usr/local/bin/bart/'
sys.path.append('/usr/local/bin/bart/python/')

import numpy as np
import h5py
import scipy.io as sio
from functionsMulti_TF import *
import bart

#define forward and backward mappging between image space and k-space
def k2i(K):
    X = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(K.copy(),axes=(0,1,2)),axes=(0,1,2),norm="ortho"),axes=(0,1,2))
    return X
def i2k(X):
    K = np.fft.ifftshift(np.fft.fftn(np.fft.fftshift(X.copy(),axes=(0,1,2)),axes=(0,1,2),norm="ortho"),axes=(0,1,2))
    return K

#load kspace data and sensitivity maps

f = h5py.File(os.path.join(sys.path[0], "data.h5"),'r')

#f = h5py.File('data.h5','r')
K = np.complex64(f["K"])
sMaps = np.complex64(f["sMaps"])
f.close()

print('Sucessfuly loaded data, start recon')

#combine coil data according to Roemer and take max intensity of image to scale regularization weight accordingly
sos = np.sqrt(np.sum(np.abs(sMaps[:,:,:,:,None,None,None])**2,3))+1e-6
rcomb = np.sum(k2i(K)*np.conj(sMaps[:,:,:,:,None,None,None]),3)/sos
regFactor = np.max(np.abs(rcomb.flatten()))
print('scaling Factor for recon: ', regFactor)

#set up Recon
regWeight = 0.012 #lambda in Cost function
blk = 16 #block size for locally low rank recon
bart_string = 'pics -u1 -w 1 -H -d5 -e -i 80 -R L:7:7:%.2e -b %d' % (regFactor*regWeight,blk) #define Bart strings


#loop through velocity encodings and perform LLR recon
szIm = np.shape(K[:,:,:,0,:,:,:]) #image dimensions
I = np.zeros(szIm,dtype=np.complex64)
for i in range(np.size(K, 5)):
    # perform recom with BART; BART expects other dimensions, therefore we add singleton dimensions to shift dynamics to position
    # where BART does not interpret them in a specific way
    tmp = bart.bart(1,bart_string,K[:,:,:,:,None,None,None,None,:,i,:],sMaps)
    print('I.shape', I.shape, 'tmp.shape ', tmp.shape)
    I[:,:,:,:,i,:] = np.squeeze(tmp)

#store recon results
sio.savemat(os.path.join(sys.path[0], "img_tf.mat"),{'I':I})

I = I[:,:,:,:,:,0] #take expiratory data only

venc = np.array([0.5, 1.5])

kv = np.array([0, np.pi / venc[0], np.pi / venc[1]])
kv = np.tile(kv[:, None], (1, 3))

#start Bayes unfololding, then save mean velociteis results_v and intravoxel standard deviations results_std
print('perform multipoint recon for image with dims (x,y,z,hp,venc): ',I.shape)
results_v,results_std = BayesUnfold(I,kv)
sio.savemat(os.path.join(sys.path[0],'bayes_res_TF.mat'), {'results_v': results_v,'I': I,'results_std':results_std})
