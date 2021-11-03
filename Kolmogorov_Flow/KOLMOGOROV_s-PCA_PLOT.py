# -*- coding: utf-8 -*-
"""
Created on Mon JUL 13 08:48:56 2020

@author: iagkneer

###############################################################################
            NEURAL NETWORL TO GENERATE POD FROM DATA
###############################################################################

#SUPERIOR:            None
#INFERIOR:            None
#VERSION:             1.0
#
#BASIC IMPLEMENTATION: -
#AUTHOR:               Simon Kneer
#
#RECENT CHANGES:
#DESCRIPTION:
# This is a first implementation to generate POD from data using a neural net
#INPUT:    /data
#          
#
#OUTPUT:   /out


###############################################################################
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as ks
from matplotlib import pyplot as plt
import pickle as pk

#Number of modes
n_modes=2
#Number of modes to be displayed in comparison
max_modes=30

tf.config.set_visible_devices([], 'GPU')

#Load flow-field data
#------------------------
fnstr="../DATA/Kolmogorov_Re35_N16_T30000_32x32.pickle"
# Pickle load

with open(fnstr, 'rb') as f:
    X = pk.load(f) #Data is ordered (t,Y,X)

X = np.expand_dims(X, axis=3)

#Fourier coordinates
k1 = np.linspace(0,15,16)
k2 = np.linspace(-16,-1,16)
k = np.concatenate((k1, k2),axis=0)
kk, dump = np.meshgrid(k,k)
imag = tf.complex(0.0,1.0)


def transformX(INS):
    X,S = INS
    X = tf.squeeze(X,axis=3)

    X_C = tf.cast(X, tf.complex64)
    S_C = tf.cast(S, tf.complex64)
    S_C = tf.expand_dims(S_C,axis = 2)

    X_F = tf.signal.fft(X_C)
    s = tf.math.exp(-imag*2*np.pi/32*kk*S_C)

    X_F = X_F * s

    X_C = tf.signal.ifft(X_F)

    X_C = tf.expand_dims(X_C,axis=3)
    return tf.cast(X_C, tf.float32)

#------------------------
#MODEL
#------------------------
input_img = ks.Input(shape=(X.shape[1], X.shape[2],1))
#STN
STN = tf.keras.Sequential()
STN.add(ks.Reshape([32*32*1]))
STN.add(ks.Dense(2,activation=None,use_bias=False,name='STN_LIN'))
COS_SIN = STN(input_img)
COS = ks.Lambda(lambda x: x[:,0:1])(COS_SIN)
SIN = ks.Lambda(lambda x: x[:,1:2])(COS_SIN)
SX = tf.math.atan2(COS,SIN)/(2*np.pi)*32


SHIFTED = ks.Lambda(lambda x: transformX(x))([input_img,SX])
TRANSFORMER = tf.keras.models.Model(input_img,SHIFTED)

## Encoder
ENC = tf.keras.Sequential()

ENC.add(ks.Reshape([32*32*1]))
ENC.add(ks.Dense(n_modes,activation=None,use_bias=False))

## Decoder
latent_inputs = ks.Input(shape=(n_modes))
DEC = tf.keras.Sequential()

DEC.add(ks.Dense(32*32*1,activation=None,use_bias=False))
DEC.add(ks.Reshape([32,32,1]))

#Shift input
SHIFT = TRANSFORMER(input_img)

#Discrete transforms
INPUTS = []
for j in range(2):
    A = tf.image.rot90(SHIFT, k=2*j)
    for i in range(8):
        if((i+1)%2==0):
            A = tf.reverse(A, [2])
        INPUTS.append((-1)**i*tf.roll(A, shift=i*4, axis=1))

ET = ks.Concatenate(axis=3)(INPUTS)

#Feed through Autoencoder
PASS_TROUGH = []
for i in range(16):
    PASS_TROUGH.append(DEC(ENC(INPUTS[i])))

E = ks.Concatenate(axis=3)(PASS_TROUGH)
#Note that unshifting the sample is not necessary for training
autoencoder = tf.keras.models.Model(input_img, [E, ET])

loss = tf.math.square(E-ET)
loss = tf.math.reduce_mean(loss,axis=(1,2,))
loss = tf.math.reduce_min(loss,axis=1)
loss = tf.math.reduce_mean(loss)

Q, QT = autoencoder(input_img)
LSS = tf.math.square(Q-QT)
LSS = tf.math.reduce_mean(LSS,axis=(1,2,))
#Get index of best fit siamese branch
IND = tf.math.argmin(LSS,axis=1)

OUT = tf.gather(Q, IND, axis=3,batch_dims=1)
OUTT = tf.gather(QT, IND, axis=3,batch_dims=1)

INDICER = tf.keras.models.Model(input_img, IND)
PRINTER = tf.keras.models.Model(input_img, [OUT, OUTT])
autoencoder.add_loss(loss)
PRINTER.add_loss(loss)


opt = tf.keras.optimizers.Adam(learning_rate=1)

autoencoder.compile(optimizer=opt)
autoencoder.load_weights('models/s-PCA_{}MODES'.format(n_modes))

OUTP, INP = PRINTER.predict(X[:,:,:,:], batch_size=50, verbose=1, steps=None, callbacks=None)
print('Reconstruction error with {} modes'.format(n_modes),np.mean((OUTP-INP)**2)/np.mean((INP)**2))


S2 = STN.predict(X[:,:,:,:], batch_size=50, verbose=1, steps=None, callbacks=None)
BRANCH  = INDICER.predict(X[:,:,:,:], batch_size=50, verbose=1, steps=None, callbacks=None)

indstr = 'INDICES_CONTINUOUS.pk'
S = np.arctan2(S2[:,0],S2[:,1])/(2*np.pi)*32

#Load shift modes
weights = STN.get_layer("STN_LIN").get_weights()[0]

weights = weights.reshape([32,32,2])
#Plot shift modes
plt.figure()
plt.imshow(weights[:,:,0],extent=(0, 2*np.pi, 0, 2*np.pi))
plt.xlabel('$x$')
plt.ylabel('$y$')
ax = plt.gca()
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_yticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.figure()
plt.imshow(weights[:,:,1],extent=(0, 2*np.pi, 0, 2*np.pi))
plt.xlabel('$x$')
plt.ylabel('$y$')
ax = plt.gca()
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_yticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

NT = np.linspace(0,0.1*5000,5000)
#Plot shift
plt.figure()
plt.plot(NT,S[142000:147000]/16*np.pi,c='red',label="${STN}$")

plt.legend(loc='upper right')
plt.xlabel('$t$')
plt.ylabel("$s$")
plt.ylim(-np.pi,np.pi)
ax = plt.gca()
ax.set_yticks([-np.pi, -np.pi/2,0, np.pi/2, np.pi])
ax.set_yticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$','$0$', r'$\frac{\pi}{2}$', r'$\pi$'])

plt.xlim(0,500)

#Plot branches
counts = np.bincount(BRANCH)
fig, ax = plt.subplots(figsize=[12.8, 4.8])
ax.bar(range(16), counts, width=1, align='center',edgecolor = "black")
my_xticks = ['${P}_0{S}_0$','${P}_0{S}_1$','${P}_0{S}_2$','${P}_0{S}_3$','${P}_0{S}_4$','${P}_0{S}_5$','${P}_0{S}_6$','${P}_0{S}_7$','${P}_1{S}_0$','${P}_1{S}_1$','${P}_1{S}_2$','${P}_1{S}_3$','${P}_1{S}_4$','${P}_1{S}_5$','${P}_1{S}_6$','${P}_1{S}_7$']
ax.set_xticks(np.arange(16))
ax.set_xticklabels(my_xticks)

ax.set(xlim=[-1, 16])

plt.xlabel('$l$')
ax.xaxis.label.set_color('white')
plt.ylabel('$N$')
plt.yscale('log')
plt.ylim(1,5E5)


#Compute s-PCA modes
X = INP.reshape((INP.shape[0], int(INP.shape[1]*INP.shape[2]*INP.shape[3])))

#Covariance matrix
m=X.shape[1]
R=np.matmul(X.T,X)/m

#Eigenvalues and vectors
w, v = scipy.linalg.eigh(R,subset_by_index = [R.shape[0]-max_modes, R.shape[0]-1] )
w = w[::-1]
v = v[:,::-1]

#Recreate spatial eigenfunctions from snapshot ones
modes=v

#Coefficients
amplitudes=np.matmul(modes.T,X.T)

modes = modes[:,:n_modes]
calc_amp = amplitudes[:n_modes,:]

reconstructed=np.matmul(calc_amp.T,modes.T)

modes = modes.reshape(D.shape[1],D.shape[2],n_modes)

for i in range(n_modes):
    plt.figure("MODE {}".format(i+1))
    plt.imshow(modes[:,:,i],extent=(0, 2*np.pi, 0, 2*np.pi))
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    ax = plt.gca()
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])




#Plot Normalized Modal Energies
w = w/np.sum(w)
N_modes = np.linspace(1,max_modes,max_modes)
plt.figure('Energies')
plt.scatter(N_modes,w[:max_modes])
plt.ylabel('$e_i$')
plt.xlabel('$i$')
plt.yscale('log')
plt.xlim([1,max_modes])
plt.ylim([1E-4,1])


plt.show()

