# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021

@author: svk20

###############################################################################
          NEURAL NETWORK TO GENERATE s-PCA FOR THE BURGERS' DATA
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
# This is an implementation to generate symmetry aware PCA (s-PCA) from data using a neural net and a following eigenvalue ansatz
#INPUT:    /data
#          
#
#OUTPUT:   /models


###############################################################################
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as ks
from matplotlib import pyplot as plt
import pickle as pk

#Set learning rate
learning_rate=0.01
#Set batch size
batch_size=500
#Set number of epochs
epochs=2000
#Set patience for early stopping
patience=500
#Set min delta for early stopping
min_delta = 1E-9
#Set steps for exponantial learning rate decay
decay_steps=10000
#Set decay rate for exponantial learning rate decay
decay_rate=0.1


#Set number of modes for training
n_modes_train=1
#Set number of modes for PCA on transformed data
n_modes=4
#Set Modes for which to show energies
n_modes_eneg = 30

#load flow-field data
#------------------------
fnstr="DATA/BURGERS_1D.pickle"
with open(fnstr, 'rb') as f:
    X = pk.load(f) #Data is ordered (t,Y,X)

#FOURIER COORDINATES
k1 = np.linspace(0,31,32)
k2 = np.linspace(-32,-1,32)
k = np.concatenate((k1, k2),axis=0)
imag = tf.complex(0.0,1.0)


#Transformation function for sample using FFT
def transformX(INS):
    X,S = INS

    X_C = tf.cast(X, tf.complex64)
    S_C = tf.cast(S, tf.complex64)

    X_F = tf.signal.fft(X_C)

    s = tf.math.exp(-imag*2*np.pi/32*k*S_C)
    X_F = X_F * s

    X_C = tf.signal.ifft(X_F)


    return tf.cast(X_C, tf.float32)

#Smooth STN weights after each epoch
class smoothingcallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        weights = STN.get_layer("STN_LIN").get_weights()[0]
        weigts_p = tf.roll( weights, 1, axis=0)
        weigts_m = tf.roll( weights, -1, axis=0)
        weights = (2*weights+weigts_p+weigts_m)/4
        w = [weights]
        STN.get_layer("STN_LIN").set_weights(w)

#Set activation function
def custom_activation(x):
    return tf.nn.swish(x)


input_img = ks.Input(shape=(X.shape[1]))

#STN
STN = tf.keras.Sequential()
STN.add(ks.Dense(2,activation=None,use_bias=False,name='STN_LIN'))
COS_SIN = STN(input_img)
COS = ks.Lambda(lambda x: x[:,0:1])(COS_SIN)
SIN = ks.Lambda(lambda x: x[:,1:2])(COS_SIN)
SX = tf.math.atan2(COS,SIN)/(2*np.pi)*32
SHIFTEDX = ks.Lambda(lambda x: transformX(x))([input_img,SX])
TRANSFORMER = tf.keras.models.Model(input_img,SHIFTEDX)

## Encoder
ENC = tf.keras.Sequential()
ENC.add(ks.Dense(n_modes_train, activation=None,use_bias=False,name='ENC'))

## Decoder
latent_inputs = ks.Input(shape=(n_modes_train))
DEC = tf.keras.Sequential()
DEC.add(ks.Dense(X.shape[1] , activation=None,use_bias=False,name='DEC'))

#Shifted input
SHIFT = TRANSFORMER(input_img)
#Shifted output
SHIFTED_OUTPUT = DEC(ENC(SHIFT))
#Unshifted output
output_img = ks.Lambda(lambda x: transformX(x))([SHIFTED_OUTPUT,-1*SX])
#Full network
autoencoder = tf.keras.models.Model(input_img,output_img)

#Loss
loss = tf.math.square(output_img-input_img)
loss = tf.math.reduce_mean(loss)
autoencoder.add_loss(loss)

#Learning rate scheduler
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
learning_rate,
decay_steps=decay_steps,
decay_rate=decay_rate,
staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

#Early stopping
earl_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience,  restore_best_weights=True,min_delta=min_delta)
#Compile full network
autoencoder.compile(optimizer=opt)

#Train network
history=autoencoder.fit(X,X,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[earl_stop,smoothingcallback()],
                        verbose=1
                        )


#save weights
autoencoder.save_weights('models/s-PCA_{}MODES'.format(n_modes_train))

#Model for extracting latent variables
latents = ENC(SHIFT)
INTERS = tf.keras.models.Model(input_img,latents)

#Model for extracting shifted states
SHIFTED_STATES = tf.keras.models.Model(input_img,SHIFT)

X_SHIFT = SHIFTED_STATES.predict(X, batch_size=batch_size, verbose=1, steps=None, callbacks=None)
#Do PCA on shifted samples
R=np.matmul(X_SHIFT.T,X_SHIFT)

#Eigenvalues and vectors
w, modes = np.linalg.eig(R)

#Recreate spatial eigenfunctions from snapshot ones

#Coefficients
amplitudes=np.matmul(modes.T,X_SHIFT.T)

modes = modes[:,:n_modes]
calc_amp = amplitudes[:n_modes,:]

reconstructed=np.matmul(calc_amp.T,modes.T)


#Plot modes
x_coords = np.linspace(0,2*np.pi,64)
plt.figure("Modes",figsize=(12, 6))
colors = ['black','red','blue','green','gold','lime','hotpink']
marks = ['v','^','<','>']
for i in range(n_modes):
    plt.plot(x_coords,modes[:,i],c=colors[i],marker=marks[i],markevery=5,label='$i={}$'.format(i+1))
plt.xlabel('$x$')
plt.xlim([0,2*np.pi])
plt.ylim([-0.25,0.25])

ax = plt.gca()
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])
ax.legend(loc="upper right")
plt.ylabel('$\phi_i$')

#Plot Normalized Modal Energies
w = w/np.sum(w)
N_modes = np.linspace(1,n_modes_eneg,n_modes_eneg)
plt.figure('Energies')
plt.scatter(N_modes,w[:n_modes_eneg])
plt.ylabel('$e_i$')
plt.xlabel('$i$')
plt.yscale('log')
plt.xlim([1,n_modes_eneg])
plt.ylim([1E-16,1])


#Make reconstruction
P = autoencoder.predict(X, batch_size=batch_size, verbose=1, steps=None, callbacks=None)

#Print reconstruction error
print('Reconstruction error with {} modes'.format(n_modes),np.mean((P-X)**2)/np.mean((X)**2))

#Get STN weigths and plot them
weights = STN.get_layer("STN_LIN").get_weights()[0]
NS = np.linspace(0,2*np.pi,64)
SIN = np.cos(NS)
COS = np.sin(NS)
plt.figure()
plt.plot(NS,weights[:,0]/np.amax(weights[:,0]),c='red',label="$\mathbf{W}_{s_1}$")
plt.plot(NS,weights[:,1]/np.amax(weights[:,0]),c='blue',label="$\mathbf{W}_{s_2}$")
plt.plot(NS,SIN,'--',c='red',label="$\cos(x)$")
plt.plot(NS,COS,'--',c='blue',label="$\sin(x)$")
plt.xlabel('$x$')
plt.xlim(0,2*np.pi)
ax = plt.gca()
ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
ax.set_xticklabels(['$0$', r'$\frac{\pi}{2}$', r'$\pi$', r'$\frac{3\pi}{2}$', r'$2\pi$'])

plt.legend(loc='upper right')



plt.show()
