# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021

@author: svk20

###############################################################################
          NEURAL NETWORK TO GENERATE s-nlPCA FOR THE BURGERS' DATA
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
# This is an implementation to generate symmetry aware nonlinear PCA (s-nlPCA) from data using a neural net
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
learning_rate=0.001
#Set batch size
batch_size=500
#Set number of epochs
epochs=15000
#Set patience for early stopping
patience=5000
#Set min delta for early stopping
min_delta = 1E-9
#Set steps for exponantial learning rate decay
decay_steps=100000
#Set decay rate for exponantial learning rate decay
decay_rate=0.1


#Set number of modes
n_modes=1

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
ENC.add(ks.Dense(512,activation=custom_activation,name='ENC_1'))
ENC.add(ks.Dense(n_modes, activation=None,use_bias=True,name='ENC_2'))

## Decoder
latent_inputs = ks.Input(shape=(n_modes))
DEC = tf.keras.Sequential()
DEC.add(ks.Dense(512,activation=custom_activation,name='DEC_1'))
DEC.add(ks.Dense(X.shape[1] , activation=None,use_bias=True,name='DEC_2'))

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
autoencoder.save_weights('models/s-nlPCA_{}MODES'.format(n_modes))

#Model for extracting latent variables
latents = ENC(SHIFT)
INTERS = tf.keras.models.Model(input_img,latents)


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
