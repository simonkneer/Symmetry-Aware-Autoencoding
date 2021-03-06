# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021
@author: svk20
###############################################################################
          NEURAL NETWORK TO GENERATE s-PCA FOR THE KOLMOGOROV FLOW DATA
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
# This is an implementation to generate symmetry-aware PCA (s-PCA) from data using a neural net
#INPUT:    ../DATA
#          
#
#OUTPUT:   /models
###############################################################################
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers as ks
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import pickle as pk

#Set learning rate
learning_rate=0.01
#Set batch size
batch_size=500
#Set number of epochs
epochs=15
#Set patience for early stopping
patience=5
#Set number of modes
n_modes=2
#Set min delta for early stopping
min_delta = 1E-5


#Load flow-field data
#------------------------
fnstr="../DATA/Kolmogorov_Re35_N16_T30000_32x32.pickle"
# Pickle load

with open(fnstr, 'rb') as f:
    X = pk.load(f) #Data is ordered (t,Y,X)

X = np.expand_dims(X, axis=3)

#FOURIER COORDINATES
k1 = np.linspace(0,15,16)
k2 = np.linspace(-16,-1,16)
k = np.concatenate((k1, k2),axis=0)
kk, dump = np.meshgrid(k,k)


imag = tf.complex(0.0,1.0)

#Smooth Weights
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
    #def on_train_batch_end(self, batch, logs=None):
        weights = STN.get_layer("STN_LIN").get_weights()[0]
        weights = np.reshape(weights,[32,32,2])
        weigts_xp = tf.roll( weights, 1, axis=0)
        weigts_xm = tf.roll( weights, -1, axis=0)
        weigts_yp = tf.roll( weights, 1, axis=1)
        weigts_ym = tf.roll( weights, -1, axis=1)

        weigts_xpyp = tf.roll( weigts_xp, 1, axis=1)
        weigts_xpym = tf.roll( weigts_xp, -1, axis=1)
        weigts_xmyp = tf.roll( weigts_xm, 1, axis=1)
        weigts_xmym = tf.roll( weigts_xm, -1, axis=1)

        weights = (4*weights + 2*weigts_xp + 2*weigts_xm + 2*weigts_yp + 2*weigts_ym + weigts_xpyp + weigts_xpym + weigts_xmyp + weigts_xmym)/16
        weights = np.reshape(weights,[32*32,2])

        w = [weights]
        STN.get_layer("STN_LIN").set_weights(w)
    

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

#Find minimum loss
loss = tf.math.square(E-ET)
loss = tf.math.reduce_mean(loss,axis=(1,2,))
loss = tf.math.reduce_min(loss,axis=1)
loss = tf.math.reduce_mean(loss)

autoencoder.add_loss(loss)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
learning_rate,
decay_steps=2000,
decay_rate=0.2,
staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

autoencoder.compile(optimizer=opt)
earl_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience,  restore_best_weights=True,min_delta=min_delta)

history=autoencoder.fit(X,X,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[earl_stop,CustomCallback()],
                        verbose=1
                        )

autoencoder.save_weights('models/s-PCA_{}MODES'.format(n_modes))

