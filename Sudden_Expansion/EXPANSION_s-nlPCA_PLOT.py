# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021
@author: svk20
###############################################################################
          NEURAL NETWORK TO PLOT s-nlPCA FOR THE SUDDEN EXPANSION DATA
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
# This is an implementation to plot symmetry aware nonlinear PCA (s-nlPCA) from data using a neural net
#INPUT:    /DATA
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

#Disable GPU for plotting if the GPU memory is not large enough
tf.config.set_visible_devices([], 'GPU')


#Set learning rate
learning_rate=0.001
#Set batch size
batch_size=100
#Set number of epochs
epochs=10
#Set patience for early stopping
patience=50
#Set min delta for early stopping
min_delta = 1E-7
#Set steps for exponantial learning rate decay
decay_steps=1000
#Set decay rate for exponantial learning rate decay
decay_rate=0.2
#Set hidden dimension size
nonlinear_dim=512

#Set number of modes
n_modes=1


def custom_activation(x):
    return tf.nn.swish(x)


#Load flow-field data
"""
Note:
Due to the unstructed form of the data for the sudden expansion case,
i.e. each sample being a stacked vector of inflow and main domain, 
we have prepared a modified dataset that is one dimension larger than
the original data. This dimension contains the original flow field at 
position one and the reflected state at position two.
"""
#------------------------
fnstr="DATA/State_ALL_rect.pickle"
# Pickle load
with open(fnstr, 'rb') as f:
    X = pk.load(f)
#Original Field
X_OG = np.expand_dims(X[:,:,:,0],axis=3)
#Flipped Field
X_FLIP = np.expand_dims(X[:,:,:,1],axis=3)
    
#MODEL
#------------------------
input_img = ks.Input(shape=(X.shape[1], X.shape[2],1))
input_flip = ks.Input(shape=(X.shape[1], X.shape[2],1))
## Encoder
ENC = tf.keras.Sequential()
ENC.add(ks.Reshape([X.shape[1]*X.shape[2]]))
ENC.add(ks.Dense(nonlinear_dim,use_bias=True,activation=custom_activation))
ENC.add(ks.Dense(n_modes,use_bias=True,activation=None))
## Decoder
DEC = tf.keras.Sequential()
DEC.add(ks.Dense(n_modes,use_bias=True,activation=custom_activation))
DEC.add(ks.Dense(X.shape[1]*X.shape[2],use_bias=True,activation=None))
DEC.add(ks.Reshape([X.shape[1],X.shape[2],1]))

INPUTS = []
INPUTS.append(input_img)
INPUTS.append(input_flip)
INPUT_VECTOR = ks.Concatenate(axis=3)(INPUTS)


#Create list containing both reconstructions of original and flipped image
PASS_TROUGH = []
for i in range(2):
    PASS_TROUGH.append(DEC(ENC(INPUTS[i])))

RECONSTRUCTION_VECTOR = ks.Concatenate(axis=3)(PASS_TROUGH)

#Create model reconstructing both inputs
autoencoder = tf.keras.models.Model([input_img, input_flip], [RECONSTRUCTION_VECTOR, INPUT_VECTOR])

#Find minimum loss in the last dimension, i.e. the one containing 
#the flipped and the original state

loss = tf.math.square(RECONSTRUCTION_VECTOR-INPUT_VECTOR)
loss = tf.math.reduce_mean(loss,axis=(1,2))
loss = tf.math.reduce_min(loss,axis=1)
loss = tf.math.reduce_mean(loss)
autoencoder.add_loss(loss)


lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
learning_rate,
decay_steps=decay_steps,
decay_rate=decay_rate,
staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
autoencoder.compile(optimizer=opt)


earl_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience,  restore_best_weights=True)


autoencoder.load_weights('models/s-nlPCA_{}MODES'.format(n_modes))


#Create Printer Model 
OUTS, INS = autoencoder([input_img, input_flip])
LSS = tf.math.square(OUTS-INS)
LSS = tf.math.reduce_mean(LSS,axis=(1,2))
IND = tf.math.argmin(LSS,axis=1)

CHOSEN_OUTS = tf.gather(OUTS, IND, axis=3,batch_dims=1)
CHOSEN_INS = tf.gather(INS, IND, axis=3,batch_dims=1)
PRINTER = tf.keras.models.Model([input_img, input_flip], [CHOSEN_OUTS, CHOSEN_INS])
INDICER = tf.keras.models.Model([input_img, input_flip], IND)


#clear session
#tf.keras.backend.clear_session()

#make reconstruction
P,PT = PRINTER.predict([X_OG, X_FLIP], batch_size=50, verbose=1, steps=None, callbacks=None)
#extract chosen branch
INDICES = INDICER.predict([X_OG, X_FLIP], batch_size=batch_size, verbose=1, steps=None, callbacks=None)

plt.figure('CHOSEN INDICES')
plt.plot(INDICES)
plt.xlabel('$t$')
plt.ylabel('$l$')
plt.show()

