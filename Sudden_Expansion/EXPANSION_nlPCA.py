# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021
@author: svk20
###############################################################################
          NEURAL NETWORK TO GENERATE nlPCA FOR THE SUDDEN EXPANSION DATA
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
# This is an implementation to generate nonlinear PCA (nlPCA) from data using a neural net
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

output_img = DEC(ENC(input_img))

#Create model 
autoencoder = tf.keras.models.Model(input_img,output_img)

#Add L2 Loss
loss = tf.math.square(output_img-input_img)
loss = tf.math.reduce_mean(loss,axis=(1,2))
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

history=autoencoder.fit(X_OG, X_OG,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[earl_stop],
                        verbose=1
                        )

autoencoder.save_weights('models/nlPCA_{}MODES'.format(n_modes))

#make reconstruction
P = autoencoder.predict(X_OG, batch_size=50, verbose=1, steps=None, callbacks=None)

#Print reconstruction error
print('Reconstruction error with {} modes'.format(n_modes),np.mean((P-X_OG)**2)/np.mean((X_OG)**2))

