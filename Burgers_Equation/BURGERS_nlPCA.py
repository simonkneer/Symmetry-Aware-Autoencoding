# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021

@author: svk20

###############################################################################
          NEURAL NETWORK TO GENERATE nlPCA FOR THE BURGERS' DATA
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
#INPUT:    ../DATA
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
fnstr="../DATA/BURGERS_1D.pickle"
with open(fnstr, 'rb') as f:
    X = pk.load(f) #Data is ordered (t,Y,X)
#Subtract value for infinity
X = X - 1.5


#Set activation function
def custom_activation(x):
    return tf.nn.swish(x)


input_img = ks.Input(shape=(X.shape[1]))

## Encoder
ENC = tf.keras.Sequential()
ENC.add(ks.Dense(512,activation=custom_activation,name='ENC_1'))
ENC.add(ks.Dense(n_modes, activation=None,use_bias=True,name='ENC_2'))

## Decoder
latent_inputs = ks.Input(shape=(n_modes))
DEC = tf.keras.Sequential()
DEC.add(ks.Dense(512,activation=custom_activation,name='DEC_1'))
DEC.add(ks.Dense(X.shape[1] , activation=None,use_bias=True,name='DEC_2'))

output_img = DEC(ENC(input_img))
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
                        callbacks=[earl_stop],
                        verbose=1
                        )


#save weights
autoencoder.save_weights('models/s-nlPCA_{}MODES'.format(n_modes))

#Model for extracting latent variables
latents = ENC(input_img)
INTERS = tf.keras.models.Model(input_img,latents)


#Make reconstruction
P = autoencoder.predict(X, batch_size=batch_size, verbose=1, steps=None, callbacks=None)

#Print reconstruction error
print('Reconstruction error with {} modes'.format(n_modes),np.mean((P-X)**2)/np.mean((X)**2))


