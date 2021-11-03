# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021
@author: svk20
###############################################################################
          NEURAL NETWORK TO GENERATE nlPCA FOR THE KOLMOGOROV FLOW DATA
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
from tensorflow.keras import backend as K
from matplotlib import pyplot as plt
import pickle as pk


#Set learning rate
learning_rate=0.001
#Set batch size
batch_size=300
#Set number of epochs
epochs=10
#Set patience for early stopping
patience=5
#Set min delta for early stopping
min_delta = 1E-5
#Set steps for exponantial learning rate decay
decay_steps=2000
#Set decay rate for exponantial learning rate decay
decay_rate=0.2
#Set hidden dimension size
n_hidden=2048

#Set number of modes
n_modes=2

#Set Base for number of filters
BASE = 16


#Load flow-field data
#------------------------
fnstr="../DATA/Kolmogorov_Re35_N16_T30000_32x32.pickle"

# Pickle load

with open(fnstr, 'rb') as f:
    X = pk.load(f) #Data is ordered (t,Y,X)

X = np.expand_dims(X, axis=3)

#Set custom activation function
def custom_activation(x):
    return tf.nn.swish(x)

#Set up custom periodic padding because Keras doesn't support it natively
def padder(x):
    size = x.get_shape().as_list()
    b = tf.tile(x, [1,3, 3,1])
    result = b[:,size[1]-1:-(size[1]-1), size[1]-1:-(size[1]-1),:]
    size = result.get_shape().as_list()
    return result

#------------------------
#MODEL
#------------------------
input_img = ks.Input(shape=(X.shape[1], X.shape[2],1))

## Encoder
ENC = tf.keras.Sequential()
ENC.add(ks.Lambda(padder))
ENC.add(ks.Conv2D(BASE, (3, 3), activation=custom_activation, padding='valid'))
#32,32,4*BASE
ENC.add(ks.AveragePooling2D((2, 2), padding='same'))

ENC.add(ks.Lambda(padder))
ENC.add(ks.Conv2D(2*BASE, (3, 3), activation=custom_activation, padding='valid'))
#16,16,2*BASE
ENC.add(ks.AveragePooling2D((2, 2), padding='same'))

ENC.add(ks.Lambda(padder))
ENC.add(ks.Conv2D(4*BASE, (3, 3), activation=custom_activation, padding='valid'))
#8,8,BASE

ENC.add(ks.Reshape([8*8*4*BASE]))
ENC.add(ks.Dense(n_hidden,activation=custom_activation))
ENC.add(ks.Dense(n_modes,activation=None))

## Decoder
latent_inputs = ks.Input(shape=(n_modes))
DEC = tf.keras.Sequential()

DEC.add(ks.Dense(n_hidden,activation=custom_activation))
DEC.add(ks.Dense(8*8*4*BASE,activation=None))
DEC.add(ks.Reshape([8,8,4*BASE]))

DEC.add(ks.UpSampling2D((2,2)))
DEC.add(ks.Lambda(padder))
DEC.add(ks.Conv2D(2*BASE,(3,3),activation=custom_activation,padding='valid'))
#16,16,2*BASE

DEC.add(ks.UpSampling2D((2,2)))
DEC.add(ks.Lambda(padder))
DEC.add(ks.Conv2D(BASE,(3,3),activation=custom_activation,padding='valid'))
#32,32,4*BASE
DEC.add(ks.Lambda(padder))
DEC.add(ks.Conv2D(1,(3,3),activation=None,padding='valid'))


output_img = DEC(ENC(input_img))
autoencoder = tf.keras.models.Model(input_img,output_img)

loss = tf.math.square(output_img-input_img)
loss = tf.math.reduce_mean(loss,axis=(0,1,2,))

autoencoder.add_loss(loss)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
learning_rate,
decay_steps=decay_steps,
decay_rate=decay_rate,
staircase=False)
opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

autoencoder.compile(optimizer=opt)
earl_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience,  restore_best_weights=True,min_delta=min_delta)

history=autoencoder.fit(X,X,
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[earl_stop],
                        verbose=1
                        )

autoencoder.save_weights('models/nlPCA_{}MODES'.format(n_modes))


P = autoencoder.predict(X, batch_size=50, verbose=1, steps=None, callbacks=None)

#Print reconstruction error
print('Reconstruction error with {} modes'.format(n_modes),np.mean((P-X)**2)/np.mean((X)**2))
