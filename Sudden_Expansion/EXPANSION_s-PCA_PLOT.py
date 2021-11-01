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
import scipy

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

#Set number of modes for NN
n_modes=1
#Set number of modal shapes to show
n_modes_show = 4
#Set max_modes for calculating PCA
max_modes = 150
#set max modes for energy plotting
n_modes_eneg = 30

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
ENC.add(ks.Dense(n_modes,use_bias=False,activation=None))
## Decoder
DEC = tf.keras.Sequential()
DEC.add(ks.Dense(X.shape[1]*X.shape[2],use_bias=False,activation=None))
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


autoencoder.load_weights('models/s-PCA_{}MODES'.format(n_modes))


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
X_T_OUTS,X_T_INS = PRINTER.predict([X_OG, X_FLIP], batch_size=50, verbose=1, steps=None, callbacks=None)
#extract chosen branch
INDICES = INDICER.predict([X_OG, X_FLIP], batch_size=batch_size, verbose=1, steps=None, callbacks=None)

plt.figure('CHOSEN INDICES')
plt.plot(INDICES)
plt.xlabel('$t$')
plt.ylabel('$l$')


#Calculate s-PCA modes
m=X_T_INS.shape[0]
X_PCA = X_T_INS.reshape((X_T_INS.shape[0], int(X_T_INS.shape[1]*X_T_INS.shape[2])))

#Covariance matrix
R=np.matmul(X_PCA,X_PCA.T)/m

#Eigenvalues and vectors
w, v = scipy.linalg.eigh(R,subset_by_index = [R.shape[0]-max_modes, R.shape[0]-1] )
w = w[::-1]
v = v[:,::-1]
modes=np.matmul(X_PCA.T,v)/np.sqrt(abs(w))

modes = modes.T.reshape(max_modes,X_T_INS.shape[1],X_T_INS.shape[2])

x_main = np.linspace(0,15,300)
y_main = np.linspace(-1.5,1.5,60)

x_in_dim = 59

x_in = np.linspace(-3,0,60)
y_in = np.linspace(-0.5,0.5,20)

xx_main,yy_main = np.meshgrid(x_main,y_main)
xx_in,yy_in = np.meshgrid(x_in,y_in)

we = 18
h = 5

for i in range(n_modes_show):
    P_MAIN = np.reshape(modes[i,x_in_dim*y_in.shape[0]:,:],[x_main.shape[0],y_main.shape[0],2])
    P_IN = np.reshape(modes[i,:x_in_dim*y_in.shape[0],:],[x_in_dim,y_in.shape[0],2])

    P_IN = np.concatenate([P_IN, np.expand_dims(P_IN[-1,:,:],axis=0)],axis=0)

    ma_1 = np.amax(np.abs(P_MAIN[:,:,0]))
    ma_2 = np.amax(np.abs(P_IN[:,:,0]))
    max_u = np.amax([ma_1,ma_2])

    #Set corners to max values for easy colormap consistency
    P_MAIN[0,0,0] = max_u
    P_MAIN[1,0,0] = -max_u
    P_IN[0,0,0] = max_u
    P_IN[1,0,0] = -max_u


    plt.figure("U MODE {}".format(i+1),figsize=(we, h))

    plt.contourf(xx_main,yy_main,P_MAIN[:,:,0].T)
    plt.contourf(xx_in,yy_in,P_IN[:,:,0].T)
    plt.ylim(-1.5,1.5)
    plt.xlim(-3,15)
    plt.xlabel('$x$')
    plt.ylabel('$y$')
    plt.tight_layout()
    ax = plt.gca()
    ax.set_yticks([-1.5, 0,1.5])



#Plot Normalized Modal Energies
w = w/np.sum(w)
N_modes = np.linspace(1,n_modes_eneg,n_modes_eneg)
plt.figure('Energies')
plt.scatter(N_modes,w[:n_modes_eneg])
plt.ylabel('$e_i$')
plt.xlabel('$i$')
plt.yscale('log')
plt.xlim([1,n_modes_eneg])
plt.ylim([1E-3,1])



plt.show()

