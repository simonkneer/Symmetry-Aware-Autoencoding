# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021

@author: svk20

###############################################################################
          GENERATE PCA FOR THE SUDDEN EXPANSION DATA
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
# This is an implementation to generate PCA from data using the eigenvalue problem
#INPUT:    ../DATA
#          
#
#OUTPUT:   


###############################################################################
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle as pk
from scipy import linalg

#Number of reconstruction modes
n_modes = 4
#Number of modes to be displayed in comparison
max_modes=150
#Number of mode for energy display
n_modes_eneg = 30


#Load flow-field data
#------------------------
fnstr="../DATA/Sudden_Expansion_Re150_Rectilinear.pickle"

# Pickle load
with open(fnstr, 'rb') as f:
    D = pk.load(f)
D=np.asarray(D)
D = D.astype(np.float32)
D = np.concatenate([D[:,:,:,0],D[:,:,:,1]],axis=0)


X = D.reshape((D.shape[0], int(D.shape[1]*D.shape[2]))).T
#Covariance matrix
m=X.shape[1]
R=np.matmul(np.transpose(X),X)/m

#Eigenvalues and vectors
w, v = linalg.eigh(R,subset_by_index = [R.shape[0]-max_modes, R.shape[0]-1] )
w = w[::-1]
v = v[:,::-1]


#Recreate reconstructed state and calc error_field
reconstructed=np.zeros((int(D.shape[1]*D.shape[2]),D.shape[0]))

#Coefficients
amplitudes=np.matmul(np.transpose(modes),X)
#reconstructed=np.matmul(modes,amplitudes)
for t in range(D.shape[0]):
    for i in range(n_modes):
        reconstructed[:,t]=reconstructed[:,t]+amplitudes[i,t]*modes[:,i]


#Calculate ERROR field
error_field = np.mean(np.square(reconstructed-X),axis=1)
#Reshape to spatial dim 
print('Reconstruction error with {} modes'.format(n_modes),np.mean((error_field)**2)/np.mean((X)**2))
reconstructed = np.reshape(np.transpose(reconstructed),(D.shape[0],D.shape[1],D.shape[2]))
error_field = error_field.reshape((D.shape[1],D.shape[2]))

#Rehshape and plot modes
modes = modes.T.reshape(max_modes,D.shape[1],D.shape[2])

Nt = D.shape[0]
x_main = np.linspace(0,15,300)
y_main = np.linspace(-1.5,1.5,60)

x_in_dim = 59

x_in = np.linspace(-3,0,60)
y_in = np.linspace(-0.5,0.5,20)

xx_main,yy_main = np.meshgrid(x_main,y_main)
xx_in,yy_in = np.meshgrid(x_in,y_in)

#width and height of the mode plots
wi = 18
h = 5


for i in range(n_modes):
    P_MAIN = np.reshape(modes[i,x_in_dim*y_in.shape[0]:,:],[x_main.shape[0],y_main.shape[0],2])
    P_IN = np.reshape(modes[i,:x_in_dim*y_in.shape[0],:],[x_in_dim,y_in.shape[0],2])

    P_IN = np.concatenate([P_IN, np.expand_dims(P_IN[-1,:,:],axis=0)],axis=0)

    ma_1 = np.amax(np.abs(P_MAIN[:,:,0]))
    ma_2 = np.amax(np.abs(P_IN[:,:,0]))
    max_u = np.amax([ma_1,ma_2])

    #Set corner values for inflow and main domain to max values for easy colormap matching
    P_MAIN[0,0,0] = max_u
    P_MAIN[1,0,0] = -max_u
    P_IN[0,0,0] = max_u
    P_IN[1,0,0] = -max_u


    plt.figure("U MODE {}".format(i+1),figsize=(wi, h))

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
plt.ylim([1E-13,1])

plt.show()
