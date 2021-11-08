# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021
@author: svk20
###############################################################################
          GENERATE PCA FOR THE KOLMOGOROV FLOW DATA
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
import tensorflow as tf
from matplotlib import pyplot as plt
from matplotlib import ticker as ticker
import pickle as pk
import scipy

#Number of reconstruction modes
n_modes=6
#Number of modes to be displayed in comparison
max_modes=30


#Load flow-field data
#------------------------
fnstr="../DATA/Kolmogorov_Re35_N16_T30000_32x32.pickle"

# Pickle load
with open(fnstr, 'rb') as f:
    D = pk.load(f)

D = np.expand_dims(D, axis=3)

X = D.reshape((D.shape[0], int(D.shape[1]*D.shape[2]*D.shape[3])))
M = np.expand_dims((np.mean(X,axis=0)),axis=0)

#Substract Mean
#X = X-M

#Covariance matrix
m=X.shape[1]
R=np.matmul(X.T,X)/m

#Eigenvalues and vectors
w, v = scipy.linalg.eigh(R,subset_by_index = [R.shape[0]-max_modes, R.shape[0]-1] )
w = w[::-1]
v = v[:,::-1]

#Recreate spatial eigenfunctions from snapshot ones
modes=np.matmul(X,v)/np.sqrt(abs(w))


#Coefficients
amplitudes=np.matmul(modes.T,X.T)

modes = modes[:,:n_modes]
calc_amp = amplitudes[:n_modes,:]

reconstructed=np.matmul(calc_amp.T,modes.T)

#Calculate ERROR field
print('Reconstruction error with {} modes'.format(n_modes),np.mean((reconstructed-X)**2)/np.mean((X)**2))

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
