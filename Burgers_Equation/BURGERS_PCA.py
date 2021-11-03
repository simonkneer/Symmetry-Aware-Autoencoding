# -*- coding: utf-8 -*-
"""
Created on Mon OCT 15 08:48:56 2021

@author: svk20

###############################################################################
          NEURAL NETWORK TO GENERATE PCA FOR THE BURGERS' DATA
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
#INPUT:    /DATA
#          
#
#OUTPUT:   


###############################################################################
"""
import numpy as np
from matplotlib import pyplot as plt
import pickle as pk

#Set number of modes for PCA on transformed data
n_modes=4
#Set Modes for which to show energies
n_modes_eneg = 30

#load flow-field data
#------------------------
fnstr="DATA/BURGERS_1D.pickle"
with open(fnstr, 'rb') as f:
    X = pk.load(f) #Data is ordered (t,Y,X)

#Subtract value for infinity
X = X -1.5
#Do PCA on samples
m = X.shape[0]
R=np.matmul(X.T,X)

#Eigenvalues and vectors
w, modes = np.linalg.eig(R)

#Coefficients
amplitudes=np.matmul(modes.T,X.T)

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


plt.show()
