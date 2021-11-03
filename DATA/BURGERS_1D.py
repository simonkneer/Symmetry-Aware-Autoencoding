import matplotlib.cm as cm
import matplotlib.animation as animation
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import math

def main():
 
    n = 8
    x_0 = 0
    x_e = 2*np.pi

    N = 64
    T0 = 0
    T1 = 10
    dt = 0.0001
    dt_rec = 0.01

    
    
    x = np.linspace(x_0,x_e,N,endpoint=False)
    dx = x[1]-x[0]

    N_t = np.int((T1-T0)/dt)
    N_t_rec = np.int((T1-T0)/dt_rec)

    U = np.zeros([N_t,N])

    #Initial solutions
    U[0,:] = 1
    U[0,np.int(N/2):] = 2
    for i in range(N_t-1):
        cx = -U[i,:]

        U_1xm = np.roll(U[i,:],-1,axis=0)
        U_2xm = np.roll(U[i,:],-2,axis=0)

        U_1xp = np.roll(U[i,:], 1,axis=0)
        U_2xp = np.roll(U[i,:], 2,axis=0)
        
        #Upwinding
        sx = np.sign(cx)
        gx = 1/dx * (0.5 * (sx + 1) * (U[i,:]**2 - U_1xm**2) - 0.5 * (sx - 1) * (U_1xp**2 - U[i,:]**2))

        ddx = 1/dx**2 *(U_1xp -2*U[i,:] + U_1xm)

        U[i+1,:] = U[i,:] + dt * (1/2 * gx)
       
    U_OUT = np.zeros([N_t_rec,N])
    for i in range(N_t_rec):
        U_OUT[i,:] = U[i*np.int(N_t/N_t_rec),:]
    

    pk.dump( U_OUT, open( 'BURGERS_1D.pickle', "wb" ) ,protocol=pk.HIGHEST_PROTOCOL)
if __name__ == '__main__':
    main()

