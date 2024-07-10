#-------------------------------------------------------------------------------
# Name:        Variogram etc. calculation using FFT
#-------------------------------------------------------------------------------

import os
import time
import datetime
import random
import numpy as np
import matplotlib.pylab as plt
import scipy.stats as st
import scipy.spatial as sp
import scipy.optimize as opt
import scipy.io
import shutil as sh
import itertools as it
import IPython
import sys
from statsmodels.distributions.empirical_distribution import ECDF



def main():
    # EXAMPLE 1
    # that is the field that Denis used in the FFTMA-SA paper
    matfile = scipy.io.loadmat('asymfield.mat')
    afield = matfile['Z'][0,0]

    plt.figure()
    plt.imshow(afield, interpolation='nearest', origin='lower')
    plt.title('Input field')
    plt.colorbar()
    plt.show()

    # calculate directional asymmetry function using the 
    # whole field as input
    gh = varioFFT2D(afield, icode=2)

    # plot directional asymmetry
    plt.figure()
    plt.imshow(gh)
    plt.title('Directional asymmetry')
    plt.colorbar()
    plt.show()

    gh = varioFFT2D(afield, icode=3)
    # plot asymmetry
    plt.figure()
    plt.imshow(gh)
    plt.title('Order asymmetry')
    plt.colorbar()
    plt.show()

    # EXAMPLE 2
    # do the same with just 1000 points randomly sampled from afield
    # here we only have 1000 point values but we need to create 
    # a regular grid with nan-s at locations without 
    # values
    xy = np.mgrid[[slice(0, afield.shape[i], 1) for i in range(2)]].reshape(2,-1).T
    np.random.shuffle(xy)
    xy = xy[:1000]

    vals = afield[xy[:,0], xy[:,1]]

    # check max coordinates
    maxx = int(xy.max() + 1)
    field = np.ones((maxx,maxx)) * np.nan
    field[xy[:,0], xy[:,1]] = vals

    gh = varioFFT2D(field, icode=2)

    # plot directional asymmetry
    plt.figure()
    plt.imshow(gh)
    plt.title('Directional asymmetry from sparse data')
    plt.colorbar()
    plt.show()




    # EXAMPLE 3
    x_y_vals = np.loadtxt('vals.txt')
    # check max coordinates
    maxx = int(x_y_vals[:,:2].max() + 1)
    # and create a regagular grid with them
    # which has nan at locations with values
    field = np.ones((maxx,maxx)) * np.nan
    field[x_y_vals[:,0].astype(int), x_y_vals[:,1].astype(int)] = x_y_vals[:,2]

    # calculate the variogram
    gh = varioFFT2D(field, icode=1)

    # take only the middle window from -2*xr to +2*xr
    xr = 50
    w = [int(gh.shape[i]/2) for i in range(2)]
    gh = gh[w[0]-xr:w[0]+xr,w[1]-xr:w[1]+xr]
    plt.figure()
    plt.imshow(gh)
    plt.title('Variogram')
    plt.colorbar()
    plt.show()

    # calculate the rank correlation
    gh = varioFFT2D(field, icode=4)

    # take only the middle window from -2*xr to +2*xr
    xr = 50
    w = [int(gh.shape[i]/2) for i in range(2)]
    gh = gh[w[0]-xr:w[0]+xr,w[1]-xr:w[1]+xr]
    plt.figure()
    plt.imshow(gh)
    plt.title('Rank correlation')
    plt.colorbar()
    plt.show()

    # average over the four main directions to get a 1d 'isotropic' rank correlation
    rf1 = gh[50:, 50]
    rf2 = gh[:50, 50][::-1]
    rf3 = gh[50, 50:]
    rf4 = gh[50, :50][::-1]

    rf = np.mean((rf1, rf2, rf3, rf4), axis=0)

    plt.figure()
    plt.plot(rf)
    plt.title('Rank correlation')
    plt.show()



def varioFFT2D(z, icode=1):
    '''
    z needs to be a regular grid with missing values indicated by nan
    '''

    # find closest multiple of 8
    nn = []
    for i in range(z.ndim):
        nn.append(np.int(np.ceil(((2*z.shape[i])-1.)/8.)*8.))
    nn = np.array(nn)


    # create an indicator matrix with 1s for data and 0s for missing
    Z = np.copy(z)
    Z[np.isnan(z)] = 0
    Zid = np.ones(z.shape)
    Zid[np.isnan(z)] = 0


    # compute the number of pairs
    fx = np.fft.fftn(Z, nn)
    fxid = np.fft.fftn(Zid, nn)
    fx2 = np.fft.fftn(Z*Z, nn)

    nh = np.round(np.real(np.fft.ifftn(np.conj(fxid)*fxid)))


    # compute the different functions accroding to icode
    if icode == 1:
        # variogram
        t1 = np.fft.fftn(Z*Zid, nn)
        t2 = np.fft.fftn(Z*Zid, nn)
        t12 = np.fft.fftn(Z*Z, nn)
        gh = np.real(np.fft.ifftn(np.conj(fxid)*t12 + np.conj(t12)*fxid - np.conj(t1)*t2 - t1*np.conj(t2)))/np.maximum(nh,1)/2

    elif icode == 2:
        # asymmetry Bardossy and Hoerning (2017)
        if np.isnan(z).any():
            zvals = z[~np.isnan(z)]
            ecdf = ECDF(zvals)
            Fz = ecdf(zvals)
            # bring values back into matrix 
            F = np.copy(z)
            F[np.isnan(z)] = 0.
            F[~np.isnan(z)] = Fz
        else:
            ecdf = ECDF(z.flatten())
            F = ecdf(z.flatten()).reshape(z.shape)

        f3 = np.fft.fftn(F*F*F, nn)
        f2 = np.fft.fftn(F*F, nn)
        f1 = np.fft.fftn(F, nn)
        gh = np.real(np.fft.ifftn(np.conj(f3)*fxid - 3.*np.conj(f2)*f1 + 3.*np.conj(f1)*f2 - np.conj(fxid)*f3))/np.maximum(nh,1)

    elif icode == 3:
        # asymmetry Bardossy and Guthke (2016)
        if np.isnan(z).any():
            zvals = z[~np.isnan(z)]
            ecdf = ECDF(zvals)
            Fz = ecdf(zvals)
            # bring values back into matrix 
            F = np.copy(z)
            F[np.isnan(z)] = 0.
            F[~np.isnan(z)] = Fz
        else:
            ecdf = ECDF(z.flatten())
            F = ecdf(z.flatten()).reshape(z.shape)

      
        f3 = np.fft.fftn(F*F*F, nn)
        f2 = np.fft.fftn(F*F, nn)
        f1 = np.fft.fftn(F, nn)
        # IPython.embed()
        gh = np.real(np.fft.ifftn(np.conj(f3)*fxid + 3.*np.conj(f2)*f1 - 3.*np.conj(f2)*fxid \
                                                   + 3.*np.conj(f1)*f2 - 6.*np.conj(f1)*f1 + 3.*np.conj(f1)*fxid \
                                                   + np.conj(fxid)*f3 - 3.*np.conj(fxid)*f2 + 3.*np.conj(fxid)*f1 - np.conj(fxid)*fxid))/np.maximum(nh,1)


    elif icode == 4:
        # rank correlation
        if np.isnan(z).any():
            zvals = z[~np.isnan(z)]
            ecdf = ECDF(zvals)
            Fz = ecdf(zvals)
            # bring values back into matrix 
            F = np.copy(z)
            F[np.isnan(z)] = 0.
            F[~np.isnan(z)] = Fz
        else:
            ecdf = ECDF(z.flatten())
            F = ecdf(z.flatten()).reshape(z.shape)

        f1 = np.fft.fftn(F, nn)
        gh = 12. * np.real(np.fft.ifftn(np.conj(f1)*f1 - 0.5*np.conj(f1)*fxid - 0.5*np.conj(fxid)*f1 + 0.25*np.conj(fxid)*fxid))/np.maximum(nh,1)

    else:
        print('icode not defined')
        raise Exception


 
    # reduce matrices to required size
    t = (nn/2 + 1).astype(int)
    n = z.shape[0]
    p = z.shape[1]
    nh = np.fft.fftshift(nh)[t[0]-n:t[0]+n-1, t[1]-p:t[1]+p-1]
    gh = np.fft.fftshift(gh)[t[0]-n:t[0]+n-1, t[1]-p:t[1]+p-1]
    

    return gh



if __name__ == '__main__':
    main()

