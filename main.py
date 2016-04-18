'''
Created on Oct 10, 2013

@author: rhein
'''
import traceback
import numpy as np
import scipy as sp
import scipy.ndimage as spim
import matplotlib.pyplot as plt 
import skfmm

def main():
    '20130824164324-cam12010988-1-75900-35423847.jpg'
    im = spim.imread('20130824164324-cam12010988-1-75900-35423847.jpg', True)
#     im = im[im.shape[0] / 2:, im.shape[1] / 2:]
    
    N = np.gradient(im)
    N = np.concatenate((N[1][..., None], N[0][..., None], np.ones_like(N[0][..., None])), axis=2)
    N = N / np.sqrt((N ** 2).sum(2))[..., None]
    
    L = N.dot([0, 0, 1])
    epsilon = 1e-9
     
    W = np.sqrt(1 / L ** 2 - 1)
    W[W < epsilon] = epsilon
    W[~np.isfinite(W)] = epsilon
     
    p = np.empty_like(W)
    p[...] = -1
#     p[704, 1485] = 1
#     p[193, 159] = 1
    p[L > (1 - epsilon)] = 1
     
    t = skfmm.travel_time(p, 1 / W)

    plt.figure()     
    plt.imshow(im, cmap='gray') 
    plt.figure()
    plt.imshow(N, cmap='gray')
    plt.figure()
    plt.imshow(L, cmap='gray')
    plt.figure()
    plt.imshow(W, cmap='gray')
    plt.figure()
    plt.imshow(p, cmap='gray')
    plt.figure()
    plt.imshow(-t, cmap='gray')
    
    
    plt.show()

if __name__ == '__main__':
    try:
        main()
    except Exception, e:
        traceback.print_exc()
