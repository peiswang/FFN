# -*- coding: utf-8 -*-

import numpy as np, math, operator
import time
import pickle

def sdd(A, kmax=100, alphamin=0.001, lmax=1000, rhomin=0, max_epoch=20, savedir=None, prefix=None):
    #   SDD  Semidiscrete Decomposition.
    
    ### Check Input Arguments    
    try: 
        'A'
    except NameError:
        print('Incorrect number of inputs.')
    
    if 'rhomin' in locals():
        rhomin = rhomin**2

    # Initialization    
    [m,n] = A.shape         # size of A
    res = (np.linalg.norm(A,'fro'))**2

    # iitssav = np.zeros((kmax)) not use
    xsav = np.zeros((m,kmax))
    ysav = np.zeros((n,kmax))
    dsav = np.zeros((kmax,))
    
    A_copy = np.copy(A)
    # Outer loop
    start_time = time.time()
    for k in range(0,kmax):
        # A_copy = A - np.dot(xsav * dsav, ysav.T)  ## about 10s
        # A_copy = A - np.dot(xsav[:,:k] * dsav[:k], ysav[:,:k].T)
        # Initialize y by all ones 
        y = np.ones((n,))
                
        # Inner loop
        for l in range (0,lmax):
            # Fix y and Solve for x
            s = np.dot(A_copy,y)            
            [x, xcnt, _] = sddsolve(s, m)

            # Fix x and Solve for y
            s = np.dot(x, A_copy)
            [y, ycnt, fmax] = sddsolve(s, n)
            
            # Check Progress
            d = np.sqrt(fmax * ycnt) / (ycnt * xcnt)
            beta = d**2 * ycnt * xcnt #not use
            if l > 0: # python is zero-based
                 if dold==d and np.dot((x-xold), (x-xold))==0 and np.dot((y-yold),(y-yold))==0:
                     break

            xold = x
            yold = y
            dold = d
            #end l-loop

        # Save        
        xsav[:, k] = x            # shape conflict (matlab deals with this internally)        
        ysav[:, k] = y
        dsav[k] = d

        A_copy = A_copy - np.dot(xsav[:,k:k+1] * dsav[k], ysav[:,k:k+1].T)  ## about 5s
        # A_copy = A_copy - (xsav[:,k:k+1] * dsav[k] * ysav[:,k])  ## about 5s
        # A_copy = A_copy - (x[:, np.newaxis] * d * y)  ## about 5s

    end_time = time.time()
    print('sdd init ' + str(k+1) + ' of ' + str(kmax) + ' in', end_time-start_time, 's')

    #end k-loop

    print('sdd init done!')
    if savedir is not None:
        with open(savedir+'/'+prefix+'sdd_epoch'+str(1) + '.pkl', 'wb') as f:
            pickle.dump({'D': dsav, 'U': xsav, 'V': ysav}, f, pickle.HIGHEST_PROTOCOL)

    res_new = math.pow(np.linalg.norm(A-np.dot(np.dot(xsav,np.diag(dsav.flatten().tolist())),ysav.T),'fro'),2)
    res_old = res_new
    rho = res_new /res

    print(res_new)
    for epoch in range(max_epoch-1):
        print('sdd update epoch ' + str(epoch+2) + ' of ' + str(max_epoch))
        A_copy = np.copy(A)
        for k in range(kmax):
            y = ysav[:, k]
            dsav[k] = 0
            A_copy = A - np.dot(np.dot(xsav,np.diag(dsav.flatten().tolist())),ysav.T)
            for l in range (lmax):
                # Fix y and Solve for x
                s = np.dot(A_copy,y)            
                [x, xcnt, _] = sddsolve(s, m)
  
                # Fix x and Solve for y
                #s = np.dot(A.T,x)
                s = np.dot(x, A_copy)
                [y, ycnt, fmax] = sddsolve(s, n)
                
                # Check Progress
                d = np.sqrt(fmax * ycnt) / (ycnt * xcnt)
                beta = d**2 * ycnt * xcnt #not use  
              
                if l > 0: # python is zero-based
                     if dold==d and np.dot((x-xold).T, (x-xold))==0 and np.dot((y-yold).T,(y-yold))==0:
                         break
  
                xold = x
                yold = y
                dold = d
                #end l-loop
  
            # Save        
            xsav[:, k] = x.T            # shape conflict (matlab deals with this internally)        
            ysav[:, k] = y.T
            dsav[k] = d              # python is zero-based
        #end k-loop
        res_new = math.pow(np.linalg.norm(A-np.dot(np.dot(xsav,np.diag(dsav.flatten().tolist())),ysav.T),'fro'),2)
        print(res_new)
        if savedir is not None:
            with open(savedir+'/'+prefix+'sdd_epoch'+str(epoch+2) + '.pkl', 'wb') as f:
                pickle.dump({'D': dsav, 'U': xsav, 'V': ysav}, f, pickle.HIGHEST_PROTOCOL)

        #Threshold Test
        if res_old==0 or (abs(res_new-res_old)/res_old)<= rhomin:
          break

        res_old = res_new
        rho =res_new/res

    dsav = dsav.astype(np.float32)
    xsav = xsav.astype(np.float32)
    ysav = ysav.astype(np.float32)

    return dsav, xsav, ysav

################# SDD subproblem solver ############################
def sddsolve(s, m):
    #   SDDSOLVE Solve SDD subproblem
    #
    #For use with SDD.
    #Yannick De Bock, KU Leuven, 2014
    #
    #Derived from SDDPACK 
    #Tamara G. Kolda, Oak Ridge National Laboratory, 1999.
    #Dianne P. O'Leary, University of Maryland and ETH, 1999.

    x = np.ones((m,))   
    x[s<0] = -1
    s = np.abs(s)
    
    indexsort = np.argsort(-s)
    sorts = s[indexsort]
    f = np.add.accumulate(sorts)
    f = np.divide(f**2, np.arange(1,m+1,1))
    
    imax = np.argmax(f)
    fmax = f[imax]
    
    x[indexsort[imax+1:m]] = 0

    imax += 1                   # + 1 to correct imax
        
    return x, imax, fmax 

