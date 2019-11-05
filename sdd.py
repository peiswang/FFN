# -*- coding: utf-8 -*-

import numpy as np, math, operator
import time
import pickle

def sdd(A, kmax=100, alphamin=0.001, lmax=1000, rhomin=0, yinit=2, max_epoch=20, savedir = None):
    #   SDD  Semidiscrete Decomposition.
    
    ### Check Input Arguments    
    try: 
        'A'
    except NameError:
        print('Incorrect number of inputs.')
    
    if 'rhomin' in locals():
        rhomin = rhomin**2
    idx = 0             # only used for yinit = 1 (python is zero-based contrary to matlab)
    
    # Initialization    
    [m,n] = A.shape         # size of A
    res = (np.linalg.norm(A,'fro'))**2

    # iitssav = np.zeros((kmax)) not use
    xsav = np.zeros((m,kmax))
    ysav = np.zeros((n,kmax))
    dsav = np.zeros((kmax,))
    
    # Outer loop
    for k in range(0,kmax):
        print('sdd init ' + str(k+1) + ' of ' + str(kmax))
        # Initialize y for inner loop
        if yinit == 0:          # Threshold
            s = np.zeros((m,))
            iits = 0
            while math.pow(np.linalg.norm(s),2) < (float(rho)/n):                
                y = np.zeros((n,))                     
                y[idx] = 1
                s = np.dot(A,y)
                if k>0:       # python is zero-based             
                    s = s - (np.dot(xsav,(np.multiply(dsav,(np.dot(ysav.T,y))))))                    
                    
                idx = np.mod(idx, n) + 1
                if idx == n:        # When idx reaches n it should be changed to zero (otherwise an index out of bounds error will occur)
                    idx = 0
                iits = iits + 1
            
        elif yinit == 1:        # Cycling Periodic Ones
            y = np.zeros((n,))
            index = np.mod(k-1,n)+1
            if index < n:                 
                y[index] = 1
            else:
                y[0] = 1   
        elif yinit == 2:        # All Ones
            y = np.ones((n,))
        elif yinit == 3:        # Periodic Ones
            y = np.zeros((n,))
            ii = np.arange(0,n,100)
            for i in ii: # python is zero-based
                y[i] = 1 
        else:
            try:
                pass
            except ValueError:
                print('Invalid choice for C.')
                
        # Inner loop
        for l in range (0,lmax):
            # Fix y and Solve for x
            s = np.dot(A,y)            
            if k > 0:       # python is zero-based
                s = s - np.dot(xsav, np.multiply(dsav, np.dot(ysav.T, y)))
            [x, xcnt, _] = sddsolve(s, m)

            # Fix x and Solve for y
            s = np.dot(x, A)
            if k > 0:
                s = s - np.dot(ysav, np.multiply(dsav, np.dot(xsav.T, x)))
            [y, ycnt, fmax] = sddsolve(s, n)
            
            # Check Progress
            d = np.sqrt(fmax * ycnt) / (ycnt * xcnt)
            beta = d**2 * ycnt * xcnt #not use
            if l > 0: # python is zero-based
                 if np.dot((x-xold).T, (x-xold))==0 and np.dot((y-yold).T,(y-yold))==0 and dold==d:
                     break

            xold = x
            yold = y
            dold = d
            #end l-loop

        # Save        
        xsav[:, k] = x.T            # shape conflict (matlab deals with this internally)        
        ysav[:, k] = y.T
        dsav[k] = d
    #end k-loop

    print('sdd init done!')
    if savedir is not None:
        with open(savedir+'_epoch'+str(1) + '.pkl', 'wb') as f:
            pickle.dump({'D': dsav, 'U': xsav, 'V': ysav}, f, pickle.HIGHEST_PROTOCOL)

    res_new = math.pow(np.linalg.norm(A-np.dot(np.dot(xsav,np.diag(dsav.flatten().tolist())),ysav.T),'fro'),2)
    res_old = res_new
    rho = res_new /res

    print(res_new)
    for epoch in range(max_epoch-1):
        print('sdd update epoch ' + str(epoch+2) + ' of ' + str(max_epoch))
        for k in range(kmax):
            y = ysav[:, k]
            for l in range (lmax):
                # Fix y and Solve for x
                s = np.dot(A,y)            
                s = s - np.dot(xsav, np.multiply(dsav, np.dot(ysav.T,y)))
                s = s + np.dot(xsav[:,k], np.multiply(dsav[k], np.dot(ysav[:,k].T, y)))
                [x, xcnt, _] = sddsolve(s, m)
  
                # Fix x and Solve for y
                #s = np.dot(A.T,x)
                s = np.dot(x, A)
                s = s - (np.dot(ysav,(np.multiply(dsav,(np.dot(xsav.T,x))))))
                s = s + np.dot(ysav[:,k], np.multiply(dsav[k], np.dot(xsav[:,k].T, x)))
                [y, ycnt, fmax] = sddsolve(s, n)
                
                # Check Progress
                d = np.sqrt(fmax * ycnt) / (ycnt * xcnt)
                beta = d**2 * ycnt * xcnt #not use  
              
                if l > 0: # python is zero-based
                     if np.dot((x-xold).T, (x-xold))==0 and np.dot((y-yold).T,(y-yold))==0 and dold==d:
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
            with open(savedir+'_epoch'+str(epoch+2) + '.pkl', 'wb') as f:
                pickle.dump({'D': dsav, 'U': xsav, 'V': ysav}, f, pickle.HIGHEST_PROTOCOL)

        #Threshold Test
        if res_old==0 or (abs(res_new-res_old)/res_old)<= rhomin:
          break

        res_old = res_new
        rho =res_new/res

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

