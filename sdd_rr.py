# -*- coding: utf-8 -*-
import numpy as np, math, operator
import time
import torch

def sdd_rr(X, Y, D, U, V, max_epoch=20):

    D = D.astype(np.float32)
    U = U.astype(np.float32)
    V = V.astype(np.float32)

    kmax = len(D)
    m = U.shape[0]
    n = V.shape[0]

    X_row_norm = (X*X).sum(axis=1)

    for epoch in range(max_epoch):
        print(str(epoch)+' of '+str(max_epoch))
        start_time = time.time()
        for k in range(kmax):
            #print(str(k)+' of '+str(kmax))
            # given u, v, solve d
            D[k] = 0
            u = U[:, k]
            v = V[:, k]
            E = Y - np.dot(np.dot(np.multiply(U, D), V.T), X) #E = Y - UDV^tX
            vtX = np.dot(v, X)
            Etu = np.dot(u, E)
            d = np.dot(vtX, Etu) / (np.dot(u,u)*np.dot(vtX,vtX))
            D[k] = d
            # given d, v, solve u
            min_score = None
            t = 2*d*np.dot(E, vtX)
            indexsort = np.argsort(-np.abs(t))
            for j in range(m+1):
                score = d**2*np.dot(vtX, vtX)*j - np.abs(t[indexsort[0:j]]).sum()
                if min_score is None or score < min_score:
                    min_score = score
                    best_j = j
            u = np.sign(t)
            u[indexsort[best_j:len(u)]] = 0
            U[:, k] = u
            # given d, u, solve v
            t = 2*d*np.dot(X, np.dot(u, E)) # t=2du^tEX^t
            dduu = d**2*np.dot(u,u)
            s = vtX
            for j in range(n):
                x = X[j, :]
                s = s - v[j]*x  # s = hat(X)^T*hat(v)
                tmp = 2*dduu*np.dot(x,s)-t[j]
                # if dduu*X_row_norm[j]-abs(tmp) < 0:
                if dduu*np.dot(x,x)-abs(tmp) < 0:
                    v[j] = -np.sign(tmp)
                else:
                    v[j] = 0
                s = s + v[j]*x
            V[:, k] = v
        end_time = time.time()
        print('solve one iter in', end_time-start_time, 's')

    return D, U, V

