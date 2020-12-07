'''Implements 10-fold cross validation to find s2

Main problem: where to look for grid search when doing kfold CV
   - Approach 1: search when given a true sigma^2
'''

import numpy as np
from sklearn.model_selection import KFold

def MMalgCV(X, y, cv, s2_true):    # here we are given the true s2 to find where to search
    n, p = X.shape
    
    # set up the grid search for regularisation s2
    ngrid = 40
    s2_grid = np.logspace(np.log10(s2_true/100), np.log10(s2_true*10), ngrid)
    tloss_grid = np.full((ngrid, cv), np.nan)
    dg = np.ones((p, 1))# temporary delta vector resused from iterations to speed up things
    
    kf = KFold(n_splits = cv, shuffle = True)
    #looping through each fold
    count = 0
    for train_index, test_index in kf.split(X):
        X_train = X[train_index, :]
        X_test = X[test_index, :]
        y_train = y[train_index]
        y_test = y[test_index]
        
        # QR decomposition, overwrite X with Q
        X_train, R_train = np.linalg.qr(X_train)
        # X_test, R_test = np.linalg.qr(X_test)
        z_train = X_train.T @ y_train
        # z_test = X_test.T @ y_test
        # yqz_test = np.sum((y_test - X_test @ z_test)**2)   # excess SSR
        
        dg = np.ones((p, 1))
        for i in range(ngrid):
            bg, dg, _ = bhat(s2_grid[i], dg, R_train, z_train, tol = 10**-5)
            # tloss_grid[i, count] = yqz_test + np.sum((z_test - R_test @ bg)**2)   # Sum square error
            tloss_grid[i, count] = np.sum((y_test - X_test @ bg)**2)   # Sum square error
        count += 1
    
    # we sum up the test errors for each test fold to determine best s2
    s2_cv = s2_grid[np.sum(tloss_grid, axis = 1).argmin()]
    
    # QR decomposition of the full data
    X, R = np.linalg.qr(X)
    z = X.T @ y
    
    # fit full data with optimal s2
    b, d, Pi = bhat(s2_cv, np.ones((p, 1)), R, z, tol = 10**-6)
    b = b * (Pi > 0.05)   # simple Pi threshold to set zeros
    return b, d, s2_cv, Pi, tloss_grid, s2_grid


def bhat(s2, d, R, z, tol):
    nu, p = R.shape
    max_iter = 2500
    for itrn in range(max_iter):
        # QR decomposition
        Rd = R * np.sqrt(d.T)   # make sure d is a col vector before this
        Q1,_ = np.linalg.qr(np.vstack((Rd.T, np.sqrt(s2) * np.eye(nu))))   # Don't need R matrix
        Q2 = Q1[p:, :]
        Q1 = Q1[:p, :]
        
        Pi = np.sum(Q1 * Q1, axis = 1).reshape(p, 1)
        Pz = Q1 @ (Q2.T @ z) / np.sqrt(s2)
        
        # stop condition check
        grad = Pi - Pz**2
        KKT = np.sum(grad**2)
        if KKT < tol:
            break
        elif itrn != max_iter:        # update d
            I = (d > 0).reshape(-1)
            d[I] = d[I] * np.abs(Pz[I]) / np.sqrt(Pi[I])

    # final calculation of b
    b = np.sqrt(d) * Pz
    
    #print('iter:', itrn+1, '\ns2:', s2, '\ntLoss:', tLoss, '\nKKT:', KKT)
    return b, d, Pi