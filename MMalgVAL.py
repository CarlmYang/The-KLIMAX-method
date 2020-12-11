'''NOTE: added Pi > 0.05 threshold at the end'''
'''Changes made:
    - Added split function to make training and test set
    - Added 'fixed' boolean input to determine fixed point or not in bhat
    - Changed variable names from X1 X2 to X_train and X_test etc
'''

import numpy as np

def MMalgVAL(X, y):
    n, p = X.shape
    
    # splitting into training and test set
    X_train, X_test, y_train, y_test = split(X, y, 0.8)
    n_test = len(y_test)    # size of test set
    
    # QR decomposition, overwrite X with Q
    X_train, R_train = np.linalg.qr(X_train)
    X_test, R_test = np.linalg.qr(X_test)
    z_train = X_train.T @ y_train
    z_test = X_test.T @ y_test
    yqz_train = np.sum((y_train - X_train @ z_train)**2)  # excess SSR
    yqz_test = np.sum((y_test - X_test @ z_test)**2)
    
    # Find fixed point for s2 (initialise with 1)
    s2fp, d,_,_ = bhat(1, np.ones((p, 1)), R_train, z_train, R_test, z_test, 
                     yqz_test, n_test, 10**-3, True)
    
    # get a plot over a grid of s2 values
    ngrid = 40
    s2_grid = np.logspace(np.log10(s2fp/100), np.log10(s2fp*10), ngrid)
    b_grid = np.full((p, ngrid), np.nan)
    tloss_grid = np.full(ngrid, np.nan)
    dg = np.ones((p, 1))# temporary delta vector resused from iterations to speed up things
    # we are using the previous iteration's d here, maybe test a full reset each time?
    for i in range(ngrid):
        lossg, dg, bg,_ = bhat(s2_grid[i], dg, R_train, z_train, R_test, z_test, 
                               yqz_test, n_test, 10**-5, False)
        b_grid[:, i] = bg.reshape(-1)   # gotta turn into basic array
        tloss_grid[i] = lossg
    
    # get the full data
    R_full = np.vstack((R_train, R_test))
    z_full = np.vstack((z_train, z_test))
    
    # make a fit with the optimal s2 based on grid search and the full data
    _, d, b, Pi = bhat(s2_grid[tloss_grid.argmin()], np.ones((p, 1)), R_full, z_full, 
                       R_full, z_full, yqz_train + yqz_test, n, 10**-6, False)
    b = b * (Pi > 0.05)   # simple Pi threshold
    return b, d, s2fp, Pi, tloss_grid, b_grid, s2_grid

def split(X, y, pctg):   # gives a shuffled split of the data (THERE IS RANDOMISATION HERE)
    '''pctg = percentage of the data that is training set'''
    n = len(y)
    split = int(np.ceil(pctg*n))
    perm = np.random.permutation(n)
    train = perm[:split]
    test = perm[split:]
    return X[train,:], X[test,:], y[train], y[test]

def bhat(s2, d, R, z, Rt, zt, yqzt, nt, eps, fixed = False):
    ''' s2, d, R, z, eps are required to run the iterative process (training data)
        Rt, zt, yqzt, nt are required to compute test loss and used for fixed point calculation'''
    nu, p = R.shape
    for itrn in range(2600):
        # QR decomposition
        Rd = R * np.sqrt(d.T)   # make sure d is a col vector before this
        Q1,_ = np.linalg.qr(np.vstack((Rd.T, np.sqrt(s2) * np.eye(nu))))   # Don't need R matrix
        Q2 = Q1[p:, :]
        Q1 = Q1[:p, :]
        
        Pi = np.sum(Q1 * Q1, axis = 1).reshape(p, 1)
        Pz = Q1 @ (Q2.T @ z) / np.sqrt(s2)
        I = (d > 0).reshape(-1)
        d[I] = d[I] * abs(Pz[I]) / np.sqrt(Pi[I])
        
        # This is run if looking for a fixed point solution
        if fixed:
            b = np.sqrt(d) * Pz
            tLoss = (yqzt + np.sum((zt - Rt @ b)**2)) / nt   # computes test loss (use this as don't have original data)
            s2 = tLoss
        
        # stop condition check
        grad = Pi - Pz**2
        KKT = np.sum(grad**2)
        if KKT < eps: break

    # final calculation of b and test loss
    b = np.sqrt(d) * Pz
    tLoss = (yqzt + np.sum((zt - Rt @ b)**2)) / nt
    
    #print('iter:', itrn+1, '\ns2:', s2, '\ntLoss:', tLoss, '\nKKT:', KKT)
    return tLoss, d, b, Pi