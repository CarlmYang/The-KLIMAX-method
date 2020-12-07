from MMalgCV import MMalgCV
from MMalgVAL import MMalgVAL

import numpy as np
from numpy import random
from scipy.linalg import cholesky

# from sklearn.model_selection import KFold
# from sklearn import linear_model

import matplotlib.pyplot as plt

'''Data Inputs'''

p = 10
n = 750
sig = 2
beta_nz = [8, 6, -5, 1, 2]
beta = np.array(beta_nz + [0]*(p-len(beta_nz))).reshape(p, 1)
rho = 0.9

row_S, col_S = np.mgrid[1:(p + 1), 1:(p + 1)]
S = rho**(abs(row_S - col_S))

random.seed(42)
'''normal distribution'''
X = random.normal(size = (n, p)) @ cholesky(S)
eps = sig * random.normal(size = (n, 1))
y = X @ beta + eps


'''Single run testing'''
# b, d, s2_cv, Pi, tloss_grid, s2_grid = MMalgCV(X, y, 10, sig**2)

# plt.figure(dpi = 400)
# plt.plot(s2_grid, np.sum(tloss_grid, axis = 1), 'ro', ms = 2)
# plt.axvline(sig**2, linewidth = 0.5)



'''--------------------------------------------------
   Multiple runs - Cross Validation
--------------------------------------------------'''
setting = r'$n$ = {}, $p$ = {}, $\sigma$ = {}, $\rho$ = {}'.format(n, p, sig, rho)
coeff_string = r'$\beta$ = {}'.format(beta_nz)
repetitions = 100
s2min = np.zeros(repetitions)
tlossmin = np.zeros(repetitions)
plt.figure(dpi = 400)
for rep in range(repetitions):
    b, d, s2_cv, Pi, tloss_grid, s2_grid = MMalgCV(X, y, 10, sig**2)
    s2min[rep] = s2_cv
    tlossmin[rep] = min(np.sum(tloss_grid, axis = 1)/n) 
    plt.plot(s2_grid, np.sum(tloss_grid, axis = 1)/n, linewidth = 0.3)
    print('Iter', rep+1, 'done')
# overlay with s2min
plt.plot(s2min, tlossmin, 'ro', ms = 2)
plt.axvline(sig**2, linewidth = 0.5)
plt.title(setting + '\n' + coeff_string)
plt.xlabel(r'$\sigma^2$')
plt.ylabel('CV Loss')

# plt.figure(dpi = 400)
# plt.plot(s2min, 'ro', ms = 2)
# plt.axhline(min(s2_grid), linewidth = 0.5)

'''--------------------------------------------------
   Multiple runs - Validation Set
--------------------------------------------------'''
# setting = r'$n$ = {}, $p$ = {}, $\sigma$ = {}, $\rho$ = {}'.format(n, p, sig, rho)
# coeff_string = r'$\beta$ = {}'.format(beta_nz)
# repetitions = 1
# s2min = np.zeros(repetitions)
# tlossmin = np.zeros(repetitions)
# s2f = np.zeros(repetitions)
# plt.figure(dpi = 400)
# for rep in range(repetitions):
#     b, d, s2fp, Pi, tloss_grid, b_grid, s2_grid = MMalgVAL(X, y)
#     s2f[rep] = s2fp
#     s2min[rep] = s2_grid[tloss_grid.argmin()]
#     tlossmin[rep] = min(tloss_grid)
#     plt.plot(s2_grid, tloss_grid, linewidth = 0.3)
#     print('Iter', rep+1, 'done')
# # overlay with s2min
# plt.plot(s2min, tlossmin, 'ro', ms = 2)
# plt.axvline(sig**2, linewidth = 0.5)
# plt.title(setting + '\n' + coeff_string)
# plt.xlabel(r'$\sigma^2$')
# plt.ylabel('Test Loss')
