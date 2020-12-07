from MMalgCV import MMalgCV
#from MMalgVAL import MMalgVAL
#from MMalgOLD import MMalgOLD

from sklearn import linear_model
from scipy.optimize import minimize_scalar, minimize

import numpy as np
import pandas as pd
from numpy import random
from scipy.linalg import cholesky

import matplotlib.pyplot as plt

# from timeit import default_timer as timer

def Lasso_MSE(alpha, beta, X, y):
    alpha = abs(alpha)
    p = len(beta)
    # print(alpha)
    if alpha == 0:
        fit = linear_model.LinearRegression(fit_intercept = False).fit(X, y)
    else:
        fit = linear_model.Lasso(alpha = alpha, fit_intercept = False).fit(X, y)
    
    #fit = linear_model.Lasso(alpha = alpha, fit_intercept = False).fit(X, y)
    return sum((fit.coef_.reshape(p, 1) - beta)**2).item()


def testing(p, n, sig, rho_vec, beta_nz, repetitions):
    p_zero = p - len(beta_nz)
    beta = np.array(beta_nz + [0]*p_zero).reshape(p, 1)
    
    
    '''-------------------------
           Model Fitting
    -------------------------'''
    
    SE, WSE, C, IC, Correct, Mistakes = [np.zeros((len(rho_vec), repetitions, models)) for i in range(6)]
    # 5 is how many tables, models is a global variable (number of models)
    
    for table_num in range(len(rho_vec)):
        rho = rho_vec[table_num]
        row_S, col_S = np.mgrid[1:(p + 1), 1:(p + 1)]
        S = rho**(abs(row_S - col_S))
    
        for i in range(repetitions):
            random.seed(i)   # start off with the same datasets each time
            # Data generation
            X = random.normal(size = (n, p)) @ cholesky(S)   # could use random.t as well
            eps = sig * random.normal(size = (n, 1))
            y = X @ beta + eps
            
            # start = timer()
            # fitting the KLIMAX models
            b_CV,_,_,_,_,_ = MMalgCV(X, y, 10, sig**2)   # PROVIDING TRUE SIGMA HERE
            # end = timer()
            # print('Iter', i, '- time:', end - start)
            
            # fitting Lasso (no intercept) using 10 fold cv to tune lambda
            lasso_model = linear_model.LassoCV(fit_intercept = False, cv = 10)
            lasso_model.fit(X, y.reshape(-1))
            b_lass = lasso_model.coef_.reshape(p, 1)
            
            # fitting best alpha lasso (check two versions, one looks around cv estimate)
            optimal_bnd = minimize_scalar(lambda x : Lasso_MSE(x, beta, X, y), 
                                         method = 'Bounded', bounds = (0, 1))
            optimal_cv = minimize(lambda x : Lasso_MSE(x, beta, X, y), 
                                    x0 = lasso_model.alpha_, bounds = ((0, 5),))
            print('rho', rho, 'Iter:', i, 
                  '\nCV alpha:      ', lasso_model.alpha_, Lasso_MSE(lasso_model.alpha_, beta, X, y), 
                  '\nalpha bounded: ', optimal_bnd.x, optimal_bnd.fun, 
                  '\nalpha around cv', optimal_cv.x.item(), optimal_cv.fun)
            if optimal_cv.fun < optimal_bnd.fun:
                best_alpha = optimal_cv.x.item()
                print('CHOSE ALPHA AROUND CV')
            else:
                best_alpha = optimal_bnd.x
                print('CHOSE ALPHA WITH NO GUESS')
            best_lasso = linear_model.Lasso(alpha = best_alpha, fit_intercept = False)
            best_lasso.fit(X, y)
            b_bestlass = best_lasso.coef_.reshape(p, 1)
            
            # recording diagnostics - MMalgCv, then old MMalg, then Lasso
            # number of correctly identified 0's
            C[table_num, i, :] = [np.sum((beta == 0) & (b_CV == 0)), 
                                  np.sum((beta == 0) & (b_lass == 0)), 
                                  np.sum((beta == 0) & (b_bestlass == 0))]
            
            # number of incorrectly identified 0's
            IC[table_num, i, :] = [np.sum((beta != 0) & (b_CV == 0)), 
                                   np.sum((beta != 0) & (b_lass == 0)), 
                                   np.sum((beta != 0) & (b_bestlass == 0))]
            
            # Sum square error from true beta
            SE[table_num, i, :] = [np.sum((beta - b_CV)**2),  
                                   np.sum((beta - b_lass)**2), 
                                   np.sum((beta - b_bestlass)**2)]
            
            # Weighted Square error using S matrix = Cov(X)
            WSE[table_num, i, :] = [np.sum((b_CV - beta).T @ S @ (b_CV - beta)), 
                                    np.sum((b_lass - beta).T @ S @ (b_lass - beta)), 
                                    np.sum((b_bestlass - beta).T @ S @ (b_bestlass - beta))]
            
            # whether the correct model identified (in terms of 0 and nonzero coeff)
            Correct[table_num, i, :] = (C[table_num, i, :] == p_zero) & (IC[table_num, i, :] == 0)
            
            # how many mistaken variables (IC + complement of C)
            Mistakes[table_num, i, :] = IC[table_num, i, :] + (p_zero - C[table_num, i, :])
        
    # Outputs are matrices of size: rho x rep x model
    # each table is for diff rho and has rep x model
    # results below are of size rho x model
    prop_correct = np.mean(Correct, axis = 1)
    C_mean = np.mean(C, axis = 1)
    IC_mean = np.mean(IC, axis = 1)
    SE_median = np.median(SE, axis = 1)
    WSE_median = np.median(WSE, axis = 1)
    Mistakes_mean = np.mean(Mistakes, axis = 1)
    
    results = {'Mean C': C_mean, 
               'Proportion correct': prop_correct, 
               'Median SE': SE_median, 
               'Mean IC': IC_mean, 
               'Mean Mistakes': Mistakes_mean, 
               'Median Weighted SE': WSE_median}
    return results, SE, WSE, C, IC, Correct, Mistakes




'''Data Inputs'''
n = 1000
p = 15
sig = 2
repetitions = 100
beta_nz = [8, 6, -5, 1, 2]
# beta_nz = list(range(1, 15))
rho_vec = np.arange(1, 10, 1)/10
# rho_vec = [0.1, 0.5, 0.9]
model_names = ['MMalgCV', 'CV Lasso', 'Best Lasso']

models = len(model_names) 
results, SE, WSE, C, IC, Correct, Mistakes = testing(p, n, sig, rho_vec, beta_nz, repetitions)

'''Plotting of results against rho'''
setting = r'$n$ = {}, $p$ = {}, $\sigma$ = {}, reps = {}'.format(n, p, sig, repetitions)
coeff_string = r'$\beta$ = {}'.format(beta_nz)
fig, axs = plt.subplots(2, 3, dpi = 300, figsize = (10,6))
for i in range(len(results)):
    table = list(results)[i]
    for j in range(models):
        axs.flat[i].plot(rho_vec, results[table][:, j], label = model_names[j])
        axs.flat[i].set_title(table)
        axs.flat[i].set_xlabel(r'$\rho$')
        # axs.flat[i].set_ylim(bottom = -0.1, top = max(results[table][:, j] + 0.1)
        # axs.flat[i].set_xlim(left = 0)
plt.tight_layout()
fig.suptitle(setting + '\n' + coeff_string, y = 1.1, fontsize = 20)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.03), shadow=True, ncol=3, 
           prop = {'size': 15})

'''EXPORT DATA TO EXCEL'''
with pd.ExcelWriter('Data.xlsx') as writer:
    for table in results:
        df = pd.DataFrame(results[table], index = rho_vec, columns = model_names)
        df.to_excel(writer, sheet_name = table)