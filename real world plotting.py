from MMalgCV import MMalgCV
from MMalgCV import bhat
import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

def comparison(X_raw, y, data_set_name):
    n, p = X_raw.shape
    X_ones = np.hstack((np.ones((n, 1)), X_raw))
    
    '''OLS FIT'''
    ols = linear_model.LinearRegression()
    ols.fit(X_raw, y)
    # b_ols = np.vstack((ols.intercept_, ols.coef_.reshape(-1, 1)))
    s2_ols = np.sum((y - X_raw @ ols.coef_.reshape(-1, 1) - ols.intercept_)**2)/(n-(p+1))
    
    
    '''REPEATED SPLITTING TESTS'''

    error_klimax = np.zeros(repetitions)
    # error_klimax_ols = np.zeros(repetitions)   # using constant OLS noise estimate
    error_lasso = np.zeros(repetitions)
    
    for i in range(repetitions):
        print('Iter', i)
        np.random.seed(i)
        X_train, X_test, y_train, y_test = train_test_split(X_ones, y)
        
        b_klimax, d, s2_cv, Pi, tloss_grid, s2_grid = MMalgCV(X_train, y_train, 5, s2_ols)
        print('s2_ols:', s2_ols, '     s2_cv:', s2_cv)
        # plt.plot(s2_grid, np.sum(tloss_grid, axis = 1)/n)
        
        # Q, R = np.linalg.qr(X_train)
        # z = Q.T @ y_train
        # b_klimax_ols, d, Pi = bhat(s2_ols, np.ones((p+1,1)), R, z, 10**-5)
        
        lasso = linear_model.LassoCV(cv = 5)
        lasso.fit(X_train[:, 1:], y_train.reshape(-1))
        b_lasso = np.vstack((lasso.intercept_, lasso.coef_.reshape(-1, 1)))
        
        #test error diagnostics
        error_klimax[i] = np.sum((y_test - X_test @ b_klimax)**2)
        # error_klimax_ols[i] = np.sum((y_test - X_test @ b_klimax_ols)**2)
        error_lasso[i] = np.sum((y_test - X_test @ b_lasso)**2)
    
    n_test = len(y_test)
    error_data = [error_klimax/n_test, error_lasso/n_test]
    
    fig, ax = plt.subplots(1, 1, dpi = 400, figsize = (6, 10))
    ax.boxplot(error_data, widths = 0.8)
    ax.set_xticklabels(['KLIMAX', 'Lasso'])
    fig.suptitle('{} data set with {} splittings'.format(data_set_name, repetitions))
    ax.set_ylabel('Test Error')
    
    return error_data



'''INPUTS START FROM HERE'''
repetitions = 100
data_sets = ['Abalone', 'Diabetes', 'Auto MPG', 'Boston Housing']

'''Abalone'''
data = pd.read_csv('abalone.txt', sep = ',', header = None)
data.columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'shucked weight',
                'Viscera weight', 'Shell weight', 'Rings']
data['M'] = data['Sex'] == 'M'
data['F'] = data['Sex'] == 'F'
del data['Sex']
y_abalone = data.Rings.values.reshape(-1, 1)
del data['Rings']
X_raw_abalone = data.values.astype(np.float)

error_data_abalone = comparison(X_raw_abalone, y_abalone, data_sets[0])


'''Diabetes'''
diabetes = load_diabetes()
X_raw_diabetes = diabetes['data']
y_diabetes = diabetes['target'].reshape(-1, 1)

error_data_diabetes = comparison(X_raw_diabetes, y_diabetes, data_sets[1])


'''Auto MPG'''
data = pd.read_csv('cars.csv', sep = ',', header = 0)
y_auto = data.MPG.values.reshape(-1, 1)
del data['MPG']
X_raw_auto = data.values.astype(np.float)

error_data_auto = comparison(X_raw_auto, y_auto, data_sets[2])


'''Boston Housing'''
data = pd.read_csv('BostonHousing.csv', sep = ',', header = 0)
y_boston = data.medv.values.reshape(-1, 1)
del data['medv']
X_raw_boston = data.values.astype(np.float)

error_data_boston = comparison(X_raw_boston, y_boston, data_sets[3])


'''FINAL PLOTTING'''

error_dict = {'Abalone': error_data_abalone, 
              'Diabetes': error_data_diabetes, 
              'Auto MPG': error_data_auto, 
              'Boston Housing': error_data_boston}

colors = ['#5ab5f2', '#ffa64d']
fig, axs = plt.subplots(2, 2, dpi = 400, figsize = (6, 8))
for i in range(len(data_sets)):
    name = data_sets[i]
    error = error_dict[name]
    tmp = axs.flat[i].boxplot(error, widths = 0.75, patch_artist = True, medianprops = dict(color = 'black'))
    axs.flat[i].set_xticklabels(['KLIMAX', 'Lasso'])
    axs.flat[i].set_title(name)
    axs.flat[i].set_ylabel('Test Error')
    axs.flat[i].set_facecolor('#f2f2f2')
    for patch, color in zip(tmp['boxes'], colors):
        patch.set_facecolor(color)
plt.tight_layout()
