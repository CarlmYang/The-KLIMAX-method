'''Script for plotting the data outputted from 'Performance Tests.py' if it has been exported'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''MANUAL INPUTS'''
file_name = 'non_sparse.xlsx'
n = 1000
p = 15
sig = 5
# beta_nz = [8, 6, -5, 1, 2]
beta_nz = list(range(1, 15))
repetitions = 100

tmp = pd.read_excel(file_name, sheet_name = 'Mean C', index_col = 0)
rho_vec = tmp.index.to_numpy()
model_names = list(tmp.columns)
model_names[0] = 'KLIMAX'
models = len(model_names)

C_mean = pd.read_excel(file_name, sheet_name = 'Mean C', index_col = 0).to_numpy()
SE_median = pd.read_excel(file_name, sheet_name = 'Median SE', index_col = 0).to_numpy()
IC_mean = pd.read_excel(file_name, sheet_name = 'Mean IC', index_col = 0).to_numpy()
WSE_median = pd.read_excel(file_name, sheet_name = 'Median Weighted SE', index_col = 0).to_numpy()
prop_correct = pd.read_excel(file_name, sheet_name = 'Proportion correct', index_col = 0).to_numpy()
Mistakes_mean = pd.read_excel(file_name, sheet_name = 'Mean Mistakes', index_col = 0).to_numpy()

# Horizontal plots layout (presentation)
results = {'Mean C': C_mean, 
            'Proportion correct': prop_correct, 
            'Median SE': SE_median, 
            'Mean IC': IC_mean, 
            'Mean Mistakes': Mistakes_mean, 
            'Median Weighted SE': WSE_median}

#Vertical plots layout (thesis)
# results = {'Mean C': C_mean, 
#            'Mean IC': IC_mean, 
#            'Proportion correct': prop_correct, 
#            'Mean Mistakes': Mistakes_mean, 
#            'Median SE': SE_median, 
#            'Median Weighted SE': WSE_median}
'''-------------------
   Plotting options
-------------------'''


'''Plotting of results against rho (horizontal)'''
setting = r'$n$ = {}, $p$ = {}, $\sigma$ = {}, reps = {}'.format(n, p, sig, repetitions)
coeff_string = r'$\beta$ = {}'.format(beta_nz)
fig, axs = plt.subplots(2, 3, dpi = 400, figsize = (8.5, 4.5))
for i in range(len(results)):
    table = list(results)[i]
    for j in range(models):
        axs.flat[i].plot(rho_vec, results[table][:, j], label = model_names[j])
        axs.flat[i].set_title(table, fontsize = 10)
        axs.flat[i].set_xlabel(r'$\rho$')
        # axs.flat[i].set_ylim(bottom = -0.1, top = max(results[table][:, j] + 0.1)
        # axs.flat[i].set_xlim(left = 0)
plt.tight_layout()
fig.suptitle(setting + '\n' + coeff_string, y = 1.08, fontsize = 12)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), shadow=True, ncol=3, 
            prop = {'size': 10})

# '''Plotting of results against rho (vertical)'''
# setting = r'$n$ = {}, $p$ = {}, $\sigma$ = {}, reps = {}'.format(n, p, sig, repetitions)
# coeff_string = r'$\beta$ = {}'.format(beta_nz)
# fig, axs = plt.subplots(3, 2, dpi = 300, figsize = (6, 8))
# for i in range(len(results)):
#     table = list(results)[i]
#     for j in range(models):
#         axs.flat[i].plot(rho_vec, results[table][:, j], label = model_names[j])
#         axs.flat[i].set_title(table)
#         axs.flat[i].set_xlabel(r'$\rho$')
#         # axs.flat[i].set_ylim(bottom = -0.1, top = max(results[table][:, j] + 0.1)
#         # axs.flat[i].set_xlim(left = 0)
# plt.tight_layout()
# fig.suptitle(setting + '\n' + coeff_string, y = 1.05, fontsize = 12)
# handles, labels = axs[0, 0].get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.02), shadow=True, ncol=3, 
#            prop = {'size': 10})