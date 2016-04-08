"""
NEI
===
This module contains the core non-equilibrium ionization routines to
model astrophysical plasmas.
"""

import numpy as np
import pandas as pd

# This pandas series allows a shortcut to finding the atomic number of
# an element.  For example, all_elements['Fe'] will return 26.

all_elements = pd.Series(np.arange(28)+1,
                         index=['H' ,'He',
                                'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                                'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',
                                'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni',
                                ])

#------------------------------------------------------------------------------
# function: Find te node on the Te table
#------------------------------------------------------------------------------
def func_index_te(te, te_arr):
    res = np.where(te_arr >= te) # Assuming te_arr is a monotonic array
    res_ind = res[0]
    ind =  res_ind[0]
    dte_l = abs(te-te_arr[ind-1]) # re-check the neighbor point
    dte_r = abs(te-te_arr[ind])
    if (dte_l <= dte_r):
        ind = ind - 1
    return ind

#------------------------------------------------------------------------------
# function: Time-step estimate function
#------------------------------------------------------------------------------
def func_dt_eigenval(te, ne):
    change_perct = 1.0e-3
    ind = func_index_te(te, te_arr) # index of node in temperature table
    eval_arr = eigenvals[:,ind]
    eval_max = max(abs(eval_arr))
    dt_est = change_perct/(eval_max*ne)
  
    # to be continue ..
    # need a loop over all elements    
    # ...
    return dt_est # end function time_step_eigenvalue

#------------------------------------------------------------------------------
# function: Time-Advance: solover
#------------------------------------------------------------------------------
def func_solver_eigenval(natom, te, ne, dt, f0):
    ind = func_index_te(te, te_arr) # index of node in temperature table

    evals = eigenvals[:,ind] # find eigenvalues on the chosen Te node
    evect = eigenvector[:,:,ind]
    evect_invers = eigenvector_invers[:,:,ind]

    # define the temperary diagonal matrix
    diagona_evals = np.zeros((natom+1, natom+1))
    for ii in np.arange(0, natom+1, dtype=np.int):
        diagona_evals[ii,ii] = np.exp(evals[ii]*dt*ne)

    # matirx operation
    matrix_1 = np.dot(diagona_evals, evect)
    matrix_2 = np.dot(evect_invers, matrix_1)

    # get ions fraction at (time+dt)
    ft = np.dot(f0, matrix_2)

    # re-check the smallest value
    minconce = 1.0e-15
    for ii in np.arange(0, natom+1, dtype=np.int):
        if (abs(ft[ii]) <= minconce):
            ft[ii] = 0.0
    return ft
