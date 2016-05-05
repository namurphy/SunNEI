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
# Update
#   2016-05-03, by Chengcai.
#   Replace input parameter 'natom' by 'element'. 
#   Add a argument 'AtomicData',  which is a panda structure, created by 
#   using neipy.read_atomic_data.
#
def func_solver_eigenval(element, AtomicData, te, ne, dt, f0):

    nstates = AtomicData[element]['nstates']
    natom = nstates - 1
    
    #ind = func_index_te(te, te_arr) # index of node in temperature table
    ind = func_index_te(te, AtomicData['temperatures'])

    #find eigenvalues on the chosen Te node
    evals = AtomicData[element]['eigenvalues'][ind,:]
    
    # eigen vectors
    evect_0 = AtomicData[element]['eigenvector'][ind,:,:]
    evect_1 = np.reshape(evect_0, (nstates*nstates))
    evect = np.reshape(evect_1, (nstates, nstates), order='F')
    
    # eigenvector_invers
    evect_inv_0 = AtomicData[element]['eigenvector_inv'][ind,:,:]
    evect_inv_1 = np.reshape(evect_inv_0, (nstates*nstates))
    evect_inv = np.reshape(evect_inv_1, (nstates, nstates), order='F')

    # define the temperary diagonal matrix
    diagona_evals = np.zeros((nstates, nstates))
    for ii in np.arange(0, natom+1, dtype=np.int):
        diagona_evals[ii,ii] = np.exp(evals[ii]*dt*ne)

    # matirx operation
    matrix_1 = np.dot(diagona_evals, evect)
    matrix_2 = np.dot(evect_inv, matrix_1)

    # get ions fraction at (time+dt)
    ft = np.dot(f0, matrix_2)

    # re-check the smallest value
    minconce = 1.0e-15
    for ii in np.arange(0, natom+1, dtype=np.int):
        if (abs(ft[ii]) <= minconce):
            ft[ii] = 0.0
    return ft
