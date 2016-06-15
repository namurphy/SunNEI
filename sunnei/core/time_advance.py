"""
NEI
===
This module contains the core non-equilibrium ionization routines.
"""

from __future__ import print_function
import numpy as np
import pandas as pd

# This pandas series allows a shortcut to finding the atomic number of
# an element.  For example, AtomicNumbers['Fe'] will return 26.

AtomicNumbers = pd.Series(np.arange(28)+1,
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
# Update:
#   2016-05-06, Chengcai
#   Add the loop over all elements in the list 'elements_arr';
#   Add the dt_te estimation.
#
def func_dt_eigenval(elements_arr, AtomicData, te_list, ne_list, dt_in, 
                     change_perct=1.0e-3,
                     safety_factor=0.40,
                     dt_ne=1.0e5,
                     dt_te=1.0e5,):
    
    ind_0 = func_index_te(te_list[0], AtomicData['temperatures'])
    ind_1 = func_index_te(te_list[1], AtomicData['temperatures'])
    ind_list = [ind_0, ind_1]

    # estimate (or re-check) dt_te using temperature
    ind_dte = np.absolute(ind_1 - ind_0)
    if ind_dte == 0:
        dt_te = dt_in
    else:
        dt_te = safety_factor*dt_in/(ind_dte+1.0)
    
    # estimate dt_ne using temperature
    for element in elements_arr:

        for itime in [0,1]: #find eigenvalues on the chosen Te node
            eval_arr = AtomicData[element]['eigenvalues'][ind_list[itime],:]

            eval_max = max(abs(eval_arr))
            dt_est = change_perct/(eval_max*ne_list[itime])
            if dt_est <= dt_ne:
                dt_ne = dt_est

    # return dt_out
    dt_out = dt_te
    if dt_ne >= dt_te:
        dt_out = dt_ne
    return dt_out

#------------------------------------------------------------------------------
# function: Time-Advance: solver
#------------------------------------------------------------------------------
# Update
#   2016-05-03
#   Replace input parameter 'natom' by 'element'. 
#   Add a argument 'AtomicData',  which is a dictionary, created by 
#   using sunnei.read_atomic_data.
#   2016-05-17
#   Replace input parameter 'element' by 'elements'.
#   Add a loop to over a element list.
#   f0_dic and ft_dic are charge_state dictionary defined using function 
#   sunnei.create_ChargeStates_dictionary.
#
def func_solver_eigenval(elements, AtomicData, te, ne, dt, f0_dic):
    
    # copy a dictionary to save the advanced charge state. 
    ft_dic = f0_dic.copy()

    for element in elements:
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
    
        # eigenvector_inverse
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
        f0 = f0_dic[element]
        ft = np.dot(f0, matrix_2)

        # re-check the smallest value
        minconce = 1.0e-15
        for ii in np.arange(0, natom+1, dtype=np.int):
            if ((ft[ii]) <= minconce):
                ft[ii] = 0.0
        ft_dic[element] = ft        
    return ft_dic
