## functions_eigenvals_math.py
import numpy as np
from scipy.io import FortranFile

#------------------------------------------------------------------------------
# open file and read
#------------------------------------------------------------------------------
# Parameters
path_eigentb = '/Users/ccai/Works/Project/Ionization_calc/Code_develop/\
ionization_code_reu2016/python_script/chianti_8/'
element = 'o'
file_name = element+'eigen.dat'

# Open file
file_eigentb = path_eigentb + file_name
f = FortranFile(file_eigentb, 'r')

# Read file
[nte, natom]=f.read_ints(dtype=np.int32)
te_arr = f.read_reals(dtype=np.float64)
eqistate = f.read_reals(dtype=np.float64).reshape((natom+1, nte), order='F')
eigenvals = f.read_reals(dtype=np.float64).reshape((natom+1, nte), order='F')
eigenvector = f.read_reals(dtype=np.float64).reshape((natom+1, natom+1, nte), order='F')
eigenvector_invers = f.read_reals(dtype=np.float64).reshape((natom+1, natom+1, nte), order='F')

# Close file
f.close()


# Note from Nick: I copied the next three routines to time_advance.py
# but am leaving these also here for now.


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

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
# Initial conditions, set plasma temperature, density and dt
print '************************************'
print 'TEST: chemical element =', element
te0 = 2.0e+6
ne0 = 1.0e+7

# Start from any ionizaiont states, e.g., Te = 4.0d4 K,
time = 0
ind = func_index_te(4.0e+4, te_arr)
f0 = eqistate[:, ind]

print 'START-------------------------------'
print 'time_sta = ', time
print f0
print 'Sum(f0) = ', np.sum(f0)

# After time + dt:
dt = 1.0e+6
ft = func_solver_eigenval(natom, te0, ne0, time+dt, f0)

print 'END---------------------------------'
print 'time_end = ', time+dt
print ft
print 'Sum(ft) = ', np.sum(ft)

print 'EQI---------------------------------'
ind = func_index_te(te0, te_arr)
print eqistate[:, ind]
print 'Sum(feqi) = ', np.sum(eqistate[:, ind])
print '************************************'
