# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 21:23:29 2016

@author: Chengcai

Purpose: test
"""

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
import numpy as np
import neipy
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------
# read AtomicData
#------------------------------------------------------------------------------
AtomicData = neipy.read_atomic_data(screen_output=True)

# Define the plasma parameters
element = 'O'
natom = AtomicData[element]['nstates'] - 1
te_ini = 4.0e+4
te0 = 2.0e+6
ne0 = 1.0e+7
te_list = [te0, te0]
ne_list = [ne0, ne0*0.9]

# Create a dictionary to save results
ChargeStateDic = neipy.create_ChargeStates_dictionary(['He', 'O', 'Fe'])
#ChargeStateTime = [ChargeStateDic]

#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
# Initial conditions, set plasma temperature, density and dt
print('************************************')
print('TEST: chemical element =', element)

# Start from any ionizaiont states, e.g., Te = 4.0d4 K,
time = 0
ind = neipy.core.func_index_te(te_ini, AtomicData['temperatures'])
f0 = AtomicData[element]['equistate'][ind,:]
ChargeStateDic[element] = f0
#ChargeStateTime[0][element][:] = f0[:]

print('START-------------------------------')
print('time_sta = ', time)
print(f0)
print('Sum(f0) = ', np.sum(f0))

# After time + dt:
dt_in = 5.0e+3

i = 0
while (i < 20):
    i = i+1
    
    dt = neipy.core.func_dt_eigenval(element, AtomicData, te_list, ne_list, dt_in)

    nec = 0.5*(ne_list[0] + ne_list[1])
    ft = neipy.core.func_solver_eigenval(element, AtomicData, te0, nec, dt, f0)
    
    # for the next step
    ne_list[0] = ne_list[1]
    ne_list[1] = ne_list[0]*0.9
        
    f0 = ft

    # save ractions 
    ChargeStateDic[element]=np.vstack((ChargeStateDic[element], ft))
    #ChargeStateTime.append(ChargeStateDic)
    #ChargeStateTime[i][element][:] = ft[:]

print('END---------------------------------')
print('time_end = ', time+i*dt)
print(ft)
print('Sum(ft) = ', np.sum(ft))

print('EQI---------------------------------')
ind = neipy.core.func_index_te(te0, AtomicData['temperatures'])
print(AtomicData[element]['equistate'][ind,:])
print('************************************')

#------------------------------------------------------------------------------
# plot
#------------------------------------------------------------------------------
ionistate = np.arange(1, natom+1+1, 1)
i = 0
while (i < 20):
    plt.plot(ionistate, ChargeStateDic[element][i, :]) 
    i = i + 1
plt.ylabel('Ion fraction')
plt.yscale('log')
plt.ylim(1.0e-06,1.0)
plt.xlabel('Ionization states')
#plt.show()
plt.savefig('test_core_timeadvance.eps', format='eps', dpi=120)
print("Test: normal stop!")





