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


element = 'O'
natom = AtomicData[element]['nstates'] - 1
te0 = 2.0e+6
ne0 = 1.0e+7


#------------------------------------------------------------------------------
# Tests
#------------------------------------------------------------------------------
# Initial conditions, set plasma temperature, density and dt
print('************************************')
print('TEST: chemical element =', element)
te0 = 1.0e+6
te_ini = 4.0e+4
ne0 = 1.0e+7

# Start from any ionizaiont states, e.g., Te = 4.0d4 K,
time = 0
ind = neipy.core.func_index_te(te_ini, AtomicData['temperatures'])
f0 = AtomicData[element]['equistate'][ind,:]
ChargeStateOverTime = neipy.create_ChargeStates_dictionary([element])
ChargeStateOverTime['He'] = f0

print('START-------------------------------')
print('time_sta = ', time)
print(f0)
print('Sum(f0) = ', np.sum(f0))

# After time + dt:
dt = 1.0e+2
#ft = neipy.core.func_solver_eigenval(element, AtomicData, te0, ne0, dt, f0)

i = 0
while (i < 20):
   #print('i = ', i)
   ft = neipy.core.func_solver_eigenval(element, AtomicData, te0, ne0, dt, f0)
   ChargeStateOverTime[element]=np.vstack((ChargeStateOverTime[element], ft))
   f0 = ft
   i = i + 1

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
   plt.plot(ionistate, ChargeStateOverTime[element][i, :]) 
   #print(ChargeStateOverTime[element][i, :])
   i = i + 1
plt.ylabel('Ion fraction')
plt.yscale('log')
plt.ylim(1.0e-06,1.0)
plt.xlabel('Ionization states')
plt.show()





