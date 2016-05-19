"""
CMEheat
=====
This module contains non-equilibrium ionization routines to
investigate the heating of coronal mass ejection (CME) plasma.
"""

import numpy as np
import pandas as pd

from neipy.core import func_index_te, func_dt_eigenval, func_solver_eigenval
from neipy.core import read_atomic_data, create_ChargeStates_dictionary

# Definining constants

RSun = 6.957e5 # km
gamma = 5.0/3.0
gamm1 = gamma-1.0

# This pandas series allows a shortcut to finding the atomic number of
# an element.  For example, AtomicNumbers['Fe'] will return 26.

AtomicNumbers = pd.Series(np.arange(28)+1,
                          index=['H' ,'He',
                                 'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                                 'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',
                                 'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni',
                                 ])


def cmeheat_track_plasma(
    initial_height       = 0.05,    # in solar radii
    final_height         = 6.0 ,    # height to output charge states
    log_initial_temp     = 6.0,     # K
    log_initial_dens     = 9.6,     # number density in cm^-3
    vfinal               = 750.0,   # km/s
    vscaletime           = 2400.0,  # s
    ExpansionExponent    = -2.5,    # dimensionless
    max_steps            = 50,      # maximum number of steps
    timestep             = 200.0,   # s
    elements = ['H', 'He', 'C',     # elements to be modeled
                'N', 'O', 'Ne',
                'Mg', 'Si', 'S', 
                'Ar', 'Ca', 'Fe', ],
    screen_output=True,
    ):
    
    '''
    The main program for tracking the ionization states of a blob of
    plasma as it is moving away from the low corona.  
    '''

    # Check to make sure that realistic values for the inputs are
    # being used.  Suggest changes, if necessary.
 
    assert initial_height >= 0.01 and initial_height <= 0.5, \
        'Choose an initial height between 0.01 and 0.5 RSun (usually 0.05 to 0.1 is best)'
    assert vfinal >= 50.0 and vfinal <= 5000.0, \
        'Need vfinal between 50.0 and 5000.0 km/s (usually 250 to 2500 km/s is best)'
    assert log_initial_temp >= 3.8 and log_initial_temp <= 8.0, \
        'Need log_initial_temp between 3.8 and 8.0 (usually 4.5 to 7.0 is best)'
    assert max_steps >= 1, 'Need max_steps >= 1'
    assert elements.__contains__('H'), 'The elements list must include H'
    assert elements.__contains__('He'), 'The elements list must include He'

#    ExpansionExponent = (log_final_dens-log_initial_dens)/(np.log10(height_of_final_dens)-np.log10(initial_height))

    assert ExpansionExponent>=-4.0 and ExpansionExponent<=-0.9, \
        'Need ExpansionExponent between -4 and -0.9 (usually between -3.5 and -1.5)'

    # Make sure output heights are monotonically increasing with no
    # duplicate values

#    heights = np.array(heights, dtype=np.float64)
#    heights = np.unique(heights)

#    assert np.min(heights) > initial_height, \
#        'Need min(heights) > initial_height  [note: units of RSun]'

    # Initialize arrays

    time = np.zeros(max_steps+1)        # seconds
    height = np.zeros(max_steps+1)      # units of RSun
    velocity = np.zeros(max_steps+1)    # km/s
    density = np.zeros(max_steps+1)     # cm**-3
    temperature = np.zeros(max_steps+1) # K

    height[0] = initial_height 
    density[0] = 10**log_initial_dens
    temperature[0] = 10**log_initial_temp

    # Read in the atomic data needed for the non-equilibrium
    # ionization modeling.  Use it to set up the initial charge states
    # which assume ionization equilibrium.

    AtomicData = read_atomic_data(elements, screen_output=False)
    InitialChargeStates = create_ChargeStates_dictionary(elements,temperature[0],AtomicData)
    ChargeStateList = [InitialChargeStates]

    # The main time loop

    i = 1
    while (i <= max_steps):
        
        # Determine the time step

#        timestep = 5.0  # second, need to update this!!!!!!!!!!!

#        timestep = func_dt_eigenval(elements, AtomicData, te_list, ne_list, dt_in, 
#                                    change_perct=1.0e-3,
#                                    safety_factor=0.40,)
#                         dt_ne=1.0e5,
#                         dt_te=1.0e5,)
        time[i] = time[i-1] + timestep

        # Determine the velocity and height

        velocity[i] = vfinal*(1.0 - np.exp(-time[i]/vscaletime))
        height[i] = initial_height + vfinal*(time[i] + vscaletime*(np.exp(-time[i]/vscaletime) - 1.0))/RSun

        # Density is a power law with height
        #  - Still need to account for electron density
        
        density[i] = density[0] * (height[i] / height[0])**ExpansionExponent

        # The temperature evolution currently includes adiabatic cooling.
        # We may later wish to include additional terms such as
        # radiative cooling, energy stored in ionization, and additional heating

        temperature[i] = temperature[i-1]*(density[i]/density[i-1])**gamm1

        # Ionization time advance
        
        mean_temperature = 0.5*(temperature[i]+temperature[i-1])
        mean_density = 0.5*(density[i]+density[i-1])

        NewChargeStates = func_solver_eigenval(elements, AtomicData, mean_temperature, mean_density, timestep, ChargeStateList[i-1])
        ChargeStateList.append(NewChargeStates.copy())
        
        i = i + 1

    # Create an object to store the inputs and outputs
 
    inputs = pd.Series([
                        initial_height,
                        log_initial_temp,
                        log_initial_dens, 
                        vfinal,
                        vscaletime,
                        ExpansionExponent, 
#                        heights
                        ],
                       index=[
                              'initial_height',
                              'log_initial_temp',
                              'log_initial_dens',
                              'vfinal',
                              'vscaletime',
                              'ExpansionExponent', 
#                              'heights',
                              ])

    if screen_output:
        
        # Inputs

#        Initial height = 0.30 RSun with log(n)=10.20 and log(T) = 7.00
#        Final height = 3.00 RSun with log(n)=6.22
#        

        print('Density is given by a power law with index {0:<7.3f}'.format(ExpansionExponent))

        if ExpansionExponent>-1.0 or ExpansionExponent<-3.5:
            print('Cautionary note: power law index should be between -3.5 and -1.0')

        print()

        # Initial height & dens; Final height & dens ; ExpansionExponent
        # Initial temperature
        # vfinal, vscaletime
        

        # Outputs
        
        if max_steps <= 10 :
            output_interval = 1
        elif max_steps <= 20: 
            output_interval = 2
        elif max_steps <= 60:
            output_interval = 5
        elif max_steps <= 240: 
            output_interval = 20
        elif max_steps <= 1000:
            output_interval = 50
        elif max_steps > 1000:
            output_interval = 100
            
        # Output information on height, velocity, and thermodynamics

        for i in range(max_steps+1):
            if (np.mod(i,output_interval) == 0) or (i == max_steps):
                print(('i={0:4d}  t={1:>8.2f}  V={2:>6.1f}  h={3:>6.3f}  log(n)={4:>5.2f}'+\
                           '  log(T)={5:>5.2f}').format(i, time[i], velocity[i], 
                                                        height[i], np.log10(density[i]), 
                                                        np.log10(temperature[i])))
        print()

        np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

        for element in elements:

            if element == 'H' or element=='He' or element=='C' or \
               element == 'N' or element=='O'  or element=='Si' or \
               element =='Fe':

                print('Initial and final charge states for '+element)
                print()
                print(ChargeStateList[0][element])
                if AtomicNumbers[element] >= 10:
                    print()
                print(ChargeStateList[max_steps][element])
                print()

    return ChargeStateList
