"""
CMEheat
=====
This module contains non-equilibrium ionization routines to
investigate the heating of coronal mass ejection (CME) plasma.
The primary developers are Nick Murphy and Chengcai Shen.
"""

import numpy as np
import pandas as pd

from neipy.core import func_index_te, func_dt_eigenval, func_solver_eigenval
from neipy.core import read_atomic_data, create_ChargeStates_dictionary

# This pandas series allows a shortcut to finding the atomic number of
# an element.  For example, all_elements['Fe'] will return 26.

all_elements = pd.Series(np.arange(28)+1,
                         index=['H' ,'He',
                                'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                                'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',
                                'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni',
                                ])
# Definining constants

RSun = 6.957e5 # km
gamma = 1.666666666666667
gamm1 = 0.666666666666667

def cmeheat_track_plasma(
                 initial_height       = 0.1,     # in solar radii
                 log_initial_temp     = 6.0,     # K
                 log_initial_dens     =12.0,     # number density in cm^-3
                 log_final_dens    = 6.6,     #                 
                 height_of_final_dens = 3.0,     # in solar radii
                 vfinal               = 500.0,   # km/s
                 vscaletime           = 3600.0,  # s
#                 expansion_exponent   = 3.0,     # from 2.0 to 3.0
                 max_steps            = 25,     # maximum number of steps
                 output_heights       = [2.,3.], # heights to output charge states
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

    # Add in assert statements to double check inputs, and suggest
    # characteristic values.  
 
    assert initial_height >= 0.01 and initial_height <= 0.5, \
        'Choose an initial height between 0.01 and 0.5 RSun (usually 0.05 to 0.1 is best)'
    assert initial_height < height_of_final_dens, \
        'Need initial_height < height_of_final_dens'
    assert vfinal >= 50.0 and vfinal <= 5000.0, \
        'Need vfinal between 50.0 and 5000.0 km/s (usually 250 to 2500 km/s is best)'
    assert log_initial_temp >= 3.8 and log_initial_temp <= 8.0, \
        'Need log_initial_temp between 3.8 and 8.0 (usually 4.5 to 7.0 is best)'
    assert log_initial_dens >= 8.0 and log_initial_dens <= 12.0, \
        'Need log_initial_dens between 8.0 and 12.0 (usually 9.0 to 11.0 is best)'
    assert max_steps >= 1, 'Need max_steps >= 1'
    assert elements.__contains__('H'), 'The elements list must include H'
    assert elements.__contains__('He'), 'The elements list must include He'
 
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

    # The density in this model evolves self-similiarly as a power
    # law: n/n0 = (h/h0)^ExpansionExponent

    ExpansionExponent = (log_final_dens-log_initial_dens)/(np.log10(height_of_final_dens)-np.log10(initial_height))

    # The main time loop

    i = 1
    while (i <= max_steps):
        
        # Determine the time step

        timestep = 200.0  # need to update with better values

        time[i] = time[i-1] + timestep

        # Determine evolution of plasma parameters

        velocity[i] = vfinal*(1.0 - np.exp(-time[i]/vscaletime))
        height[i] = initial_height + vfinal*(time[i] + vscaletime*(np.exp(-time[i]/vscaletime) - 1.0))/RSun

        # Density is a power law with height
        #  - Still need to account for electron density
        
        density[i] = density[0] * (height[i] / height[0])**ExpansionExponent

        # The temperature evolution currently includes adiabatic cooling.
        # We may later wish to include additional terms such as
        # radiative cooling, energy stored in ionization, and additional heating

        temperature[i] = temperature[i-1]*(density[i]/density[i-1])**gamm1

        # Ionization time advance - to be added!!!!!!







        # Screen output if necessary


        i = i + 1


    # Create an object to store the inputs and outputs
 
    inputs = pd.Series([
                        initial_height,
                        log_initial_temp,
                        log_initial_dens, 
                        log_final_dens,
                        height_of_final_dens,
                        vfinal,
                        vscaletime,
                        ExpansionExponent, 
                        output_heights
                        ],
                       index=[
                              'initial_height',
                              'log_initial_temp',
                              'log_initial_dens',
                              'log_final_dens',
                              'height_of_final_dens',
                              'vfinal',
                              'vscaletime',
                              'ExpansionExponent', 
                              'output_heights',
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
        
        for i in range(max_steps+1):
            print(('i={0:4d}  t={1:>8.2f}  V={2:>6.1f}  h={3:>6.3f}  log(n)={4:>5.2f}'+\
                       '  log(T)={5:>5.2f}').format(i, time[i], velocity[i], 
                                                    height[i], np.log10(density[i]), 
                                                    np.log10(temperature[i])))



            # Example: 
            # print('{0:2d} {1:3d} {2:4d}'.format(x, x*x, x*x*x))


    return inputs
