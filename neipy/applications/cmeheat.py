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
RSun = 6.957e5 # km

def cmeheat_track_plasma(
                 initial_height       = 0.1,     # in solar radii
                 log_initial_temp     = 6.0,     # K
                 log_initial_dens     = 9.6,     # number density in cm^-3
                 log_final_density    = 6.4,     #                 
                 height_of_final_dens = 3.0,     # in solar radii
                 vfinal               = 500.0,   # km/s
                 vscaletime           = 3600.0,  # s
                 expansion_exponent   = 3.0,     # from 2.0 to 3.0
                 max_steps            = 100,     # maximum number of steps
                 output_heights       = [2.,3.], # heights to output charge states
                 elements = ['H', 'He', 'C',     # elements to be modeled
                             'N', 'O', 'Ne',
                             'Mg', 'Si', 'S', 
                             'Ar', 'Ca', 'Fe', ] 
                 ):
    '''
    The main program for tracking the ionization states of a blob of
    plasma as it is moving away from the low corona.  
    '''

    # Create a structure to store the inputs and outputs
    
    inputs = pd.Series([
                        initial_height,
                        log_initial_temp,
                        log_initial_dens, 
                        log_final_density,
                        height_of_final_dens,
                        vfinal,
                        vscaletime,
                        expansion_exponent, 
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
                              'expansion_exponent', 
                              'output_heights',
                              ])



    # Initializations

    initial_temp = 10**log_initial_temp

    AtomicData = read_atomic_data(elements, screen_output=False)
    InitialChargeStates = create_ChargeStates_dictionary(elements,initial_temp,AtomicData)

#    density_scaleheight = find_density_scaleheight(log_initial_dens, 
#                                                   initial_height,
#                                                   log_final_dens,
#                                                   height_of_final_dens)
    
    time = np.ndarray(max_steps+1)
    height = np.ndarray(max_steps+1)
    velocity = np.ndarray(max_steps+1)
    density = np.ndarray(max_steps+1)
    temperature = np.ndarray(max_steps+1)



    # The main time loop

    i = 1
    while (i < max_steps):
        
        # Determine the time step

        timestep = 100.0  # need to update with better values
        time[i] = time[i-1] + timestep

        # Determine evolution of plasma parameters

        velocity[i], height[i] = find_velocity_and_height(time[i], vfinal, vscaletime, initial_height)
       
        # 

        i = i + 1

    return inputs


def find_velocity_and_height(time, Vfinal, scaletime, initial_height):

    velocity = Vfinal*(1.0 - np.exp(-time/scaletime))
    height = initial_height*RSun + Vfinal*(time + scaletime*(np.exp(-time/scaletime) - 1.0))

    return velocity, height

def find_density_scaleheight(log_initial_dens,
                             initial_height,
                             log_final_dens,
                             height_of_final_dens,
                             expansion_exponent):
    log_density_scaleheight = np.log10(height_of_final_dens) + e
                             
    return 1

def find_density(height,
                 log_initial_dens,
                 density_scaleheight,
                 expansion_exponent):
    return 1
    
