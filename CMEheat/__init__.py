
"""
CMEheat
=====

A module to investigate plasma heating during CMEs
"""


def track_plasma(
                 initial_height       = 0.1,     # in solar radii
                 log_initial_temp     = 6.0,     # K
                 log_initial_dens     = 9.6,     # cm^-3
                 log_final_density    = 6.4,     #
                 height_of_final_dens = 3.0,     # in solar radii
                 expansion_exponent   = 3.0,     # from 2.0 to 3.0
                 max_steps            = 100,     # maximum number of steps
                 output_heights       = [2.,3.],  
                 ):
    '''
    This module...
    '''

    import numpy as np
    import pandas as pd

    # Create a structure of some sort to store inputs and outputs
    #  - What should this structure be?
    
    inputs = pd.Series([
                        initial_height,
                        log_initial_temp,
                        log_initial_dens, 
                        log_final_density, 
                        height_of_final_dens,
                        expansion_exponent, 
                        output_heights
                        ],
                       index=[
                              'initial_height',
                              'log_initial_temp',
                              'log_initial_dens',
                              'log_final_dens',
                              'height_of_final_dens',
                              'expansion_exponent', 
                              'output_heights',
                              ])

    time = np.ndarray(max_steps+1)
    height = np.ndarray(max_steps+1)
    velocity = np.ndarray(max_steps+1)

    i = 1
    while (i < max_steps+1):
        
        timestep = find_timestep()
        time[i] = time[i-1]+timestep
        
        i = i + 1
        
                
#    print(time)

    return inputs
    
def find_timestep():
    return 1.0

def find_velocity(time, Vfinal, scaletime, velocity_model):
    
    


    return Vfinal*(1.0 - np.exp(time/scaletime))

def find_density(log_initial_density,
                 log_final_density,
                 time):
    return 1.0
