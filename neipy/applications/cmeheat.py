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
    initial_height       = 0.1,    # in solar radii
    final_height         = 6.0 ,    # height to output charge states
    log_initial_temp     = 6.0,     # K
    log_initial_dens     = 10.0,     # number density in cm^-3
    vfinal               = 250.0,   # km/s
    vscaletime           = 1800.0,  # s
    ExpansionExponent    = -2.0,    # dimensionless
    max_steps            = 2500,    # maximum number of steps
    dt                   = 20.0,   # s
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
    # being used.  Suggest appropriate ranges.
 
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

    # Find the time associated with the final height

    final_time = find_time_for_height(final_height,vfinal,vscaletime,initial_height)

    # The main time loop

    i = 1
    while (i <= max_steps):
        
        # Determine the time step

#        dt = 5.0  # second, need to update this!!!!!!!!!!!

#        dt = func_dt_eigenval(elements, AtomicData, te_list, ne_list, dt_in, 
#                                    change_perct=1.0e-3,
#                                    safety_factor=0.40,)
#                         dt_ne=1.0e5,
#                         dt_te=1.0e5,)

        if time[i-1] + dt > final_time:
            dt = final_time - time[i-1]
            FinalStep = True
        else:
            FinalStep = False

        time[i] = time[i-1] + dt

        # Determine the velocity and height

        velocity[i] = find_velocity(time[i],vfinal,vscaletime)
        height[i] = find_height(time[i], vfinal, vscaletime, initial_height)

        # Density is a power law with height
        #  - Still need to account for electron density
        
        density[i] = density[0] * (height[i]/height[0])**ExpansionExponent

        # The temperature evolution currently includes adiabatic cooling.
        # We may later wish to include additional terms such as
        # radiative cooling, energy stored in ionization, and additional heating

        temperature[i] = temperature[i-1]*(density[i]/density[i-1])**gamm1

        # Ionization time advance
        
        mean_temperature = 0.5*(temperature[i]+temperature[i-1])
        mean_density = 0.5*(density[i]+density[i-1])

        NewChargeStates = func_solver_eigenval(elements, AtomicData, mean_temperature, mean_density, dt, ChargeStateList[i-1])
        ChargeStateList.append(NewChargeStates.copy())

        if FinalStep: 
            nsteps = i
            break
        else:
            i = i + 1
    
    # Create a dictionary to store the inputs and outputs
 
    output = {
        'time':time[0:nsteps+1],
        'height':height[0:nsteps+1],
        'velocity':velocity[0:nsteps+1],
        'density':density[0:nsteps+1],
        'temperature':temperature[0:nsteps+1],
        'ChargeStateList':ChargeStateList,
        'initial_height':initial_height,
        'final_height':final_height,
        'log_initial_dens':log_initial_dens,
        'log_initial_temp':log_initial_temp,
        'ExpansionExponent':ExpansionExponent,
        'vfinal':vfinal,
        'vscaletime':vscaletime,
        'nsteps':nsteps,
        'final_time':final_time,
        'elements':elements,
        }

    if screen_output:
        print_screen_output(output)
       
    return output

def find_velocity(time, vfinal, vscaletime):
    '''
    Find the velocity of a blob as a function of time.  
    '''
    return vfinal*(1.0 - np.exp(-time/vscaletime))

def find_height(time, vfinal, vscaletime, initial_height):
    '''
    Find the height
    '''
    return initial_height+vfinal*(time+vscaletime*(np.exp(-time/vscaletime)-1.0))/RSun

def find_time_for_height(height, vfinal, vscaletime, initial_height):
    '''
    Find the time corresponding to a certain height
    '''
    from scipy.optimize import fsolve
    assert height>=initial_height, 'Need height >= initial_height'
    time = fsolve(find_time_for_height_aux, height, 
                  args=(height, vfinal, vscaletime, initial_height),
                  xtol=1e-13)
    assert time[0]>=0, 'Negative time found using find_time_for_height'
    return time[0]

def find_time_for_height_aux(time, height, vfinal, vscaletime, initial_height):
    '''
    An auxiliary function for finding the time corresponding to a
    particular height.
    '''
    return height - find_height(time, vfinal, vscaletime, initial_height)

def print_screen_output(out):
    '''
    Function for printing out the inputs and outputs of a run.  The
    input 'out' is the dictionary outputted by cmeheat_track_plasma.
    '''
    print()
    print('Input parameters:')
    print()
    print(('Initial height = {0:<5.3f} RSun').format(out['initial_height']))
    print(('Final height = {0:<5.3f} RSun').format(out['final_height']))
    print(('Log initial dens = {0:<7.2f}').format(out['log_initial_dens']))
    print(('Log initial temp = {0:<7.2f}').format(out['log_initial_temp']))
    print(('ExpansionExponent = {0:<7.3f}').format(out['ExpansionExponent']))
    print(('Initial height = {0:<7.3f}').format(out['initial_height']))
    print(('vscaletime = {0:<7.3f}').format(out['vscaletime']))
    print(('vfinal = {0:<7.3f}').format(out['vfinal']))

    if out['ExpansionExponent']>-1.0 or out['ExpansionExponent']<-3.5:
        print()
        print('Cautionary note: power law index should be between -3.5 and -1.0')
        print()

    # Find the frequency of screen output
            
    if out['nsteps'] <= 5:
        output_interval = 1
    elif out['nsteps'] <= 15: 
        output_interval = 2
    elif out['nsteps'] <= 35:
        output_interval = 5
    elif out['nsteps'] <= 70:
        output_interval = 10
    elif out['nsteps'] <= 160: 
        output_interval = 20
    elif out['nsteps'] <= 500:
        output_interval = 50
    elif out['nsteps'] <= 1200:
        output_interval = 100
    elif out['nsteps'] > 1200:
        output_interval = 200
        
    # Screen output on height, velocity, density, and temperature
        
    print()
    print('Output on time (s), velocity (km/s), and height (RSun)')
    print('with logarithms of temperature (K) and density (cm**-3)')
    print()

    for i in range(out['nsteps']+1):
        if (np.mod(i,output_interval) == 0) or (i == out['nsteps']):
            print(
                ('i={0:4d}  t={1:>7.1f}  V={2:>6.1f}  h={3:>6.3f}  log(n)={4:>5.2f}'+\
                     '  log(T)={5:>5.2f}').format(i, 
                                                  out['time'][i], 
                                                  out['velocity'][i], 
                                                  out['height'][i], 
                                                  np.log10(out['density'][i]), 
                                                  np.log10(out['temperature'][i]),
                                                  )
                )
            
    print()
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

    for element in out['elements']:

        if element == 'H' or element=='He' or element=='C' or \
                element=='O'  or element =='Fe':
            
            print('Initial and final charge states for '+element)
            print()
            print(out['ChargeStateList'][0][element])
            if AtomicNumbers[element] >= 10:
                print()
            print(out['ChargeStateList'][out['nsteps']][element])
            print()
                
