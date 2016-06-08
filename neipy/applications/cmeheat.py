"""
CMEheat
=====
This module contains non-equilibrium ionization routines to
investigate the heating of coronal mass ejection (CME) plasma.
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from neipy.core import func_index_te, func_dt_eigenval, func_solver_eigenval
from neipy.core import read_atomic_data, create_ChargeStates_dictionary

# Definining to be used constants

RSun = 6.957e5 # km
gamma = 5.0/3.0 # ratio of specific heats
gamm1 = gamma-1.0

# This pandas series allows a shortcut to finding the atomic number of
# an element.  For example, AtomicNumbers['Fe'] will return 26.

AtomicNumbers = pd.Series(np.arange(28)+1,
                          index=['H' ,'He',
                                 'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                                 'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',
                                 'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni',
                                 ])

He_per_H = 0.1 # number of Helium atoms per Hydrogen atom, regardless of ionization state

def cmeheat_track_plasma(
    initial_height       = 0.1,     # in solar radii
    final_height         = 5.0,    # height to output charge states
    log_initial_temp     = 6.0,     # K
    log_initial_dens     = 10.0,    # number density in cm^-3
    vfinal               = 500.0,   # km/s
    vscaletime           = 1800.0,  # s
    ExpansionExponent    = -2.5,    # dimensionless
    elements = ['H', 'He', 'C',     # elements to be modeled
                'N', 'O', 'Ne',
                'Mg', 'Si', 'S', 
                'Ar', 'Ca', 'Fe', ],
    screen_output=True,
    floor_log_temp = 3.6,
    safety_factor = 1.0,            # safety factor for time step of order 1
    ):
    
    '''
    The main program for tracking the ionization states of a blob of
    plasma as it is moving away from the Sun.  

    Example

    output = neipy.cmeheat_track_plasma(log_initial_temp=6.4, 
                                        log_initial_dens=9.4,
                                        vfinal=2500.0,
                                        ExpansionExponent=-2.5)
    
    Inputs

        initial_height: the distance above the solar photosphere of
        the starting position of the blob, in units of solar radii.
        This should generally be between 0.05 and 0.10 RSun.

        final_height: the distance above the solar photosphere of the
        ending position of the blob, in units of solar radii

        log_initial_temp: the logarithm of the initial temperature in
        Kelvin.  Should generally be between 4.6 and 7.0.

        log_initial_dens: the logarithm of the initial number density
        of hydrogen (both neutral and ionized).  If initial_height is
        changed, then this should be changed accordingly.

        vfinal: the final velocity of the blob, in km/s.

        vscaletime: the characteristic time scale for acceleration, in
        units of seconds.

        ExpansionExponent: the exponent that governs how the number
        density of hydrogen changes with height.  The equation is:
        (n/n0) = (h/h0)**ExpansionExponent.  This variable should
        generally be between -3.0 (in which case the density drops
        rapidly with height) and -1.5 (in which case the density drops
        mildly with height).  [Note: this variable is sometimes called
        alpha.]
        
        elements: a list containing strings of the symbols for the
        elements to be modeled.  This defaults to the dozen most
        abundant elements.  This list must include 'H' and 'He' in
        order to calculate the electron density later on.

        floor_log_temp: the logarithm of the minimum temperature in
        Kelvin for the ejecta.  If the temperature of the blob is
        calculated to drop below this critical temperature, then the
        temperature is reset to this value.  In effect, this parameter
        adds heat to the plasma.

        safety_factor: a value of order 1 that acts as a coefficient
        for how the time step is calculated.  For high quality runs,
        use a safety_factor of perhaps 0.3.  For practice runs, you may
        use a safety factor of 2 or 3 to save computational time.  

    Remaining tasks
        -Include radiative losses
        -Include ionization energy
        -Include different heating mechanisms
        -Include observational predictions (different program?)
        -Use photoionization to set lowest temperature somehow?
    '''

    # Check to make sure that realistic values for the inputs are
    # being used.  Suggest appropriate ranges, if needed.
 
    assert initial_height >= 0.01 and initial_height <= 0.5, \
        'Choose an initial height between 0.01 and 0.5 RSun (from 0.05 to 0.1 is best)'

    assert vfinal >= 50.0 and vfinal <= 5000.0, \
        'Need vfinal between 50.0 and 5000.0 km/s (from 250 to 2500 km/s is best)'

    assert log_initial_temp >= 3.0 and log_initial_temp <= 8.0, \
        'Need log_initial_temp between 3.0 and 8.0 (from 4.6 to 7.0 is best)'

    assert elements.__contains__('H'), \
        'The elements list must include H to calculate electron density'

    assert elements.__contains__('He'), \
        'The elements list must include He to calculate electron density'

    assert ExpansionExponent>=-4.0 and ExpansionExponent<=-0.9, \
        'Need ExpansionExponent between -4 and -0.9 (usually between -3.0 and -1.5)'

    # Read in the atomic data to be used for the non-equilibrium
    # ionization calculations.

    AtomicData = read_atomic_data(elements, screen_output=False)

    # The atomic data used for these calculations are stored in grids
    # that are a function of temperature at some resolution in
    # logarithm space (typically 0.01).  Find this resolution from the
    # AtomicData dictionary so that it can be used to update the time
    # step over the course of the simulation.

    logTres = np.log10(AtomicData['temperatures'][1]) - np.log10(AtomicData['temperatures'][0])

    # Estimate the maximum number of steps needed to complete the
    # simulation.

    max_steps = np.int64(12.0/(safety_factor*logTres))

    # Initialize arrays that will store physical conditions over the
    # course of the simulation

    time = np.zeros(max_steps+1)        # seconds
    height = np.zeros(max_steps+1)      # units of RSun
    velocity = np.zeros(max_steps+1)    # km/s
    density = np.zeros(max_steps+1)     # cm**-3: number density of H 
    electron_density = np.zeros(max_steps+1) # cm**-3
    temperature = np.zeros(max_steps+1) # K

    # Use the inputs to store values of important parameters at t=0

    height[0] = initial_height 
    density[0] = 10**log_initial_dens
    temperature[0] = 10**log_initial_temp

    # Set up the initial charge states.  Assume ionization equilibrium
    # at t=0 for the initial temperature.

    InitialChargeStates = create_ChargeStates_dictionary(elements,temperature[0],AtomicData)
    ChargeStateList = [InitialChargeStates]

    # Find the electron density associated with the initial time

    electron_density[0] = density[0]*electron_density_factor(ChargeStateList[0], He_per_H=He_per_H)

    # Find the time associated with the final height, which will be
    # needed to figure out when the simulation should end.

    final_time = find_time_for_height(final_height,vfinal,vscaletime,initial_height)

    # The time loop to calculate how the charge state distributions
    # change as the plasma blob moves away from the Sun.

    i = 1

    while (i <= max_steps):
        
        # Adjust the time step [turn this into a separate function?]

        if i==1:
            # Pick the first time step as a fraction of the
            # acceleration time scale
            dt_vs = vscaletime/200.0
            dt = dt_vs
        elif i>1 and i < max_steps-25:
            logTdiff = np.abs(np.log10(temperature[i-1])-np.log10(temperature[i-2]))
            if logTdiff > 0:
                # For most of the simulation, the difference in log T
                # between time steps should be comparable to the log T
                # resolution of the atomic data.
                dt = dt * safety_factor * (logTres/logTdiff)
                if time[i-1] <= (vscaletime/10.0) and dt > dt_vs:
                    # At very early times when the velocity is really
                    # slow, make sure that the time step does not
                    # exceed 1% of vscaletime to prevent ridiculously
                    # large time steps.
                    dt = dt_vs
            else:
                # If the temperature reaches the floor value, then
                # pick a time step based on how quickly the density is
                # changing.
                logdensdiff = np.abs(np.log10(density[i-1]) - np.log10(density[i-2]))
                dt = dt * safety_factor * (0.1/logdensdiff)
        elif i == max_steps-25:
            # If the number of iterations is approaching the maximum
            # number of time steps, then make dt be even for the remaining
            # time steps.  This avoids the situation where the last time
            # step becomes extremely long.
            dt = (final_time - time[i-1])/(max_steps-i-1)

        # Find the time step for the end of the simulation

        if time[i-1] + dt > final_time or i==max_steps:
            dt = final_time - time[i-1]
            FinalStep = True
        else:
            FinalStep = False

        time[i] = time[i-1] + dt

        # Determine the velocity and height using auxiliary functions

        velocity[i] = find_velocity(time[i], vfinal, vscaletime)
        height[i] = find_height(time[i], vfinal, vscaletime, initial_height)

        # The number density of Hydrogen is a power law with height.
        # The larger the magnitude of ExpansionExponent, the more
        # rapidly the density drops with height.
        
        density[i] = density[0] * (height[i]/height[0])**ExpansionExponent

        # The temperature evolution currently includes adiabatic
        # cooling.  We may later wish to include additional terms such
        # as radiative cooling, energy stored in ionization, and
        # different heating parameterizations.  Make sure that the
        # temperature does not drop below a floor value.  

        temperature[i] = temperature[i-1]*(density[i]/density[i-1])**gamm1

        if temperature[i] < 10**floor_log_temp:
            temperature[i] = 10**floor_log_temp

        # The ionization time advance requires a mean temperature and
        # a mean electron density.  Average these quantities from the
        # previous and current time indices.  To calculate the ratio
        # of the electron number density to the number density of H,
        # use the charge states of H and He from the previous time
        # index as an approximation.
        
        mean_temperature = 0.5*(temperature[i]+temperature[i-1])

        mean_electron_density = 0.5*(density[i]+density[i-1]) * \
            electron_density_factor(ChargeStateList[i-1], He_per_H=He_per_H)
        
        # Perform the ionization time advance using the eigenvalue
        # method described by C. Shen et al. (2015) and references
        # therein.  The main advantage of this method is its
        # stability: if you take an extremely long time step, then the
        # charge states will approach the equilibrium value for that
        # temperature.  

        NewChargeStates = func_solver_eigenval(elements, 
                                               AtomicData, 
                                               mean_temperature, 
                                               mean_electron_density, 
                                               dt, 
                                               ChargeStateList[i-1])

        ChargeStateList.append(NewChargeStates.copy())

        # Since the charge state distributions for this step have been
        # found, we can finally calculate the electron number density.

        electron_density[i] = density[i] * \
            electron_density_factor(ChargeStateList[i], He_per_H=He_per_H)

        # When calculating nsteps, note that it is the index of the
        # final element in each array.  

        if FinalStep: 
            nsteps = i 
            break
        else:
            i = i + 1
    
    '''
    Create a dictionary to store the inputs and outputs.  To access
    the contents of this dictionary, use output[key] where key is a
    string (in quotes if it is not a variable).
     
    # Accessing an input variable
      output['vfinal'] --> the final velocity of the blob in km/s
      output['elements'] --> the list of elements
      output['elements'][0] --> the first element listed, probably 'H'
    
    Accessing the time NumPy array:
      output['time'] --> the full time array
      output['time'][0] --> the starting time, which is zero
      output['time'][-1] --> the final time
      
    Accessing charge state information  
      output['ChargeStates'] --> the list of charge state dictionaries for different times
      output['ChargeStates'][0] --> the charge state dictionary for t=0
      output['ChargeStates'][-1] --> the charge state dictionary for the final time
      output['ChargeStates'][-1]['Fe'] --> the NumPy array containing charge states for iron at the final time
      output['ChargeStates'][-1]['Fe'][8] --> the ionization fraction for Fe 8+ at the final time
    '''

    output = {
        'time':time[0:nsteps+1],
        'height':height[0:nsteps+1],
        'velocity':velocity[0:nsteps+1],
        'density':density[0:nsteps+1],
        'electron_density':electron_density[0:nsteps+1],
        'temperature':temperature[0:nsteps+1],
        'ChargeStates':ChargeStateList,
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
        'floor_log_temp':floor_log_temp,
        'safety_factor':safety_factor,
        }

    # If requested, use a function defined elsewhere in the file to
    # print out information about this particular simulation.  This
    # function is also a part of the overall python package and can be
    # used to quickly get information about inputs plus initial/final
    # charge states.  When running multiple simulations in real time,
    # it is generally best to set screen_output=False to minimize
    # clutter on the screen.

    if screen_output:
        print_screen_output(output)
       
    # Print warnings for situations when the simulation may be inaccurate

    if i == max_steps:
        print()
        print("************************************************************************")
        print("Warning: max_steps too low! Final dt=",dt)
        print("************************************************************************")
        print()

    if safety_factor>4 and screen_output:
        print()
        print("************************************************************************")
        print("Warning: a large safety factor might prevent convergence!")
        print("safety_factor = ",safety_factor)
        print("************************************************************************")
        print()

    return output


def cmeheat_grid(
    initial_height = 0.1,
    final_height = 6.0,
    log_temp_range = [5.0,7.0], 
    log_dens_range = [9.0,11.0],
    vfinal_range = [500, 2000],
    vscaletime = 1800.0,
    ExponentRange = [-3.0,-1.5], 
    nvel = 2,
    ntemp = 2,
    ndens = 2,
    nexp = 2,                       
    max_steps = 2500, 
    dt = 20.0, 
    elements = ['H', 'He', 'C',     # elements to be modeled
                'N', 'O', 'Ne',
                'Mg', 'Si', 'S', 
                'Ar', 'Ca', 'Fe', ],
    floor_log_temp=2.0,
    ):

    '''
    Program: cmeheat_grid
    '''
    
    print()
    print("Running cmeheat_grid")


    # Make dictionaries to store 

    variables = ['V', 'T', 'n', 'e']

    ranges = {
        'V':np.array(vfinal_range),
        'T':np.array(log_temp_range),
        'n':np.array(log_dens_range),
        'e':np.array(ExponentRange),
        }
    
    sizes = {'V':nvel, 'T':ntemp, 'n':ndens, 'e':nexp, }

    gridinputs = {}

    # The number of temperatures, densities, velocities, and expansion
    # exponents is given by ntemp, ndens, nvel, and nexp.  Here we
    # make sure that these integers are consistent with the inputted
    # ranges.  
    
    # If any of the ranges have just one element, then change the size
            
    for var in variables:
        if ranges[var].size == 1:
            sizes[var] = 1
            gridinputs[var] = ranges[var][0]
        elif ranges[var].size == 2 and sizes[var] == 2:
            gridinputs[var] = ranges[var]
        elif sizes[var] > 2:
            gridinputs[var] = \
                np.linspace(ranges[var][0], ranges[var][1], sizes[var])
           
    # Print information about the grid of simulations

    print()

    print('Initial parameters:')
    print('nvel={0:>3d}   ntemp={1:>3d}   ndens={2:>3d}  nexp={3:>3d}'.format(
            gridinputs['V'].size, 
            gridinputs['T'].size, 
            gridinputs['n'].size,
            gridinputs['e'].size,
            ))

    print()

    # Loop through all of the different inputs for the grid of
    # simulations.  Add the results to the list_of_simulations.

    # Is there a better way to store this than a list of simulations???
    #  Need to put in jv, jt, jd, je

    list_of_simulations = []

    for jv in range(nvel):            
        for jt in range(ntemp):
            for jd in range(ndens):
                for je in range(nexp):

                    # Print information about each simulation

                    print('{0:>3d}{1:>3d}{2:>3d}{3:>3d}   V={4:>7.1f}  log T={5:>5.2f}  log n={6:>5.2f}  alpha={7:>5.2f}'.format(
                            jv,jt,jd,je,
                            gridinputs['V'][jv],
                            gridinputs['T'][jt],
                            gridinputs['n'][jd],
                            gridinputs['e'][je],))   
      
                    simulation = cmeheat_track_plasma(
                        initial_height = initial_height,
                        final_height = final_height,
                        log_initial_temp = gridinputs['T'][jt],
                        log_initial_dens = gridinputs['n'][jd],
                        vfinal = gridinputs['V'][jv],
                        vscaletime = vscaletime,
                        ExpansionExponent = gridinputs['e'][je],
                        max_steps = max_steps,
                        dt = dt,
                        elements = elements,
                        screen_output = False,
                        floor_log_temp = floor_log_temp,
                        )

                    list_of_simulations.append(simulation.copy())

    return list_of_simulations


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
                  xtol=1e-10)
    assert time[0]>=0, 'Negative time found using find_time_for_height'
    return time[0]

def find_time_for_height_aux(time, height, vfinal, vscaletime, initial_height):
    '''
    An auxiliary function for finding the time corresponding to a
    particular height.
    '''
    return height - find_height(time, vfinal, vscaletime, initial_height)

def electron_density_factor(ChargeStates, He_per_H=0.1):
    '''
    Find the ratio of the number density of electrons to the number
    density of Hydrogen (both neutral and ionized).  This uses the
    ChargeStates dictionary, and He_per_H is the number of Helium
    atoms divided by the number of Hydrogen atoms.
    '''
    assert He_per_H >= 0 and He_per_H <= 0.3, 'Need He_per_H between 0 and 0.3 to be realistic'
    ratio = ChargeStates['H'][1] + He_per_H*(ChargeStates['He'][1] + 2*ChargeStates['He'][2])
    assert ratio > 0 and ratio <= 1.0 + 2.0*He_per_H, 'Returning an invalid electron density factor: '+str(ratio)
    return ratio

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
            print(out['ChargeStates'][0][element])
            if AtomicNumbers[element] >= 10:
                print()
            print(out['ChargeStates'][-1][element])
            print()

def cmeheat_quicklook(output,
                      filename='quicklook.pdf'):

    # Calculate the time in hours

    time = output['time']/3600.0
    
    fig = plt.figure(figsize=(16,12))

    for i in range(4):
        ax = fig.add_subplot(3,3,i+1)
        ax.set_xlabel('Time (hours)')


        if i == 0:
            ax.plot(time, output['height'])
            ax.set_ylabel('Height (solar radii)')
            ax.set_title('Position')
            ax.axis([time[0],time[-1],0.0,np.max(output['height']*1.01)])
        elif i == 1:
            ax.plot(time, output['velocity'])
            ax.set_ylabel('Velocity (km/s)')
            ax.set_title('Velocity')
            ax.axis([time[0],time[-1],0.0,output['vfinal']*1.01])
        elif i == 2:
            ax.plot(time, np.log10(output['density']),
                    label='Hydrogen')
            ax.plot(time,
                    np.log10(output['electron_density']),
                    label='Electrons')
            ax.set_ylabel('Log number density (per cm3)')
            ax.legend(loc='best',fontsize=8.0)
            ax.set_title('Number Density')
            ax.axis([time[0],time[-1],
                     np.log10(np.min(output['electron_density']))-0.05,
                     np.log10(np.max(output['electron_density']))+0.05,
                     ])
        elif i == 3:
            ax.plot(time,
                    np.log10(output['temperature']))
            ax.set_ylabel('Log temperature (K)')
            ax.set_title('Temperature')

    # Now plot the charge states as a function of time

    for element in ['H', 'He', 'C', 'O', 'Fe']:
        i=i+1

        ChargeStateArray = MakeChargeStateArray(output,element)
        
        # figure out the range of charge states to plot over.  To make
        # the legend clearer, we can skip the 

        ChargeStatesToPlot = []

        for j in range(AtomicNumbers[element]+1):
            if np.max(ChargeStateArray[j,:] > 5e-4):
                ChargeStatesToPlot.append(j)


        ax = fig.add_subplot(3,3,i+1)
        for j in ChargeStatesToPlot:
            ax.plot(time,
                    ChargeStateArray[j,:], 
                    label=str(j)+'+')
            ax.axis([time[0],time[-1],0.0,1.01])
            ax.set_xlabel('Time (hours)')
            ax.set_ylabel('Ionization Fractions')
            ax.set_title('Charge States for '+element)
            ax.legend(loc='best',fontsize=7.0)

    fig.tight_layout(pad=1.0)

    fig.savefig('quicklook.pdf')

    plt.close(fig)

def MakeChargeStateArray(output, element='H'):

    ncharge = AtomicNumbers[element]+1
    nsteps = output['nsteps']

    ChargeStateArray = np.zeros([AtomicNumbers[element]+1,
                                 output['nsteps']+1])

    for istep in range(nsteps+1):
        ChargeStateArray[0:ncharge,istep] = \
            output['ChargeStates'][istep][element][0:ncharge]

    return ChargeStateArray
