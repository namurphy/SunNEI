"""
CMEheat
=====
This module contains non-equilibrium ionization routines to
investigate the heating of coronal mass ejection (CME) plasma.
"""

from __future__ import print_function

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from sunnei.core import func_index_te, func_dt_eigenval, func_solver_eigenval
from sunnei.core import read_atomic_data, create_ChargeStates_dictionary, \
    ReformatChargeStateList, EquilChargeStates, get_cooling_function

# Definining constants

RSun = 6.957e5    # radius of the Sun in kilometers
gamma = 5.0/3.0   # ratio of specific heats
gamm1 = gamma-1.0 # 
kB = 1.38e-16     # Boltman constant in ergs per Kelvin

# This pandas series allows a shortcut to finding the atomic number of
# an element.  For example, AtomicNumbers['Fe'] will return 26.  The
# number of charge states for an element will be the atomic number
# plus one.

AtomicNumbers = pd.Series(np.arange(28)+1,
                          index=['H' ,'He',
                                 'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                                 'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',
                                 'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe',
                                 'Co','Ni',
                                 ])

# number of Helium atoms per Hydrogen atom, regardless of ionization state

He_per_H = 0.1 


def cmeheat_track_plasma(
    initial_height       = 0.1,     # in solar radii
    final_height         = 5.0,     # height to output charge states
    log_initial_temp     = 6.0,     # logarithm of initial temperature in K
    log_initial_dens     = 8.5,     # number density in cm^-3
    vfinal               = 500.0,   # km/s
    vscaletime           = 1800.0,  # s
    ExpansionExponent    = -2.4,    # dimensionless
    floor_log_temp       = 4.0,     # logarithm of floor temperature in K
    safety_factor = 1.0,            # multiplicative factor for time step
    elements = ['H', 'He', 'C',     # elements to be modeled
                'N', 'O', 'Ne',
                'Mg', 'Si', 'S', 
                'Ar', 'Ca', 'Fe', ],
    RadiativeCooling = True,
    screen_output=True,
    quicklook=True,
    barplot=True,
    ):
    
    '''
    The main program for tracking the ionization states of a blob of
    plasma as it is moving away from the Sun.  

    Example

    output = sunnei.cmeheat_track_plasma(log_initial_temp=6.4, 
                                        log_initial_dens=8.6,
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
        changed, then this should be changed accordingly.  This will
        often be between 8.0 and 9.0 (perhaps 7.5 to 9.5) for
        initial_height = 0.10 for plasma at MK temperatures.  For
        filament/prominence plasma at temperatures of order 0.1 MK,
        log_initial_dens could be of order 10.5.

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
        use a safety_factor of perhaps 0.3.  For practice runs, you
        may use a safety factor of 2 or 3 to save computational time.

        RadiativeCooling: Use the radiative cooling curve provided by
        John Raymond.  Note that the heating time becomes incredibly
        short for high density plasma (log_initial_dens > 10.0).

        screen_output: set to True to print out information on input
        parameters, plasma parameters over time, and initial/final
        charge states to the screen 

        quicklook: if set to True, then this function will output
        quicklook.pdf which contains quicklook information.  If set to
        a string containing the name of a pdf file, then the quicklook
        file will have that filename.  If no output is desired, then
        set to False.  The quicklook information includes plots of
        height vs. time; velocity, temperature, and density
        vs. height; and ionization fractions for a few elements as
        functions of height

    The output is a a dictionary with the following keys

        time: time values in seconds (NumPy array)

        height: height values in solar radii (NumPy array)

        velocity: velocity values in km/s (NumPy array)

        density: number density of neutral plus ionized hydrogen in
        units of particles per cm^3 (NumPy array)

        electron_density: number density of electrons in units of
        particles per cm^3 (NumPy array)

        temperature: temperature in Kelvin (NumPy array)

        ChargeStates: dictionary containing the evolution of the
        charge states with each element as a key to access a NumPy
        array containing 

        nsteps: the number of steps taken over the course of the
        simulation, not including t=0.

        final_time: The time the plasma reaches the final height

    The following keys in output are the same as the inputs:

        initial_height, final_height, log_initial_dens,
        log_initial_temp, ExpansionExponent, vfinal, vscaletime,
        elements, floor_log_temp, and safety_factor

    Remaining tasks
        -Double check radiative losses
        -Include ionization energy
        -Include different heating mechanisms
        -Include observational predictions (different program?)
        -Use photoionization to set lowest temperature somehow?
    '''

    # Check to make sure that realistic values for the inputs are
    # being used.  Suggest appropriate ranges, if needed.
 
    assert initial_height >= 0.01 and initial_height <= 0.5, \
        'Choose an initial height between 0.01 and 0.5 RSun'+\
        '(from 0.05 to 0.1 is best)'

    assert initial_height < final_height, \
        'Need initial_height < final_height'

    assert vfinal >= 50.0 and vfinal <= 5000.0, \
        'Need vfinal between 50.0 and 5000.0 km/s (from 250 to 2500 km/s is best)'

    if RadiativeCooling:
        assert floor_log_temp >= 4.0, \
            'If RadiativeCooling==True, then floor_log_temp must be at least 4.0'
    
    assert log_initial_temp >= 3.6 and log_initial_temp <= 8.0, \
        'Need log_initial_temp between 3.6 and 8.0 (from 4.6 to 7.0 is best)'
    
    assert elements.__contains__('H'), \
        'The elements list must include H to calculate electron density'
    
    assert elements.__contains__('He'), \
        'The elements list must include He to calculate electron density'
    
    assert ExpansionExponent>=-4.5 and ExpansionExponent<=-0.5, \
        'Need ExpansionExponent between -4.5 and -0.5 (usually between -3.0 and -2.0)'
    
    assert quicklook == True or quicklook == False or \
        quicklook.endswith('.pdf'), \
        'Need quicklook to be True or False or a string ending in .pdf'
    
    assert log_initial_temp>=floor_log_temp, \
        'Need log_initial_temp >= floor_log_temp'

    assert safety_factor > 0 and safety_factor <= 25, \
        'Need safety_factor to be a scalar between 0 and 25 (usually between 0.1 and 2)'


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

    max_steps = np.int64(15.0/(safety_factor*logTres))

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

    electron_density[0] = density[0]*electron_density_factor(ChargeStateList[0], 
                                                             He_per_H=He_per_H)

    # Find the time associated with the final height, which will be
    # needed to figure out when the simulation should end.

    final_time = find_time_for_height(final_height,vfinal,vscaletime,initial_height)

    # Get the interpolation function for radiative cooling

    Lambda = get_cooling_function()

    # The time loop to calculate how the charge state distributions
    # change as the plasma blob moves away from the Sun.

    i = 1
    dt = None

    while (i <= max_steps):
        
        # Adjust the time step

        dt, FinalStep = cmeheat_timestep(i, vscaletime, temperature,
                                         density, time, final_time,
                                         max_steps, safety_factor, logTres, 
                                         dt, RadiativeCooling)
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

        # The tem

        if RadiativeCooling:
            dT_rad = - dt * Lambda(temperature[i-1]) * \
                (2.0/3.0)*density[i-1]*electron_density[i-1] / \
                (kB * (density[i-1]*(1.0+He_per_H)+electron_density[i-1]))
        else:
            dT_rad = 0.0

        temperature[i] = temperature[i-1]*(density[i]/density[i-1])**gamm1 + \
            dT_rad
               
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
    The charge state information that was just created is a list of
    dictionaries of NumPy arrays.  The way to access data is:
    
    ChargeStateList[TimeStepIndex][element][ChargeStateIndex]

    However, this structure is not particularly convenient since it
    does not allow indexing of multiple time steps.  Next we change
    the format so that we can access data like:

    ChargeStates[element][TimeStepIndex,ChargeStateIndex]

    Here, ChargeStates is a dictionary where the keys are each element
    and access a NumPy array that stores the charge state evolution
    over time for that element.
    '''
    
    ChargeStates = ReformatChargeStateList(ChargeStateList, 
                                           elements, 
                                           nsteps)
    
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
      output['ChargeStates'] --> a dictionary containing charge state 
          arrays for different elements
      output['ChargeStates']['Fe'] --> NumPy array containing the 
          charge state evolution for iron
      output['ChargeStates']['Fe'][0,:] --> iron charge states for t=0
      output['ChargeStates']['Fe'][0,1] --> ionization fraction of
          singly ionized iron for t=0
      output['ChargeStates']['Fe'][-1,0] --> the ionization fraction
          of neutral iron for the final time
    '''

    output = {
        'time':time[0:nsteps+1],
        'height':height[0:nsteps+1],
        'velocity':velocity[0:nsteps+1],
        'density':density[0:nsteps+1],
        'electron_density':electron_density[0:nsteps+1],
        'temperature':temperature[0:nsteps+1],
        'ChargeStates':ChargeStates,
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
        'RadiativeCooling':RadiativeCooling,
        }

    '''
    If requested, use a function defined elsewhere in the file to
    print out information about this particular simulation.  This
    function is also a part of the overall python package and can be
    used to quickly get information about inputs plus initial/final
    charge states.  When running multiple simulations in real time, it
    is generally best to set screen_output=False to minimize clutter
    on the screen.
    '''

    if screen_output:
        print_screen_output(output)
       
    if quicklook != False:
        if quicklook == True:
            quicklookfile = 'quicklook.pdf'
        else:
            quicklookfile = quicklook

        cmeheat_quicklook(output,
                          filename=quicklookfile,
                          minfrac=1e-2)

    if barplot != False:
        if barplot == True:
            barplotfile = 'barplots.pdf'
        else:
            barplotfile = barplot

        cmeheat_barplot(output, filename=barplotfile, AtomicData=AtomicData)
        
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
    vfinal_range = [500.0, 2000.0],
    vscaletime_range = [1800, 2400],
    log_temp_range = [5.0,7.0], 
    log_dens_range =  [8.0,10.5],
    ExponentRange = [-3.0,-1.5], 
    nvel = 2,
    nvtime = 2,
    ntemp = 2,
    ndens = 2,
    nexp = 2,                       
    initial_height = 0.1,
    final_height = 10.0,
    floor_log_temp=4.0,
    safety_factor=1.0,
    elements = ['H', 'He', 'C',
                'N', 'O', 'Ne',
                'Mg', 'Si', 'S', 
                'Ar', 'Ca', 'Fe', ],
    RadiativeCooling=True,
    ):
    
    '''
    This program runs a grid of simulations.  This program works, but
    still requires improvements and comments.

    The inputs log_temp_range, log_dens_range, vfinal_range,
    vscaletime_range, and ExponentRange are either scalars or
    lists/arrays that contain information on the simulations to be ....

    The following inputs specify the ranges of different parameters to
    be used in the grid of simulations.

      vfinal_range: Final velocities in km/s
      vscaletime_range: Velocity scale times in seconds
      log_temp_range: Logarithms of initial temperatures in K
      log_dens_range: Logarithms of initial densities in particles per cm^3
      ExponentRange: Expansion exponents

    The following inputs specify the number of different parameters to
    be included in the grid of simulations.  These numbers are used
    only if the corresponding range input includes exactly two values.

      nvel: The number of final velocities to be included in grid
      nvtime: The number of velocity scale times to be included in grid
      ntemp: The number of initial temperatures to be included in grid
      ndens: The number of initial number densities to be included in grid
      nexp: The number of expansion exponents to be included in grid

    The number of temperatures, densities, velocities, and expansion
    exponents is given by ntemp, ndens, nvel, and nexp.  

    The inputs initial_height, final_height, floor_log_temp,
    safety_factor, and elements are identical to cmeheat_track_plasma,
    and indeed are just fed directly into that routine.  
    '''
    
    print()
    print('Running cmeheat_grid')
    print()

    # Check to make sure the inputs are in an appropriate format

    assert np.size(initial_height)==1, \
        'Need initial_height to be a scalar'

    assert np.size(final_height), \
        'Need final_height to be a scalar'

    assert np.size(safety_factor)==1, \
        'Need safety_factor to be a scalar'
    
    # Make a list of the keys to be used for the different variables
    # in the different dictionaries.  This makes it easier to do loops
    # like for key in keys that will do stuff for every variable.

    keys = ['V', 'tau', 'T', 'n', 'e']

    # Make dictionaries to store the inputs.  Here, ranges stores the
    # different inputted ranges for the different variables and sizes
    # stores the inputted sizes.  Note that there is some special
    # handling of ranges described below, which means that the
    # inputted sizes does not necessarily result in the final sizes.

    ranges = {
        'V':np.array(vfinal_range, dtype=np.float32, ndmin=1),
        'tau':np.array(vscaletime_range, dtype=np.float32, ndmin=1),
        'T':np.array(log_temp_range, dtype=np.float32, ndmin=1), 
        'n':np.array(log_dens_range, dtype=np.float32, ndmin=1),
        'e':np.array(ExponentRange, dtype=np.float32, ndmin=1),
        }
    
    sizes = {'V':nvel, 'tau':nvtime, 'T':ntemp, 'n':ndens, 'e':nexp, }
    
    # Create the gridinputs dictionary. This dictionary contains a
    # NumPy array with the different values to be scanned over for
    # each variable.  For example, gridinputs['V'] contains a NumPy
    # array with all of the velocities to be analyzed in the grid of
    # simulations.
    
    # If a certain range is given by a scalar or a list/array with
    # only one value, then gridinputs[key] will be an array including
    # that sole value.  If the range for a variable includes two
    # values, then gridinputs[key] will include a NumPy array with
    # evenly spaced values between these two extremes where the total
    # number is the size specified in the inputs (default=2).

    # Note that the sizes array is used in this step, but not
    # afterwards.

    gridinputs = {}

    for key in keys:
        if ranges[key].size == 1:
            sizes[key] = 1
            gridinputs[key] = ranges[key]
        elif ranges[key].size == 2:
            assert sizes[key] >= 2, \
                'Two elements in range input, but number of simulations requested is 1 for '+key
            gridinputs[key] = \
                np.linspace(ranges[key][0],
                            ranges[key][1],
                            sizes[key])
        elif ranges[key].size > 2:
            sizes[key] = np.size(ranges[key])
            gridinputs[key] = np.sort(ranges[key])
        else:
            assert False, \
                'Error in setting up grid inputs for key='+key
            
    # Print information about the grid of simulations to be performed

    print('Initial parameters:')
    print('nvel={0:>3d}   nvtime={1:>3d}   ntemp={2:>3d}   ndens={3:>3d}  nexp={4:>3d}'.format(
            gridinputs['V'].size, 
            gridinputs['tau'].size,
            gridinputs['T'].size, 
            gridinputs['n'].size,
            gridinputs['e'].size,
            ))
    print()

    # Loop through all of the different inputs for the grid of
    # simulations.  Add the results to the list_of_simulations.

    # Is there a better way to store this than a list of simulations?
    # It would be really helpful if it could be indexed as [jv, jtau,
    # jt, jd, je].  A possibility would be to create a helper routine.

    formatting_string = '{0:>3d}{1:>3d}{2:>3d}{3:>3d}{4:>3d}  '+\
        'V={5:>7.1f} tau={6:7.1f} log T={7:>5.2f} log n={8:>5.2f} alpha={9:>5.2f}'

    list_of_simulations = []

    for jv in range(gridinputs['V'].size):  
        for jtau in range(gridinputs['tau'].size):
            for jt in range(gridinputs['T'].size):
                for jd in range(gridinputs['n'].size):
                    for je in range(gridinputs['e'].size):

                        # Print information about each simulation

                        print(formatting_string.format(
                                jv,jtau,jt,jd,je,
                                gridinputs['V'][jv],
                                gridinputs['tau'][jtau],
                                gridinputs['T'][jt],
                                gridinputs['n'][jd],
                                gridinputs['e'][je],))   
      
                        # Run the simulation for the particular set of
                        # input parameters

                        simulation = cmeheat_track_plasma(
                            initial_height = initial_height,
                            final_height = final_height,
                            log_initial_temp = gridinputs['T'][jt],
                            log_initial_dens = gridinputs['n'][jd],
                            vfinal = gridinputs['V'][jv],
                            vscaletime = gridinputs['tau'][jtau],
                            ExpansionExponent = gridinputs['e'][je],
                            elements = elements,
                            floor_log_temp = floor_log_temp,    
                            screen_output=False,
                            quicklook=False,
                            barplot=False,
                            RadiativeCooling=RadiativeCooling,
                            )
                        
                        # Add this simulation's output to the list of
                        # simulations that will be outputted

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

def cmeheat_timestep(i, vscaletime, temperature, density, time, final_time,
                     max_steps, safety_factor, logTres, dt, RadiativeCooling):

    if RadiativeCooling and density[i-1]>9.6:
        dt_vs = safety_factor*vscaletime/200.0
    else:
        dt_vs = safety_factor*vscaletime/100.0

    if i==1:
            # Pick the first time step as a fraction of the
            # acceleration time scale
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
                # slow, make sure that the time step does not exceed a
                # fraction of vscaletime to prevent ridiculously large
                # time steps.
                dt = dt_vs
        else:
                # If the temperature reaches the floor value, then
                # pick a time step based on how quickly the density is
                # changing.
            logdensdiff = np.abs(np.log10(density[i-1]) - \
                                     np.log10(density[i-2]))
            if density[i-1] >= 10.0:
                dt = dt * safety_factor * (0.005/logdensdiff)
            elif density[i-1] >= 9.0:
                dt = dt * safety_factor * (0.01/logdensdiff)
            else:
                dt = dt * safety_factor * (0.04/logdensdiff)
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

    assert dt>0, 'Problem with dt: not returning nonnegative value'

    return (dt, FinalStep)

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
            print(out['ChargeStates'][element][0,:])
            if AtomicNumbers[element] >= 10:
                print()
            print(out['ChargeStates'][element][-1,:])
            print()

def cmeheat_quicklook(output,
                      xaxis='height',
                      filename='quicklook.pdf',
                      minfrac=1e-3,
                      ):
    '''
    This function allows quick viewing of the output of a particular
    simulation.  The first set of plots include height, velocity,
    number density (both neutral+ionized hydrogen, and electrons), and
    temperature.  The second set of plots show the evolution of the
    charge states as a function of time as the plasma blob is moving
    away from the Sun.
    '''

    assert filename.endswith('.pdf'), 'Need filename to end in .pdf'

    # Calculate the time in hours

    if xaxis[0]=='h':
        x = output['height']
        xlabel = 'Height (Solar Radii)'
    elif xaxis[0]=='t':
        x = output['time']/3600.0
        xlabel = 'Time (hours)'

    fontsize_title = 9.0
    fontsize_labels = 8.0
    fontsize_legend = 6.8
    fontsize_ticks = 7.0

    # First set of plots: height, velocity, density, and temperature
    # as functions of time for this simulation
    
    fig = plt.figure(figsize=(11,8.5))

    for i in range(4):
        ax = fig.add_subplot(3,3,i+1)
        ax.set_xlabel(xlabel, fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        if i == 0:
            ax.plot(output['time']/3600.0, output['height'])
            ax.set_ylabel('Height (solar radii)', fontsize=fontsize_labels)
            ax.set_title('Position',fontsize=fontsize_title)
            ax.axis([output['time'][0]/3600.0,output['time'][-1]/3600.0,
                     0.0, output['final_height']])
            ax.set_xlabel('Time (hours)', fontsize=fontsize_labels)
        elif i == 1:
            ax.plot(x, output['velocity'])
            ax.set_ylabel('Velocity (km/s)', fontsize=fontsize_labels)
            ax.set_title('Velocity',fontsize=fontsize_title)
            ax.axis([x[0],x[-1],0.0,output['vfinal']*1.0])
        elif i == 2:
            ax.plot(x, np.log10(output['density']),
                    label='Hydrogen')
            ax.plot(x,
                    np.log10(output['electron_density']),
                    label='Electrons', 
                    linestyle='--')
            ax.set_ylabel('Log number density (per cm3)', fontsize=fontsize_labels)
            ax.legend(loc='best',fontsize=fontsize_legend, handlelength=3)
            ax.set_title('Number Density', fontsize=fontsize_title)
            ax.axis([x[0],x[-1],
                     np.log10(np.min([output['electron_density'],
                                      output['density']])),
                     np.log10(np.max([output['electron_density'],
                                      output['density']])),
                     ])
        elif i == 3:
            ax.plot(x, np.log10(output['temperature']), label='With radiative cooling')
            ax.set_ylabel('Log temperature (K)', fontsize=fontsize_labels)
            ax.set_title('Temperature',fontsize=fontsize_title)
            ax.axis([x[0],x[-1],
                    np.min( np.log10(output['temperature'])-0.03 ),
                    np.max( np.log10(output['temperature']) )])
            # For cases with radiative cooling, include a line
            # representing what the temperature would be in the case
            # of just adiabatic expansion
            if output['RadiativeCooling']:
                temperature_adiabatic = output['temperature'][0] * \
                    (output['density']/output['density'][0])**gamm1
                ax.plot(x, 
                        np.log10(temperature_adiabatic),
                        linestyle=':', label='Adiabatic (for comparison)', color='red')
                ax.legend(loc='best', fontsize=fontsize_legend, handlelength=3)
# Plotting style preliminaries for second set of plots

    # Choose a set of linestyles to be cycled around

    styles = ['-', '-.', '--',]

    # Make sure the lines are thick enough if there are dots and/or dashes

    if styles.__contains__(':') or styles.__contains__('-.'):
        ChargeStateLineWidth=1.3
    elif styles.__contains('--'):
        ChargeStateLineWidth=1.2
    else:
        ChargeStateLineWidth=1.12

    # Use a colorblind friendly palette 
    # For the moment, use one adapted from:
    # http://www.somersault1824.com/tips-for-designing-scientific-figures-for-color-blind-readers/

    colors = np.array([
        [  0,   0,   0],
        [  0,  73,  73],
        [0  , 146, 146],
        [255, 109, 182],
        [255, 182, 119],
        [ 73,   0, 146],
        [  0, 109, 219],
        [182, 109, 255],
        [109, 182, 255],
        [182, 219, 255],
        [146,   0,   0],
        [146,  73,   0],
        [219, 209,   0],
        [ 36, 255,  36],
        [255, 255, 109],
        ])/255.0

    # Second set of plots: charge states as a function of x for a
    # set of elements

    for element in ['H', 'He', 'C', 'O', 'Fe']:

        i=i+1
        
        # Figure out the set of charge states that need to be plotted.
        # Choose only the charge states for which the maximum
        # ionization fraction exceeds minfrac which is set as a
        # keyword above with a default value of 0.001.  For heavy
        # elements, this helps make sure the plots and legends do not
        # get too cluttered.

        ChargeStatesToPlot = []

        for j in range(AtomicNumbers[element]+1):
            if np.max(output['ChargeStates'][element][:,j] > minfrac):
                ChargeStatesToPlot.append(j)

        ax = fig.add_subplot(3,3,i+1)

        # Plot each of the different charge states for this element

        for j in ChargeStatesToPlot:
            ax.plot(x,
                    output['ChargeStates'][element][:,j],
                    linestyle=styles[j % len(styles)],
                    color=colors[j % len(colors[:,0]),:],
                    linewidth=ChargeStateLineWidth,
                    label=str(j)+'+')
            
        ax.axis([x[0],x[-1],0.0,1.00])
        ax.set_xlabel(xlabel,fontsize=fontsize_labels)
        ax.set_ylabel('Ionization Fractions',fontsize=fontsize_labels)
        ax.set_title('Charge States for '+element,fontsize=fontsize_title)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        
        handlelength = 3.0

        if len(ChargeStatesToPlot)<4:
            ncol=1
        elif len(ChargeStatesToPlot)<9:
            ncol=2
        elif len(ChargeStatesToPlot)<11:
            ncol=3
            handlelength=2.5
        else:
            ncol=5
            handlelength=2.3
            


        ax.legend(loc='best', 
                  fontsize=fontsize_legend, 
                  ncol=ncol, 
                  handlelength=handlelength,
                  columnspacing=0.6)

    fig.tight_layout(pad=0.6)

    fig.savefig(filename)

    plt.close(fig)

def cmeheat_barplot(output, 
                    element='all', 
                    filename=False,
                    ShowClosedShells=True,
                    ShowFinalEquilibrium=False,
                    AtomicData=None,
                    ):
    '''
    Make bar plots of the ionization fractions at the beginning and
    end of a simulation.  The first input is the output from
    cmeheat_track_plasma.  If element=='all' then all of the elements
    will be plotted (default).  If element equals the atomic symbol of
    an element that was modeled, then plot only that element.  If
    filename is specified, then it will save the plot to that file,
    but it must end with pdf.
    '''

    assert element=='all' or output['elements'].__contains__(element), \
        'Need element to equal all or the symbol of an element that was modeled'

    # Plotting preliminaries

    fontsize_title = 9.2
    fontsize_labels = 8.0
    fontsize_legend = 8.6

    width = 1.0

    # Choose whether or not to make bar plots

    if element=='all':
        if filename==False:
            filename='barplots.pdf'
            
        figsize=(11,8.5)
        elements = output['elements']            
        nxplots, nyplots = 4, 3
    else:
        if filename==False:
            filename='barplot_'+element+'.pdf'
        figsize=(4.0,3.5)
        elements = [element]
        nxplots, nyplots = 1, 1


    # Begin the plot
            
    fig = plt.figure(figsize=figsize)
    
    i = 0
    for elem in elements:

        if AtomicNumbers[elem] <= 18:
            fontsize_ticks = 7.6
        elif AtomicNumbers[elem] <= 22:
            fontsize_ticks = 6.4
        else:
            fontsize_ticks = 5.7

        x = np.linspace(0, AtomicNumbers[elem], AtomicNumbers[elem]+1, 
                        dtype=np.int16)

        ax = fig.add_subplot(nxplots,nyplots,i+1)

        ax.set_title(elem, fontsize=fontsize_title)
        ax.set_ylabel('Ionization fractions', fontsize=fontsize_labels)
        if nxplots==1 and nyplots==1:
            ax.set_xlabel('Charge States', fontsize=fontsize_labels)
        ax.tick_params(axis='both', labelsize=fontsize_ticks)
        ax.set_xlim(0, len(x))
        ax.set_ylim(0, 1)
        ax.set_xticks(x+width/2.0)
        ax.set_xticklabels(x)

        # If requested, show the equilibrium charge states associated
        # with the final temperature.

        if ShowFinalEquilibrium:
            EqChargeStates = EquilChargeStates(output['temperature'][-1],elem,AtomicData=AtomicData)
            ax.bar(x, 
                   EqChargeStates,
                   color='white',
                   label='Equil',
                   width=width,
                   alpha=1.0)

        # Plot the initial and final charge state distributions

        ax.bar(x, 
               output['ChargeStates'][elem][0,:],
               color='red',
               label='$'+str(output['initial_height'])+'R_\odot$',
               width=width,
               alpha=0.6,
               )

        ax.bar(x,
               output['ChargeStates'][elem][-1,:],
               color='blue',
               label='$'+str(output['final_height'])+'R_\odot$',
               width=width,
               alpha=0.6,
               )

        # Put in a vertical dotted line if there is a closed shell.
        # This corresponds to

        if ShowClosedShells:
            if AtomicNumbers[elem]>=2:
                ax.plot([AtomicNumbers[elem]-1.5, AtomicNumbers[elem]-1.5], [0.0, 1.0], 
                        'k:', linewidth=0.4)
            if AtomicNumbers[elem]>=10:
                ax.plot([AtomicNumbers[elem]-9.5, AtomicNumbers[elem]-9.5], [0.0, 1.0], 
                        'k:', linewidth=0.4)
            if AtomicNumbers[elem]>=18:
                ax.plot([AtomicNumbers[elem]-17.5, AtomicNumbers[elem]-17.5], [0.0, 1.0], 
                        'k:', linewidth=0.4)

        if i==0:
            ax.legend(loc='best', 
                      fontsize=fontsize_legend, 
                      labelspacing=0.2,
                      borderaxespad=0.66,
                      )
        i = i+1

    fig.tight_layout(pad=0.6)

    fig.savefig(filename)

    plt.close(fig)

