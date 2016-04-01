
"""
CMEheat
=====

A module to investigate plasma heating during CMEs
"""

import numpy as np
import pandas as pd

# This pandas series allows a shortcut to finding the atomic number of
# an element.  For example, all_elements['Fe'] will return 26.


def track_plasma(
                 initial_height       = 0.1,     # in solar radii
                 log_initial_temp     = 6.0,     # K
                 log_initial_dens     = 9.6,     # cm^-3
                 log_final_density    = 6.4,     #
                 height_of_final_dens = 3.0,     # in solar radii
                 expansion_exponent   = 3.0,     # from 2.0 to 3.0
                 max_steps            = 100,     # maximum number of steps
                 output_heights       = [2.,3.], # heights to output charge states
                 elements = ['H', 'He', 'C',     # elements to be modeled
                             'N', 'O', 'Ne',
                             'Mg', 'Si', 'S', 
                             'Ar', 'Ca', 'Fe', ] 
                 ):
    '''
    Inputs
    '''

#    import numpy as np
#    import pandas as pd

    atomdata = create_atomic_dataframe(elements, '...')

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



def create_atomic_dataframe(elements, data_directory):
    
    import pandas as pd

    all_elements = pd.Series(np.arange(28)+1,
                         index=['H' ,'He',
                                'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                                'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',
                                'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni',
                                ])


    from scipy.io import FortranFile
    print('Running create_atomic_dataframe')

    data_directory = '/media/Backscratch/Users/namurphy/Projects/time_dependent_fortran/sswidl_read_chianti'

#    print(len(elements))

    list_of_dataframes = []


    for element in elements:

        AtomicNumber = all_elements[element]
        nstates = AtomicNumber + 1

        filename = data_directory + '/' + element.lower() + 'eigen.dat'
        H = FortranFile(filename, 'r')

        nte, nelems = H.read_ints(np.int32)
        temperatures = H.read_reals(np.float64)
        equistate = H.read_reals(np.float64).reshape((nte,nstates))
        eigenvalues = H.read_reals(np.float64).reshape((nte,nstates))
        eigenvector = H.read_reals(np.float64).reshape((nte,nstates,nstates))
        eigenvector_inv = H.read_reals(np.float64).reshape((nte,nstates,nstates))
        c_rate = H.read_reals(np.float64).reshape((nte,nstates))
        r_rate = H.read_reals(np.float64).reshape((nte,nstates))      
        
        # Store it in a pandas DataFrame

        DF = pd.DataFrame({AtomicNumber,
                           equistate,
                           },index=['AtomicNumber','equistate'])

#        list_of_dataframes.append(DF)

        # Add the DataFrame to the list_of_stuff

        print(DF)

    # Change the list of stuff into a DataFrame
    
    
