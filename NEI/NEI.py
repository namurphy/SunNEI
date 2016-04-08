"""
NEI
===
This module contains the core non-equilibrium ionization routines to
model astrophysical plasmas.
"""

import numpy as np
import pandas as pd

# This pandas series allows a shortcut to finding the atomic number of
# an element.  For example, all_elements['Fe'] will return 26.

all_elements = pd.Series(np.arange(28)+1,
                         index=['H' ,'He',
                                'Li','Be','B' ,'C' ,'N' ,'O' ,'F' ,'Ne',
                                'Na','Mg','Al','Si','P' ,'S' ,'Cl','Ar',
                                'K' ,'Ca','Sc','Ti','V' ,'Cr','Mn','Fe','Co','Ni',
                                ])

def read_atomic_data(elements, 
                     data_directory='/media/Backscratch/Users/namurphy/Projects/time_dependent_fortran/sswidl_read_chianti', 
                     screen_output=False):

    '''
    This routine reads in the atomic data to be used for the
    non-equilibrium ionization calculations.
 
    Instructions for generating atomic data files
    =============================================
    
    The atomic data files are generated from the routines described by
    Shen et al. (2015) and are available at:
    
    https://github.com/ionizationcalc/time_dependent_fortran
    
    First, run the IDL routine 'pro_write_ionizrecomb_rate.pro' in the
    subdirectory sswidl_read_chianti with optional parameters: nte
    (number of temperature bins, default=501), te_low (low log
    temperature, default=4.0), and te_high (high log temperature,
    default=9.0) to get an ionization rate table.  The routine outputs
    the file "ionrecomb_rate.dat" which is a text file containing the
    ionization and recombination rates as a function of temperature.
    This routine requires the atomic database Chianti to be installed
    in IDL.

    Second, compile the Fortran routine 'create_eigenvmatrix.f90'.
    With the Intel mkl libraries it is compiled as: "ifort -mkl
    create_eigenvmatrix.f90 -o create.out" which can then be run with
    the command "./create.out".  This routine outputs all the
    eigenvalue tables for the first 28 elements (H to Ni).

    As of 2016 April 7, data from Chianti 8 is included in the
    CMEheat/AtomicData subdirectory.
    '''

    if screen_output:
        print('read_atomic_data: beginning program')
    
    from scipy.io import FortranFile

    '''
    Begin a loop to read in the atomic data files needed for the
    non-equilibrium ionization modeling.  The information will be
    stored in the atomic_data dictionary.

    For the first element in the loop, the information that should be
    the same for each element will be stored at the top level of the
    dictionary.  This includes the temperature grid, the number of
    temperatures, and the number of elements.

    For all elements, read in and store the arrays containing the
    equilibrium state, the eigenvalues, the eigenvectors, and the
    eigenvector inverses.
    '''

    atomic_data = {}
    
    first_element_in_loop = True

    for element in elements:

        if screen_output:
            print('read_atomic_data: '+element)

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
        
        if first_element_in_loop:
            atomic_data['nte'] = nte
            atomic_data['nelems'] = nelems  # Probably not used but store anyway
            atomic_data['temperatures'] = temperatures
            first_element_in_loop = False
        else: 
            assert nte == atomic_data['nte'], 'Atomic data files have different number of temperature levels: '+element
            assert nelems == atomic_data['nelems'], 'Atomic data files have different number of elements: '+element
            assert np.allclose(atomic_data['temperatures'],temperatures), 'Atomic data files have different temperature bins'

        atomic_data[element] = {'element':element,
                                'AtomicNumber':AtomicNumber,
                                'nstates':nstates,
                                'equistate':equistate,
                                'eigenvalues':eigenvalues,
                                'eigenvector':eigenvector,
                                'eigenvector_inv':eigenvector_inv,
                                'ionization_rate':c_rate,
                                'recombination_rate':r_rate,
                                }
        
    if screen_output:
        print('read_atomic_data: '+str(len(elements))+' elements read in')
        print('read_atomic_data: complete')

    return atomic_data




