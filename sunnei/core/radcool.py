# This file is woefully incomplete!!!!!!

from __future__ import print_function
import numpy as np
from scipy import interpolate

#>>> import matplotlib.pyplot as plt

#>>> x = np.arange(0, 10)
#>>> y = np.exp(-x/3.0)
#>>> f = interpolate.interp1d(x, y)

#>>>

#>>> xnew = np.arange(0, 9, 0.1)
#>>> ynew = f(xnew)   # use interpolation function returned by `interp1d`
#>>> plt.plot(x, y, 'o', xnew, ynew, '-')
#>>> plt.show()

def get_cooling_function():
    
    '''
    A table of the cooling rate coefficient Lambda(T) provided by John
    Raymond for the following abundance set:
    
    Abundance set:
    He    C    N    O    Ne   Mg   Si   S    Ar   Ca   Fe   Ni 
    10.93 8.82 7.96 8.52 7.96 7.52 7.60 7.20 6.90 6.30 7.60 6.30
    
    Multiply this by n_e * n_H to get cooling ergs*cm^3/s.  This includes
    lines & recombination, hydrogen, and bremsstrahlung.
    
    Note that this cooling curve assumes ionization equilibrium so there
    may be errors that arise when the plasma is far from ionization
    equilibrium!
    
    ***Need to double check units!  Is cooling in ergs?***
    '''

    logT = np.linspace(4.0,8.0,161) # Delta T = 0.025
    
    T = 10**logT

    Lambda = np.array(
        [0.662E+00,0.113E+01,0.190E+01,0.312E+01,0.492E+01,0.725E+01,
         0.968E+01,0.115E+02,0.123E+02,0.127E+02,0.123E+02,0.116E+02,
         0.107E+02,0.988E+01,0.932E+01,0.904E+01,0.901E+01,0.934E+01,
         0.991E+01,0.107E+02,0.115E+02,0.125E+02,0.137E+02,0.151E+02,
         0.168E+02,0.188E+02,0.213E+02,0.241E+02,0.275E+02,0.313E+02,
         0.357E+02,0.406E+02,0.460E+02,0.518E+02,0.576E+02,0.628E+02,
         0.671E+02,0.699E+02,0.710E+02,0.702E+02,0.677E+02,0.633E+02,
         0.580E+02,0.526E+02,0.480E+02,0.445E+02,0.422E+02,0.407E+02,
         0.397E+02,0.390E+02,0.382E+02,0.375E+02,0.368E+02,0.363E+02,
         0.357E+02,0.348E+02,0.333E+02,0.311E+02,0.280E+02,0.246E+02,
         0.214E+02,0.187E+02,0.166E+02,0.152E+02,0.142E+02,0.136E+02,
         0.133E+02,0.132E+02,0.130E+02,0.128E+02,0.124E+02,0.120E+02,
         0.116E+02,0.112E+02,0.110E+02,0.109E+02,0.109E+02,0.111E+02,
         0.112E+02,0.113E+02,0.114E+02,0.115E+02,0.115E+02,0.114E+02,
         0.113E+02,0.112E+02,0.111E+02,0.109E+02,0.106E+02,0.102E+02,
         0.974E+01,0.914E+01,0.840E+01,0.763E+01,0.687E+01,0.616E+01,
         0.553E+01,0.500E+01,0.457E+01,0.423E+01,0.396E+01,0.376E+01,
         0.360E+01,0.350E+01,0.344E+01,0.341E+01,0.341E+01,0.345E+01,
         0.350E+01,0.357E+01,0.364E+01,0.371E+01,0.377E+01,0.381E+01,
         0.382E+01,0.379E+01,0.371E+01,0.359E+01,0.343E+01,0.325E+01,
         0.305E+01,0.286E+01,0.267E+01,0.249E+01,0.234E+01,0.221E+01,
         0.210E+01,0.201E+01,0.193E+01,0.187E+01,0.183E+01,0.179E+01,
         0.177E+01,0.175E+01,0.174E+01,0.173E+01,0.173E+01,0.173E+01,
         0.174E+01,0.176E+01,0.177E+01,0.179E+01,0.182E+01,0.184E+01,
         0.187E+01,0.191E+01,0.194E+01,0.198E+01,0.202E+01,0.206E+01,
         0.210E+01,0.214E+01,0.219E+01,0.224E+01,0.229E+01,0.234E+01,
         0.239E+01,0.244E+01,0.250E+01,0.256E+01,0.261E+01,]
        )*1e-23
    
    assert T.size == Lambda.size, 'Mismatch in cooling function tables'

    return interpolate.interp1d(T,
                                Lambda, 
                                fill_value = 0.0,
                                )


    

    
    
        
        
    
