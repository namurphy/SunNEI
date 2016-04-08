if __name__=='__main__':

    import numpy as np
    from scipy.io import FortranFile

    filename = '/media/Backscratch/Users/namurphy/Projects/time_dependent_fortran/sswidl_read_chianti/heeigen.dat'

    H = FortranFile(filename, 'r')

    natom = 2


    nstates = natom+1

    nte, nelems = H.read_ints(np.int32)
    temperatures = H.read_reals(np.float64)
    equistate = H.read_reals(np.float64).reshape((nte,nstates))
    eigenvalues = H.read_reals(np.float64).reshape((nte,nstates))
    eigenvector = H.read_reals(np.float64).reshape((nte,nstates,nstates))
    eigenvector_inv = H.read_reals(np.float64).reshape((nte,nstates,nstates))
    c_rate = H.read_reals(np.float64).reshape((nte,nstates))
    r_rate = H.read_reals(np.float64).reshape((nte,nstates))



#    print(equistate)

#    for i in range(0,701):
#        print(i,temperatures[i],equistate[i,0], equistate[i,1])

    
