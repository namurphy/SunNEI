# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 09:21:40 2016

@author: ccai
"""
import neipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#------------------------------------------------------------------------------
# run a test
#------------------------------------------------------------------------------
run_output = neipy.cmeheat_track_plasma(log_initial_temp=6.4, log_initial_dens=9.4, vfinal=2500.0, ExpansionExponent=-2.5)

#------------------------------------------------------------------------------
# plot
#------------------------------------------------------------------------------
# define element for showing
element='O'

natom=neipy.applications.cmeheat.AtomicNumbers[element]
nstates = natom+1
nsteps = run_output['nsteps']
 
# set up the figure
fig = plt.figure()   
ax = plt.axes(xlim=(0, nstates), ylim=(0, 1))   
line, = ax.plot([], [], lw=2)

# initialization function:
def init():
    line.set_data([], [])   
    return line, 

# define the animate_frame function. I is the time index
def animate_frame(i):
    x = np.linspace(0, nstates, nstates)   
    #y = np.sin(2 * np.pi * (x - 0.01 * i))
    y = run_output['ChargeStates'][i][element]
    line.set_data(x, y)
    return line,
    
# draw thefigure
anim = animation.FuncAnimation(fig, animate_frame, init_func=init,   
                               frames=100, interval=20)

plt.show()