# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 09:21:40 2016

@author: Chengcai & Nick

Update: 2016-06-04
        Initialization.
        2016-06-06
        Loops for all elements.
        2016-06-15
        Changed for the programe name 'SunNEI';
        Bug fixed.
"""
import sunnei
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

#------------------------------------------------------------------------------
# run a test
#------------------------------------------------------------------------------
run_output = sunnei.cmeheat_track_plasma(log_initial_temp=6.4, log_initial_dens=9.4, vfinal=2500.0, ExpansionExponent=-2.5)

#------------------------------------------------------------------------------
# plot
#------------------------------------------------------------------------------
# define element for showing
elems = run_output['elements']
for elem in elems:
    #element='O'
    element = elem

    natom=sunnei.applications.cmeheat.AtomicNumbers[element]
    nstates = natom+1
    nsteps = run_output['nsteps']
 
    # set up the figure frame
    fig = plt.figure()
    ax = plt.axes(xlim=(0.5, nstates+0.5), ylim=(0.0001, 1))
    ax.set_yscale('log')
    ax.set_xlabel('Ion Charge States')
    ax.set_ylabel(element+' Fractions')
    ax.set_xticks(np.linspace(1, nstates, nstates, dtype=int))
    line, = ax.plot([], [])
    plt.setp(line, color='b', linewidth=2, linestyle='steps-mid')
    
    # add text and other informations
    # text, = ...
    
    # initialization function, no data:
    def init():
        # text.set_text('')
        line.set_data([], [])
        return line, 

    # define the animate_frame function. Index 'i' is for the time index
    def animate_frame(i):
        x = np.linspace(1, nstates, nstates, dtype=int)   
        #y_eqi = ?
        y_nei = run_output['ChargeStates'][element][i,:]
        line.set_data(x, y_nei)
        
        # display text
        # text.set_text('Time = '+ run_output['times'] ...)
        return line,
    
    # draw the figure
    anim = animation.FuncAnimation(fig, animate_frame, init_func=init,   
                               frames=nsteps+1, interval=20)

    # Save animation into mp4 movie.
    # Set up formatting for the movie files. 
    # Here we choose the movie enginer 'ffmpeg'. One can find the avavliable 
    # enginer for your system. For example, the following code will print out a 
    # list of all available MovieWriters:
    # from matplotlib import animation
    # print(animation.writers.list())
    # ['ffmpeg', 'ffmpeg_file', 'imagemagick', 'imagemagick_file', ...]
    # If ffmpeg is not among it, you may install it first.
    matplot_anim_Writer = animation.writers['ffmpeg']
    my_writer = matplot_anim_Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

    chargestates_moviename = element+'_chgsts.mp4'
    anim.save(chargestates_moviename, writer=my_writer) 

    # Show animation in python.
    # plt.show()

    # Clear the current window ( or figure)
    plt.close('all')

# Normal stop
print('Normal stop in here!')
