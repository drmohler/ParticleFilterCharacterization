"""File developed to implement a given particle with parameters as
   passed from PF_top.py

Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: Summer 2017"""

import Robot
import visualize
import numpy as np
from numpy.random import randn, random, uniform
from math import *
import random
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy.stats

world_size = 100.0
landmarks = [[20.0,80.0], [50.0,20.0], [80.0,80.0]]

#transport needed visualization parameters to the visualization module
vis = visualize.vis(world_size,landmarks)

#attempt at implementation of Particle Flow Particle filter (PFPF)
def PFPF(n,fnoise,tnoise,snoise,time_steps,trials,methods,graphics):


def ParticleFilt(n,fnoise,tnoise,snoise,time_steps,trials,methods,graphics):

    """
        n: number of particles
        fnoise: forward noise parameter
        tnoise: turning noise parameter
        snoise: sensing noise parameter
        time_steps: number of times the robot will move.
        trials: number of times the particle filter will be simulated with
                current parameters
        methods: list of desired resampling methods for comparison
    """

    #--------------------PARTICLE FILTERING OPERATIONS-----------------------------#
    m_len = len(methods)
    true_pos=[]
    p_init = []
    p = []
    p_m = []
    resample_count = [0]*trials
    resample_percentage = [0]*trials

    Bot = Robot.robot()
    Bot.set_params(n,world_size,landmarks) #set robot environment parameters and number
                                           # of particles desired

    #Robot input parameters, velocity, and heading change
    U1 =  [0,0,0,0,0,0,0.05,0.05,0.05,0.05,0.05,0,0,0,0,0,0,0]
    U2 = [15.0,15.0,15.0,15.0,15.0,15.0,0,0,0,0,0,0,0,0,0,0,0,0]
    state = [70.0,50.0,0.25,0.0]
    #Bot.set_noise(fnoise,tnoise,snoise) #WHY IS THIS BAD NEWS??
    Bot.set(state[0],state[1],state[3]) # Initial position of the robot, will be randomly initialized otherwise
    true_pos.append([Bot.x,Bot.y])

    z = Bot.sense() #take initial measurement of surroundings

    # for i in range(trials): #generate a particle set for each trial (list of lists)
    p_init = Robot.create_uniform_particles(n,fnoise,tnoise,snoise,world_size,landmarks)

    # create a list of lists for every trial
    for i in range(trials):
        p.append(p_init)
    # create a copy of the list for each resampling method
    for m in range(m_len):
        p_m.append(p)

    mean_estimate = [[[]for i in range(trials)]for m in range(m_len)]
    # mean_estimate = []

    PRMSE = [[]for m in range(m_len)]
    # PRMSE = []

    p  = p_init
    # test = input("wait here")

    #--------------------------------------------------------------------------
    for t in range(time_steps):

        #update states based on input arrays above (U1 and U2)
        state[2] = state[2] + U1[t%len(U1)]
        state[3] = np.radians(U2[t%len(U2)])

        #move the robot based on the input states
        Bot = Bot.move(state[3],state[2])
        state[0] = Bot.x
        state[1] = Bot.y

        true_pos.append([Bot.x,Bot.y])

        z = Bot.sense() #take a measurement
        for m in range(m_len):
            for tr in range(trials):

                p2=[]
                for i in range(n):
                    #move the particles
                    p2.append(p_m[m][tr][i].move(state[3],state[2]))
                p_m[m][tr] = p2

                w = []
                #generate particle weights based on current measurement
                for i in range(n):
                    w.append(p_m[m][tr][i].measurement_prob(z))
                # print("max: ", max(w))

                w_norm = []
                for i in range(n):
                    # normalize the importance weights
                    w_norm.append((w[i])/np.sum(w))
                shrtWeights = ["%.3f" % elem for elem in w_norm]

                neff = int(Robot.neff(w_norm)) #calculate the effective sample size

                #if the effective sample size falls below 50% resample
                if neff < n/2:
                    resample_count[tr] +=1
                    p_m[m][tr] = Robot.resample(n,w_norm,p_m[m][tr],methods[m])
                    for i in range(n):
                          w_norm[i] = 1/n

                #returns the mean and variance for each state variable
                #NOTE: only designed for 3 state variable and is not dynamic presently
                mu, var = Robot.estimate(w_norm,p_m[m][tr])
                mean_estimate[m][tr].append(mu)
                if graphics:
                    #arbitrarily select the first trial for graphics
                    vis.visualize(Bot,t,p2,p_m[0][0],w_norm,mu)
                for tr in range(trials):
                    resample_percentage[tr] = 100.0*(resample_count[tr]/time_steps)
                tr += 1
    print("Average Resampling Percentage: %", "%0.2f" % np.mean(resample_percentage))

                # Store state est. for all trials

        #----------------------------------------------------------------------#

    #Now use the stored mean estimates to calculate the PRMSE of the filter
    #for each of the resampling methods used

    for m in range(m_len):
        PRMSE[m] = Robot.PRMSE(true_pos,mean_estimate[m])

    #---------------------------------PLOTS------------------------------------#
    fig, ax = plt.subplots()
    for m in range(m_len):
        plt.plot(PRMSE[m])
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE (m)')
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('--')
    plt.show()

    fig, ax = plt.subplots()
    for x,y in true_pos:
        xt_pos = [i[0] for i in true_pos]
        yt_pos = [i[1] for i in true_pos]
        plt.plot(xt_pos,yt_pos,'-o', color="blue", markeredgecolor="black")
        for m in range(m_len):
            for tr in range(trials):
                xe_pos = [i[0] for i in mean_estimate[m][tr]]
                ye_pos = [i[1] for i in mean_estimate[m][tr]]
                plt.plot(xe_pos,ye_pos, '-o',markeredgecolor="black")

    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('--')
    plt.show()
    #-----------------------------------------------------------------------------#
