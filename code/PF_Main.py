"""File developed to implement a given particle with parameters as
   passed from PF_top.py

Development sponsor: Air Force Research Laboratory ATR Program
Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: May 2017"""

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

world_size = 500.0
landmarks = [[100.0,400.0], [250.0,100.0], [400.0,400.0]]

#transport needed visualization parameters to the visualization module
vis = visualize.vis(world_size,landmarks)

def particle_filter(n,fnoise,tnoise,snoise,steps,trials,methods):

    """
        n: number of particles
        fnoise: forward noise parameter
        tnoise: turning noise parameter
        snoise: sensing noise parameter
        steps: completions of pattern of 18 steps (total time steps = 18*steps)
        trials: number of times the particle filter will be simulated with
                current parameters
        methods: list of desired resampling methods for comparison
    """
    print(methods)
    #DEBUGGING VARIABLE VALUES
    # n = 500
    # fnoise = 0.2
    # tnoise = 0.2
    # snoise = 1.0
    # steps = 12
    # trials = 3

    #--------------------PARTICLE FILTERING OPERATIONS-----------------------------#

    Bot = Robot.robot()
    Bot.set_params(n,world_size,landmarks) #set robot environment parameters and number
                                           # of particles desired

    state = [250.0,250.0,1.0,0.0]
    #Bot.set_noise(fnoise,tnoise,snoise) #WHY IS THIS BAD NEWS??
    Bot.set(state[0],state[1],state[3]) # Initial position of the robot, will be randomly initialized otherwise

    true_pos=[]
    Bot_pos=[Bot.x,Bot.y]

    true_pos.append(Bot_pos)
    z = Bot.sense() #take initial measurement of surroundings
    p = []
    for i in range(trials): #generate a particle set for each trial (list of lists)
        p.append(Robot.create_uniform_particles(n,fnoise,tnoise,snoise,world_size,landmarks))

    mean_estimate = [[]for i in range(trials)]

    PRMSE = []

    for t in range(steps):
        j=0
        for j in range(18):

            if(j >= 6 and j <=10 ):
                state[2] = state[2] + 0.2
            if(j<6):
                # hdg = hdg + np.radians(15.0)
                state[3] = np.radians(15.0)
            else:
                state[3] = 0

            #initialize the robot that we would like to track
            Bot = Bot.move(state[3],state[2])
            state[0] = Bot.x
            state[1] = Bot.y

            Bot_pos = [Bot.x,Bot.y]
            true_pos.append(Bot_pos)

            z = Bot.sense() #take a measurement

            #-----------Loop over particles for multiple simulations------------#
            for j in range(trials):
                p2=[]
                for i in range(n):
                     p2.append(p[j][i].move(state[3],state[2])) # move the particles
                p[j] = p2

                w = []
                #generate particle weights based on measurement
                for i in range(n):
                    w.append(p[j][i].measurement_prob(z))

                w_norm = []
                for i in range(n):
                    w_norm.append((w[i])/np.sum(w)) # normalize the importance weights

                shrtWeights = ["%.3f" % elem for elem in w_norm] # show weight values to 3 decimals
                neff = int(Robot.neff(w_norm)) #calculate the effective sample size

                # flag = False
                if neff < n/2:
                    # p[j] = Robot.systematic_resample(n,w_norm,p[j])
                    p[j] = Robot.RS_resample(n,w_norm,p[j])
                    # flag = True
                    # print( 'Step =',t,', Evaluation = ', Bot.eval(Bot,p), ', neff = ', neff)

                #returns the mean and variance for each state variable
                #NOTE: only designed for 3 state variable and is not dynamic presently
                mu, var = Robot.estimate(w_norm,p[j])
                mean_estimate[j].append(mu)
                j += 1
                # Store state est. for all trials

    #Now use the stored mean estimates to calculate the PRMSE of the filter
    PRMSE = Robot.PRMSE(true_pos,mean_estimate)

    #WILL NEED TO ADJUST PLOTS TO DYNAMICALLY PLOT RESULTS FROM EACH DIFFERENT
    #RESAMPLING METHOD... PRESENTLY THIS ENTIRE BLOCK IS REPEATED FOR EACH METHOD
    #---------------------------------PLOTS--------------------------------------#
    fig, ax = plt.subplots()
    plt.plot(PRMSE)
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE (m)')
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('--')
    plt.show()
    # PRMSE.append(Robot.PRMSE(true_pos,mean_estimate))
    # vis.visualize(Bot,t,p2,p,w_norm,mu)

    fig, ax = plt.subplots()
    for x,y in true_pos:
        xt_pos = [i[0] for i in true_pos]
        yt_pos = [i[1] for i in true_pos]
        plt.plot(xt_pos,yt_pos,'-o', color="blue", markeredgecolor="black")
        for j in range(trials):
            xe_pos = [i[0] for i in mean_estimate[j]]
            ye_pos = [i[1] for i in mean_estimate[j]]
            plt.plot(xe_pos,ye_pos, '-o',markeredgecolor="black")


    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('--')
    plt.show()
    #-----------------------------------------------------------------------------#
