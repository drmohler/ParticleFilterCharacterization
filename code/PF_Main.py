""" Robot localization simulation with landmarks

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
import scipy.stats


world_size = 100.0
landmarks = [[20.0,20.0], [20.0,80.0], [20.0,50.0],
            [50.0,20.0], [50.0,80.0], [80.0,80.0],
            [80.0,20.0], [80.0,50.0]]

#transport needed visualization parameters to the visualization module
vis = visualize.vis(world_size,landmarks)

#------------------------------USER INPUTS-------------------------------#

while True:
    try:
        n = int(input("Input desired number of particles: "))

    except ValueError:
        print("ERROR: Number of particles must be an integer")

    else:
        break

while True:
    try:
        fnoise = float(input("Input forward noise parameter: "))
        tnoise = float(input("Input turning noise parameter: "))
        snoise = float(input("Input sensing noise parameter: "))

    except ValueError:
        print("ERROR: noise parameters must be an floating point values")

    else:
        break

while True:
    try:
        steps = int(input("Input desired iterations: "))

    except ValueError:
        print("ERROR: Number of iterations must be an integer")
        #Possibly add exception for error not in the correct range of values?
    else:
        break
#-----------------------------------------------------------------------------#
# n = 1000
# fnoise = 0.05
# tnoise = 0.05
# snoise = 5.0
# steps = 10


#--------------------PARTICLE FILTERING OPERATIONS-----------------------------#

Bot = Robot.robot()
Bot.set_params(n,world_size,landmarks) #set robot environment parameters and number
                                       # of particles desired
Bot.set(50,50,np.pi/2) # Initial position of the robot, will be randomly initialized otherwise
z = Bot.sense() #take initial measurement of surroundings


p = Robot.create_uniform_particles(n,fnoise,tnoise,snoise,world_size,landmarks)
control  = [-0.25,5.0]

for t in range(steps):
    #initialize the robot that we would like to track
    Bot = Bot.move(control[0],control[1])
    z = Bot.sense() #take a measurement

    #should develop more sophisticated method for defining robot path
    p2=[]
    for i in range(n):
         p2.append(p[i].move(control[0],control[1])) # move the particles
    p = p2

    w = []
    #generate particle weights based on measurement
    for i in range(n):
        w.append(p[i].measurement_prob(z))

    w_norm = []
    for i in range(n):
        w_norm.append(w[i]/np.sum(w)) # normalize the importance weights

    neff = int(Robot.neff(w_norm)) #calculate the effective sample size

    flag = False
    if neff < n/2:
        p = Robot.systematic_resample(n,w,p)
        flag = True

    print( 'Step =',t,', Evaluation = ', Bot.eval(Bot,p), 'resampled = ',flag , 'neff = ', neff)
    #if (t%10) == 0:

    #returns the mean and variance for each state variable
    #NOTE: only designed for 3 state variable and is not dynamic presently
    mu, var = Robot.estimate(w,p)
    vis.visualize(Bot,t,p2,p,w,mu)

#-----------------------------------------------------------------------------#
