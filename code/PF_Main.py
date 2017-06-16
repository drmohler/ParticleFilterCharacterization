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
import matplotlib.animation as animation
import scipy.stats


world_size = 500.0
landmarks = [[100.0,400.0], [250.0,100.0], [400.0,400.0]]

#transport needed visualization parameters to the visualization module
vis = visualize.vis(world_size,landmarks)

#------------------------------USER INPUTS-------------------------------#

# while True:
#     try:
#         n = int(input("Input desired number of particles: "))
#
#     except ValueError:
#         print("ERROR: Number of particles must be an integer")
#
#     else:
#         break
#
# while True:
#     try:
#         fnoise = float(input("Input forward noise parameter: "))
#         tnoise = float(input("Input turning noise parameter: "))
#         snoise = float(input("Input sensing noise parameter: "))
#
#     except ValueError:
#         print("ERROR: noise parameters must be an floating point values")
#
#     else:
#         break
#
# while True:
#     try:
#         steps = int(input("Input desired iterations: "))
#         trials = int(input("Input desired trials (min. 1): "))
#
#     except ValueError:
#         print("ERROR: Number of iterations and trials must integers")
#         #Possibly add exception for error not in the correct range of values?
#     else:
        # break

#DEBUGGING VARIABLE VALUES
n = 500
fnoise = 0.2
tnoise = 0.2
snoise = 1.0
steps = 24
trials = 8

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

time = 0

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
                w_norm.append(w[i]/np.sum(w)) # normalize the importance weights

            neff = int(Robot.neff(w_norm)) #calculate the effective sample size

            # flag = False
            # if neff < n/2:
            p[j] = Robot.systematic_resample(n,w_norm,p[j])
                # flag = True

            # print( 'Step =',t,', Evaluation = ', Bot.eval(Bot,p), ', neff = ', neff)
            #if (t%10) == 0:

            #returns the mean and variance for each state variable
            #NOTE: only designed for 3 state variable and is not dynamic presently
            mu, var = Robot.estimate(w_norm,p[j])
            time += 1
            mean_estimate[j].append(mu)
            j += 1
            # Store state est. for all trials

#Now use the stored mean estimates to calculate the PRMSE of the filter

PRMSE = Robot.PRMSE(true_pos,mean_estimate)
# print(PRMSE)
plt.plot(PRMSE)
plt.show()

        # PRMSE.append(Robot.PRMSE(true_pos,mean_estimate))
        # print("truth   : ",Bot)
        # print("mean estimate: ", mu)


        # vis.visualize(Bot,t,p2,p,w_norm,mu)

        # for t in range(steps):
        #     print("PRMS at time [",t,"]: ", PRMSE[t])
print(len(mean_estimate))
print(len(mean_estimate[0]))
# print(mean_estimate[0][0][0])
# print(mean_estimate[0][0])
# print(mean_estimate[0])




for x,y in true_pos:
    xt_pos = [i[0] for i in true_pos]
    yt_pos = [i[1] for i in true_pos]
    plt.plot(xt_pos,yt_pos,'-o', color="blue", markeredgecolor="black")
    for j in range(trials):
        xe_pos = [i[0] for i in mean_estimate[j]]
        ye_pos = [i[1] for i in mean_estimate[j]]
        plt.plot(xe_pos,ye_pos, '-o',markeredgecolor="black")



    # plt.plot(mean_estimate[i])
plt.show()
# print(true_pos)


#-----------------------------------------------------------------------------#
