"""File developed to implement a given particle filter with parameters as
   passed from PF_top.py

    Mentor: Clark Taylor Ph.D.
    Developer: David R. Mohler
    Developed: Summer 2017"""

import Robot
import ekf
import visualize
import numpy as np
from numpy.random import randn, random, uniform
from numpy import dot
from filterpy.common import dot3
from math import *
import random
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

world_size = 100
landmarks = [[20.0,80.0], [50.0,20.0], [80.0,80.0]]

#transport needed visualization parameters to the visualization module
vis = visualize.vis(world_size,landmarks)

#attempt at implementation of Particle Flow Particle filter (PFPF)
def PFPF(n,fnoise,tnoise,snoise,time_steps,methods,graphics):
    print("not complete")
    """
        n: number of particles
        fnoise: forward noise parameter
        tnoise: turning noise parameter
        snoise: sensing noise parameter
        time_steps: number of times the robot will move.
        methods: list of desired resampling methods for comparison
    """
    #--------------------PARTICLE FILTERING OPERATIONS-----------------------------#
    m_len = len(methods)
    nLambda = 29
    true_pos=[]
    p_init = []
    p = []
    resample_count = 0
    resample_percentage = 0

    Bot = Robot.robot()
    Bot.set_params(n,world_size,landmarks) #set robot environment parameters and number
                                           # of particles desired

    Bot2 = Robot.robot()
    Bot2.set_params(n,world_size,landmarks)

    #Robot input parameters, velocity, and heading change for the first robot
    U1 =  [0,0,0,0,0,0,0.05,0.05,0.05,0.05,0.05,0,0,0,0,0,0,0]
    U2 = [15.0,15.0,15.0,15.0,15.0,15.0,0,0,0,0,0,0,0,0,0,0,0,0]

    #Initialize robot states
    state = [50.0,50.0,1.0,0.0]

    Bot.set_noise(0.1,np.radians(2.5),0)
    Bot.set(state[0],state[1],state[2],state[3]) # Initial state of the robot
    # Bot2.set_noise(0.1,np.radians(2.5),0)
    # Bot.set(state_2[0],state_2[1],state_2[3])

    true_pos.append([Bot.x,Bot.y])

    z = Bot.sense(state) #take initial measurement of surroundings

    p_init = Robot.create_uniform_particles(n,fnoise,tnoise,snoise,state[2],
                                            world_size,landmarks)

    mean_estimate = []
    PRMSE = []
    p  = p_init

    # Initialize with equal weights
    w = [1/n]*n

    #generate initial state estimate and covariance matrix
    xbar , covar = Robot.estimate(w,p)

    #-----------------------------set up EKF-----------------------------------#
    # dt = 1# 0.05
    R = np.diag([snoise]*len(landmarks))
    # Q = np.diag([0,0,fnoise**2,tnoise**2])
    # F = np.array([[1,0,np.cos(state[3]),-state[2]*np.sin(state[3])],
    #                          [0,1,np.sin(state[3]),state[2]*np.cos(state[3])],
    #                          [0,0,1,0],
    #                          [0,0,0,1]])*dt
    #
    # kf = ekf.EKF(len(state),1,fnoise,tnoise,snoise)
    # kf.x = np.array(mu).T #initial state guess
    # kf._F = F
    # kf._P = covar #initial covariance estimate of particles
    # kf._Q = Q
    # kf._R = R

    #Generate pseudo time intervals
    lam_vec = Robot.GenerateLambda()
    #--------------------------------------------------------------------------#
    for t in range(time_steps):
        #update states based on input arrays above (U1 and U2)
        control = [U1[t%len(U1)],U2[t%len(U2)]]
        state[2] = state[2] + control[0]
        state[3] = np.radians(control[1])

        #move the robot based on the input states
        Bot = Bot.move(state[3],state[2])
        state[0] = Bot.x
        state[1] = Bot.y

        true_pos.append([Bot.x,Bot.y])

        z = np.asarray(Bot.sense(state)) #take a measurement

        p2=[]
        for i in range(n):
            #move the particles
            p2.append(p[i].move(state[3],state[2]))
        p = p2

        #generate particle weights based on current measurement
        for i in range(n):
            w[i] = p[i].measurement_prob(z)

        w_norm = []
        for i in range(n):
            # normalize the importance weights
            w_norm.append((w[i])/np.sum(w))

        #only need mean estimate, ignore new covariance
        xbar, covar = Robot.estimate(w,p)
        print("xbar:\n",xbar)
        lam = 0

        for j in range(nLambda):
            lam += lam_vec[j] #pseudo time step
            for i in range(len(p)):
                pState = Robot.ParticleState(p[i])
                print("Particle",i,":\n",p[i])
                #calculate H mat for each particle
                H = Robot.h_jacobian(pState,landmarks)
                print("current state:\n",state)
                A,b = Robot.caculate_flow_params(xbar,covar,H,R,z,lam)
                dxdl = dot(A,pState) + b
                print(dxdl)
                print(lam_vec[j]*dxdl)
                print(pState)
                pState += (lam_vec[j]*dxdl)
                print("True state:\n",Bot)
                print("particle before migrate:\n",p[i])
                p[i].set(pState[0],pState[1],pState[2],pState[3])
                print("particle after migrate:\n",p[i])
                input("observe migration")

            for i in range(n):
                w[i] = p[i].measurement_prob(z)
            w_norm = []
            for i in range(n):
                # normalize the importance weights
                w_norm.append((w[i])/np.sum(w))
            #only need mean estimate, ignore new covariance
            xbar, _ = Robot.estimate(w,p)
            print("new xbar:\n",xbar,Xbar)

#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#------------------STANDARD PARTICLE FILTER TECHNIQUE----------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

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

    Bot2 = Robot.robot()
    Bot2.set_params(n,world_size,landmarks)

    #Robot input parameters, velocity, and heading change for the first robot
    U1 =  [0,0,0,0,0,0,0.05,0.05,0.05,0.05,0.05,0,0,0,0,0,0,0]
    U2 = [15.0,15.0,15.0,15.0,15.0,15.0,0,0,0,0,0,0,0,0,0,0,0,0]

    # #Robot input parameters for the second robot
    # U1_2 =  [-0.25,-0.25,0,0,-0.25,-0.25,0,0,0.25,0.25,0,0,0.25,0.25]
    # U2_2 = [0,340.0,340.0,0,0,340.0,340.0,0,0,340.0,10.0,0,0,10.0]


    #Initialize robot states
    state = [50.0,50.0,1.0,0.0]
    # state_2 = [25.0,10.0,2.0,np.radians(90)]

    Bot.set_noise(0.1,np.radians(2.5),0)
    Bot.set(state[0],state[1],state[2],state[3]) # Initial state of the robot
    # Bot2.set_noise(0.1,np.radians(2.5),0)
    # Bot.set(state_2[0],state_2[1],state_2[3])

    true_pos.append([Bot.x,Bot.y])

    z = Bot.sense(state) #take initial measurement of surroundings

    # for i in range(trials): #generate a particle set for each trial (list of lists)
    # p_init = Robot.create_uniform_particles(n,fnoise,tnoise,snoise,state[2],world_size,landmarks)
    p_init = Robot.create_gaussian_particles(Bot,n,fnoise,tnoise,snoise,10.,world_size,landmarks)
    w = [1/n]*n
    #generate initial state estimate and covariance matrix
    xbar , covar = Robot.estimate(w,p_init)

    # create a list of lists for every trial
    for i in range(trials):
        p.append(p_init)

    # create a copy of the list for each resampling method
    for m in range(m_len):
        p_m.append(p)

    mean_estimate = [[[]for i in range(trials)]for m in range(m_len)]
    PRMSE = [[]for m in range(m_len)]

    p  = p_init

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

        z = Bot.sense(state) #take a measurement
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
                    w_norm = [1/n]*n

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
#----------------------------------------------------------------------#
    #Now use the stored mean estimates to calculate the PRMSE of the filter
    for m in range(m_len):
        PRMSE[m] = Robot.PRMSE(true_pos,mean_estimate[m])
#---------------------------------PLOTS------------------------------------#
    fig, ax = plt.subplots()
    for m in range(m_len):
        plt.plot(PRMSE[m])
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE (m)')
    plt.title("RMSE vs Time")
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('--')
    # plt.show()

    fig, ax = plt.subplots()
    flag = True
    flag2 = True
    count = 0
    for x,y in true_pos:
        xt_pos = [i[0] for i in true_pos]
        yt_pos = [i[1] for i in true_pos]

        if flag:
            label = "Truth"
            label
            flag = False
        else:
            label = None

        plt.plot(xt_pos,yt_pos,'-o', color="blue", markeredgecolor="blue", label=label)
        for m in range(m_len):
            for tr in range(trials):

                if flag2:
                    count += 1
                    trial_label = "Trial - " + str(tr+1)
                    if count == trials:
                        flag2 = False
                else:
                    trial_label = None
                xe_pos = [i[0] for i in mean_estimate[m][tr]]
                ye_pos = [i[1] for i in mean_estimate[m][tr]]
                plt.plot(xe_pos,ye_pos, '-x', label=trial_label)

    plt.legend()
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('--')
    plt.show()
    #-----------------------------------------------------------------------------#
