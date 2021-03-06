"""File developed to implement a given particle filter with parameters as
   passed from PF_top.py

    Mentor: Clark Taylor Ph.D.
    Developer: David R. Mohler
    Developed: Summer 2017"""

import Robot
# import ekf
import visualize
import numpy as np
from numpy import dot
# from filterpy.common import dot3
from math import *
import random
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

world_size = 100
landmarks = [[20.0, 80.0], [50.0, 20.0], [80.0, 80.0]]

#transport needed visualization parameters to the visualization module
vis = visualize.vis(world_size, landmarks)

#attempt at implementation of Particle Flow Particle filter (PFPF)
def PFPF(n, fnoise, tnoise, snoise, time_steps, trials, graphics):
    """
        n: number of particles
        fnoise: forward noise parameter
        tnoise: turning noise parameter
        snoise: sensing noise parameter
        time_steps: number of times the robot will move.
        methods: list of desired resampling methods for comparison
    """
    #--------------------PARTICLE FILTERING OPERATIONS-----------------------------#
    nLambda = 29
    true_pos = []
    p_init = []
    p = []
    R = np.diag([snoise]*len(landmarks)) #measurement error matrix

    Bot = Robot.robot() #Truth robot
    Bot.set_params(world_size, landmarks) #set robot environment parameters and
                                         #number of particles desired

    Bot2 = Robot.robot()
    Bot2.set_params(world_size, landmarks)

    #Robot input parameters, velocity, and heading change for the first robot
    U1 = [0, 0, 0, 0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0]
    U2 = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #Initialize robot states
    state = [50.0, 50.0, 1.0, 0.0]

    Bot.set_noise(0.1, np.radians(2.5), .1, 1.0)
    Bot.set(state[0], state[1], state[2], state[3]) # Initial state of the robot

    true_pos.append([Bot.x, Bot.y])

    z = Bot.sense() #take initial measurement of surroundings

    # p_init = Robot.create_uniform_particles(n,fnoise,tnoise,snoise,state[2],
                                            # world_size,landmarks)
    p_init = Robot.create_gaussian_particles(Bot, n, fnoise, tnoise, snoise, 10.,
                                             world_size, landmarks)

    mean_estimate = [[]for i in range(trials)]
    PRMSE = []

    # create a list of lists for every trial
    for i in range(trials):
        p.append(p_init)

    # Initialize with equal weights
    w = [1/n]*n

    #Generate pseudo time intervals
    lam_vec = Robot.GenerateLambda()
    tt = 0
    #--------------------------------------------------------------------------#
    for t in range(time_steps):
        print(t)
        #update states based on input arrays above (U1 and U2)
        control = [U1[t%len(U1)], U2[t%len(U2)]]
        state[2] = state[2] + control[0]
        state[3] = np.radians(control[1])

        #move the robot based on the input states
        Bot = Bot.move(state[3], state[2])
        state[0] = Bot.x
        state[1] = Bot.y

        true_pos.append([Bot.x, Bot.y]) #keep track of the true position

        z = np.asarray(Bot.sense()) #take a measurement

        for tr in range(trials):
            p2 = []
            for i in range(n):
                #move the particles
                p2.append(p[tr][i].move(state[3], state[2]))
            p[tr] = p2

            xbar, covar = Robot.estimate(w, p[tr])
            if graphics and t == 0:
                #arbitrarily select the first trial for graphics
                vis.visualize(Bot, t, p2, p[0], w, xbar)

            lam = 0

            for j in range(nLambda):
                lam += lam_vec[j] #pseudo time step
                for i in range(len(p[tr])):
                    pState = Robot.ParticleState(p[tr][i])
                    pmeasure = p[tr][i].sense()
                    #calculate H mat for each particle
                    H = Robot.h_jacobian(pState, landmarks)
                    A, b = Robot.calculate_flow_params(xbar, covar, H, R, z,
                                                      pmeasure, lam)
                    dxdl = dot(A, pState) + b
                    pState += (lam_vec[j]*dxdl)
                    pState[3] %= 2*np.pi #wrap heading value around
                    p[tr][i].set(pState[0], pState[1], pState[2], pState[3])

                xbar, covar = Robot.estimate(w, p[tr])
                if graphics:
                    #arbitrarily select the first trial for graphics
                    vis.visualize(Bot, j+1, p2, p[0], w, xbar)
            mean_estimate[tr].append(xbar)

    PRMSE = Robot.PRMSE(true_pos, mean_estimate)
    return mean_estimate, true_pos, PRMSE


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#------------------STANDARD PARTICLE FILTER TECHNIQUE----------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#

def ParticleFilt(n, fnoise, tnoise, snoise, time_steps, trials, methods, graphics):

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
    true_pos = []
    p_init = []
    p = []
    p_m = []
    resample_count = [0]*trials
    resample_percentage = [0]*trials

    Bot = Robot.robot()
    Bot.set_params(world_size, landmarks) #set robot environment parameters and number
                                           # of particles desired

    Bot2 = Robot.robot()
    Bot2.set_params(world_size, landmarks)

    #Robot input parameters, velocity, and heading change for the first robot
    U1 = [0, 0, 0, 0, 0, 0, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0]
    U2 = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    #Initialize robot states
    state = [50.0, 50.0, 1.0, 0.0]

    Bot.set_noise(0.1, np.radians(2.5), .1, 1.0)
    Bot.set(state[0], state[1], state[2], state[3]) # Initial state of the robot

    true_pos.append([Bot.x, Bot.y])

    z = Bot.sense() #take initial measurement of surroundings

    p_init = Robot.create_uniform_particles(n, fnoise, tnoise, snoise, state[2], 
                                            world_size, landmarks)
    # p_init = Robot.create_gaussian_particles(Bot,n,fnoise,tnoise,snoise,10.,world_size,landmarks)
    w = [1/n]*n
    #generate initial state estimate and covariance matrix
    xbar, covar = Robot.estimate(w, p_init)

    # create a list of lists for every trial
    for i in range(trials):
        p.append(p_init)

    # create a copy of the list for each resampling method
    for m in range(m_len):
        p_m.append(p)

    mean_estimate = [[[]for i in range(trials)]for m in range(m_len)]
    PRMSE = [[]for m in range(m_len)]

    p = p_init

    #--------------------------------------------------------------------------
    for t in range(time_steps):

        #update states based on input arrays above (U1 and U2)
        state[2] = state[2] + U1[t%len(U1)]
        state[3] = np.radians(U2[t%len(U2)])

        #move the robot based on the input states
        Bot = Bot.move(state[3], state[2])
        state[0] = Bot.x
        state[1] = Bot.y

        true_pos.append([Bot.x, Bot.y])

        z = Bot.sense() #take a measurement
        for m in range(m_len): #known issue: m loop is not properly structured
                               #to handle multiple resampling methods at this time.
            for tr in range(trials):

                p2 = []
                for i in range(n):
                    #move the particles
                    p2.append(p_m[m][tr][i].move(state[3], state[2]))
                p_m[m][tr] = p2

                w = []
                #generate particle weights based on current measurement
                for i in range(n):
                    w.append(p_m[m][tr][i].measurement_prob(z))

                w_norm = np.divide(w, np.sum(w))
                neff = int(Robot.neff(w_norm)) #calculate the effective sample size

                #if the effective sample size falls below 50% resample
                if neff < n/2:
                    resample_count[tr] += 1
                    p_m[m][tr] = Robot.resample(n, w_norm, p_m[m][tr], methods[m])
                    w_norm = [1/n]*n

                #returns the mean and variance for each state variable
                #only designed for 3 state variable and is not dynamic presently
                mu, var = Robot.estimate(w_norm, p_m[m][tr])
                mean_estimate[m][tr].append(mu)
                if graphics:
                    #arbitrarily select the first trial for graphics
                    vis.visualize(Bot, t, p2, p_m[0][0], w_norm, mu)
                for tr in range(trials):
                    resample_percentage[tr] = 100.0*(resample_count[tr]/time_steps)
                tr += 1
    print("Average Resampling Percentage: %", "%0.2f" % np.mean(resample_percentage))
    #----------------------------------------------------------------------#
    #Now use the stored mean estimates to calculate the PRMSE of the filter
    for m in range(m_len):
        PRMSE[m] = Robot.PRMSE(true_pos, mean_estimate[m])

    return mean_estimate, true_pos, PRMSE


#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
#--------------------------FILTER COMPARISION------------------------------#
#--------------------------------------------------------------------------#
#--------------------------------------------------------------------------#
def two_filters(n, fnoise, vnoise, tnoise, snoise, time_steps, trials, methods, 
                graphics, result_save_filename=None, random_seed=None):
    """
        This functions runs both standard and flow particle filters on
        the same set of initial data for performance comparisons.

        n: number of particles
        fnoise: forward noise parameter
        tnoise: turning noise parameter
        snoise: sensing noise parameter
        time_steps: number of times the robot will move.
        methods: list of desired resampling methods for comparison
    """
    #--------------------PARTICLE FILTERING OPERATIONS--------------------------#
    np.random.seed(random_seed)
    random.seed(random_seed)

    nLambda = 29
    true_state = []
    p_init = []
    p = []
    p_std = []
    w_std = []
    p_enkf = []
    p_aux = []
    w_aux = []
    R = np.diag([snoise]*len(landmarks)) #measurement error matrix
    resample_count = [0]*trials
    resample_percentage = [0]*trials

    Bot = Robot.robot() #Truth robot
    Bot.set_params(world_size, landmarks) #set robot environment parameters and number
                                           # of particles desired

    #Robot input parameters, velocity, and heading change for the first robot
    U1 = [0,0,0,0,0,0,0.05,0.05,0.05,0.05,0.05,0,0,0,0,0,0,0]
    U2 = [15.0,15.0,15.0,15.0,15.0,15.0,0,0,0,0,0,0,0,0,0,0,0,0]

    #Initialize robot states
    state = [random.uniform(10.0, 90.0), \
             random.uniform(10.0, 90.0), \
             random.uniform(1.75, 2.25), \
             random.uniform(0.0, 2.0*np.pi)]
    # state = [50.0, 50.0, 1.5, np.pi/2.0]

    Bot.set_noise(vnoise, tnoise, fnoise, snoise)
    Bot.set(state[0],state[1],state[2],state[3]) # Initial state of the robot

    true_state.append([Bot.x, Bot.y, Bot.vel, Bot.hdg])

    z = Bot.sense() #take initial measurement of surroundings

    mean_estimate_pfpf = [[]for i in range(trials)]
    mean_estimate_std = [[]for i in range(trials)]
    mean_estimate_enkf = [[]for i in range(trials)]
    mean_estimate_aux = [[]for i in range(trials)]
    covar_estimate_pfpf = [[]for i in range(trials)]
    covar_estimate_std = [[]for i in range(trials)]
    covar_estimate_enkf = [[]for i in range(trials)]
    covar_estimate_aux = [[]for i in range(trials)]

    # create a list of lists for every trial
    # Have to do this carefully as Python doesn't like to actually copy things
    for i in range(trials):
        p_init = Robot.create_uniform_particles(n, vnoise, tnoise, fnoise, snoise,state[2],
                                                world_size,landmarks)
        # p_init = Robot.create_gaussian_particles(Bot,n, vnoise, tnoise, fnoise, snoise,
        #                                          world_size,landmarks)

        p_tmp1=[]
        p_tmp2=[]
        p_tmp3=[]
        p_tmp4=[]
        for j in range(len(p_init)):
            p_tmp1.append(p_init[j].copy())
            p_tmp2.append(p_init[j].copy())
            p_tmp3.append(p_init[j].copy())
            p_tmp4.append(p_init[j].copy())
        p.append(p_tmp1)
        p_std.append(p_tmp2)
        p_enkf.append(p_tmp3)
        p_aux.append(p_tmp4)
        # Initialize with equal weights
        w = [1/n]*n  #Need a different object for each w_std
        w_std.append(w)
        w2 = [1/n]*n
        w_aux.append(w2)

    w = [1/n]*n  #Make one that does not have a copy sitting in w_std
    #Generate pseudo time intervals
    lam_vec = Robot.GenerateLambda()
    tt = 0
    #--------------------------------------------------------------------------#
    for t in range(time_steps):
        print('timestep', t, 'of', time_steps)
        #update states based on input arrays above (U1 and U2)
        control = [U1[t%len(U1)],U2[t%len(U2)]]

        #move the robot based on the input states
        # print('True state position, velocity, and heading are',Bot.state())        
        Bot.move(np.radians(control[1]), control[0])
        state = Bot.state()
        # print('After propagation, true state position, velocity, and heading are',state)

        true_state.append([Bot.x, Bot.y, Bot.vel, Bot.hdg])

        z = np.asarray(Bot.sense()) #take a measurement

        for tr in range(trials):
            for i in range(n):
                #move the particles
                p[tr][i].move(np.radians(control[1]), control[0])
                p_std[tr][i].move(np.radians(control[1]), control[0])
                p_enkf[tr][i].move(np.radians(control[1]), control[0])

            #generate particle weights based on current measurement
            xbar_std,covar_std = Robot.estimate(None,p_std[tr])
#            print("Std:  mean is (before weighting)", xbar_std)
            for i in range(n):
                w_std[tr][i] *= p_std[tr][i].measurement_prob(z)
            total_weight = np.sum(w_std[tr])
            for i in range(n):
                w_std[tr][i] /= total_weight
            neff = int(Robot.neff(w_std[tr])) #calculate the effective sample size
            #if the effective sample size falls below 50% resample
            p_std_prior = p_std[tr] #Keep around for visulization
            if neff < n/2:
                resample_count[tr] +=1
                p_std[tr], _ = Robot.resample(n, w_std[tr], p_std[tr], methods[0])
                w_std[tr] = [1/n]*n
            xbar_std, covar_std = Robot.estimate(w_std[tr], p_std[tr])
            # print("Std:  error is (after resample)", xbar_std[0:2]-true_state[-1][0:2])
            mean_estimate_std[tr].append(xbar_std)
            covar_estimate_std[tr].append(covar_std)

            #-----------------PARTICLE FLOW--------------------------#

            xbar, covar = Robot.estimate(w,p[tr])

            lam = 0

            # ### For Visualization only
            # p_prior=[] 
            # for i in range(n):
            #     p_prior.append(p[tr][i].copy())

            for j in range(nLambda):
                lam += lam_vec[j] #pseudo time step
                for i in range(len(p[tr])):
                    pState = Robot.ParticleState(p[tr][i])
                    pmeasure = p[tr][i].sense(add_noise=False)
                    #calculate H mat for each particle
                    H = Robot.h_jacobian(pState,landmarks)
                    A,b = Robot.calculate_flow_params(xbar,covar,H,R,z,pmeasure,lam)
                    dxdl = dot(A,pState) + b
                    pState += (lam_vec[j]*dxdl)
                    pState[3] %= 2*np.pi #wrap heading value around
                    p[tr][i].set(pState[0],pState[1],pState[2],pState[3])

                xbar, covar = Robot.estimate(w,p[tr])
            mean_estimate_pfpf[tr].append(xbar)
            covar_estimate_pfpf[tr].append(covar)

            #-----------------ENSEMBLE KALMAN FILTER--------------------------#
            #Create an array holding just the headings for display
            # hdg_array = []
            # for tmp_i in range(n):
            #     hdg_array.append(p_enkf[tr][tmp_i].hdg)
            # print('hdgs are', hdg_array )
            p_enkf_next = Robot.EnKF_update( p_enkf[tr], z )
            xbar_enkf, covar_enkf = Robot.estimate(None, p_enkf_next)
            mean_estimate_enkf[tr].append(xbar_enkf)
            covar_estimate_enkf[tr].append(covar_enkf)
            p_enkf[tr] = p_enkf_next
            # print('ENKF covar is',np.diag(covar_enkf))
            # print('ENKF mean state is',xbar_enkf)
            # print('ENKF error is',xbar_enkf[0:2]-true_pos[-1])

            #----------------AUXILIARY PARTICLE FILTER------------------------#
            #Before moving the particles, figure out how each particle might be 
            #weighted after the next step
            w_tmp = []
            for i in range(n):
                #move the particles
                p_aux_prev = p_aux[tr][i].copy()
                p_aux_prev.move(np.radians(control[1]), control[0], add_noise=False)
                w_tmp.append( p_aux_prev.measurement_prob(z) )
            #Decide whether to re-sample before propagating
            w_test = np.multiply( w_tmp, w_aux[tr] )
            # #This is manily a debugging thing
            # tmp = np.isnan(w_test)
            # print('Number of NaNs in the weights are: ',np.sum(tmp))
            #Test when numbers go to 0
            test_total = np.sum(w_test)
            if test_total == 0.0:
                w_test = [1/n]*n
            else:
                w_test /= test_total
            plt.plot(w_test)
            hdg_list=[]
            for i in range(n):
                hdg_list.append(p_aux[tr][i].hdg)
            plt.plot(hdg_list)
            plt.show()
            plt.close()
            test_neff = int(Robot.neff(w_test)) #calculate the effective sample size
            aux_resampled = (test_neff < (n*0.5))
            if aux_resampled:
                #Re-sample first, then propagate.
                total_weight = np.sum(w_test)
                w_test /= total_weight
                p_aux[tr],ancestry = Robot.resample(n, w_test, p_aux[tr], methods[0])
                w_aux[tr] = [1/n]*n
            for i in range(n):
                p_aux[tr][i].move(np.radians(control[1]), control[0])
                w_aux[tr][i] *= p_aux[tr][i].measurement_prob(z)
            #If resampled, remove weight used to determine sampling from 
            #total weight of particle
            if aux_resampled:
                tmp_array = np.array(w_tmp)
                w_aux[tr] = np.divide(w_aux[tr], tmp_array[ancestry])
            
            xbar_aux, covar_aux = Robot.estimate(None, p_aux[tr])
            mean_estimate_aux[tr].append( xbar_aux )
            covar_estimate_aux[tr].append( covar_aux )
            #print("Aux:  error is ", xbar_aux[0:2]-true_state[-1][0:2])
            
            #----------------Print out the pretty pictures with particles----------#
            if graphics:
                #arbitrarily select the first trial for graphics
                if tr==0:
                    #Visualize the Auxiliary particle filter results
                    # vis.visualize(Bot, t, p_aux[tr], p_aux[tr], w_aux[tr], xbar_aux)

                    #Visualize the EnKF results
                    # vis.visualize(Bot,t,p_enkf[0],p_enkf_next,w,xbar_enkf)

                    #Visualize the standard method
                    vis.visualize(Bot,t,p_std_prior,p_std[tr],w,xbar_std)

                    #Visualize the PFPF method
                    # vis.visualize(Bot,t,p_prior,p[tr],w,xbar)
            

    if result_save_filename != None:
        print('Saving file', result_save_filename)
        np.savez(result_save_filename, fnoise=fnoise,
                    tnoise=tnoise, snoise=snoise,
                    num_particles=n,
                    truth=true_state, 
                    std_est = mean_estimate_std,
                    std_covar = covar_estimate_std,
                    pfpf_est = mean_estimate_pfpf,
                    pfpf_covar = covar_estimate_pfpf,
                    aux_mean = mean_estimate_aux,
                    aux_covar = covar_estimate_aux,
                    enkf_mean = mean_estimate_enkf,
                    covar_enkf = covar_estimate_enkf)
    PRMSE = Robot.PRMSE(true_state, mean_estimate_pfpf)
    PRMSE_std =  Robot.PRMSE(true_state, mean_estimate_std)
    PRMSE_enkf = Robot.PRMSE(true_state, mean_estimate_enkf)
    PRMSE_aux = Robot.PRMSE( true_state, mean_estimate_aux)
    total_estimate = [mean_estimate_pfpf, mean_estimate_std, mean_estimate_enkf, mean_estimate_aux]
    total_RMSE = [PRMSE,PRMSE_std,PRMSE_enkf,PRMSE_aux]

    return total_estimate,true_state,total_RMSE