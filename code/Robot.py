""" Robot simulation for particle filtering

Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: Summer 2017"""

import numpy as np
from numpy.random import randn, random, uniform
from numpy import dot,eye
from numpy.linalg import inv
from filterpy.common import dot3
from math import *
import random
import matplotlib.pyplot as plt
import scipy.stats

class robot:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.vel = 0.0
        self.hdg = 0.0
        self.world_size = 100
        self.landmarks = [[0,0]]
        self.N = 100
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set_params(self,new_N,new_world_size,new_landmarks):
        self.N = int(new_N)
        self.world_size = int(new_world_size)
        self.landmarks = new_landmarks

    def set(self,new_x,new_y,new_vel,new_hdg): #place the robot at a given spot
        if new_x < 0 or new_x >= self.world_size:
            print("current world size: ", self.world_size)
            print("Bad X value: ",new_x)
            raise ValueError('X coordinate out of bounds')
        if new_vel < 0:
            print("Negative Velocity Occured.")
            raise ValueError('Non-positive velocity')
        if new_y < 0 or new_y >= self.world_size:
            print("Bad Y value: ",new_y)
            raise ValueError('Y coordinate out of bounds')
        if new_hdg < 0 or new_hdg >= 2*np.pi:
            print("Bad hdg value: ",new_hdg)
            raise ValueError('hdg must be in range [0,2*Pi]')

        self.x = float(new_x)
        self.y = float(new_y)
        self.vel = float(new_vel)
        self.hdg = float(new_hdg)

    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise):
        """ Set the noise parameters, changing them is often useful in particle filters
        :param new_forward_noise: new noise value for the forward movement
        :param new_turn_noise:    new noise value for the turn
        :param new_sense_noise:  new noise value for the sensing
        """

        self.forward_noise = float(new_forward_noise)
        self.turn_noise = float(new_turn_noise)
        self.sense_noise = float(new_sense_noise)

    def sense(self,x):
        """ Sense the environment: calculate distances to landmarks
        :return measured distances to the known landmarks
        """
        x = None

        z = []
        for i in range(len(self.landmarks)):
            dist = sqrt((self.x - self.landmarks[i][0]) ** 2 + (self.y - self.landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise) #0 mean noise, user defined standard deviation
            z.append(dist)
        return z #measure relative to each landmark

    def measurement_prob(self, measurement):
        """
        Apply weighting to particles based on recieved measurement
        """
        prob = 1.0
        for i in range(len(self.landmarks)):
            dist = sqrt((self.x - self.landmarks[i][0])**2 + (self.y-self.landmarks[i][1])**2)
            prob *= self.Gaussian(dist,self.sense_noise,measurement[i])

        return prob

    def Gaussian(self,mu,sigma,x):
        #calculates the probability of x for 1-Dim Gaussian
        #with mean mu and std dev sigma
        g = exp(-((mu-x)**2)/(sigma**2)/2.0)/sqrt(2.0*np.pi*(sigma**2))
        g = max(g, 1.e-50) #avoid pesky NAN issues
        return g

    def move(self,turn,forward):
        """
        turn: variable describing the change in heading (radians)
        forward: robots present velocity
        """
        if forward < 0:
            raise ValueError('Robot can only move forward')

        #turn, and add randomness to the command
        hdg = self.hdg + float(turn) + random.gauss(0.0,self.turn_noise)
        hdg %= 2*np.pi

        dist = float(forward) + random.gauss(0.0,self.forward_noise)
        vel = self.vel + dist

        #Define x and y motion based upon new bearing relative to the x axis
        x = self.x + (cos(hdg)*dist)
        y = self.y + (sin(hdg)*dist)
        x %= self.world_size #cyclic truncate
        y %= self.world_size

        #set particles
        res = robot()
        res.set_params(self.N,self.world_size,self.landmarks)
        res.set(x,y,vel,hdg) # changes particle's position to the new location
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res

    def __repr__(self):
        return '[x=%.6s y=%.6s vel=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.vel) ,str(self.hdg))

def create_uniform_particles(N,fnoise,tnoise,snoise,v_init,world_size,landmarks):
    p = [] # list of particles

    for i in range(N): #create a list of particles (uniformly distributed)
        rand_posx =  random.uniform(0.0,1.0)*world_size
        rand_posy = random.uniform(0.0,1.0)*world_size
        rand_vel = v_init + random.gauss(0.0,fnoise) #gauss dist for initial vel
        rand_hdg = random.uniform(0.0,1.0)*2.0*np.pi # relative to x axis
        r = robot()
        r.set_params(N,world_size,landmarks)
        r.set_noise(fnoise,tnoise,snoise)
        r.set(rand_posx,rand_posy,rand_vel,rand_hdg)
        p.append(r)
    return p

def create_gaussian_particles(bot,N,fnoise,tnoise,snoise,std_dev,world_size,landmarks):
    """
    Params:
    ---------
    bot: the true initial robot to center the gaussian dist of particles
    N: number of particles
    fnoise: forward noise
    tnoise: turn noise
    snoise: sensing noise
    std_dev: the standard deviation of the particle distribution
    world_size: size of the available area
    landmarks: measurement landmarks

    returns:
    p: list of particles
    """
    p = []
    for i in range(N):
        rand_posx =  random.gauss(bot.x,std_dev)
        rand_posx %= world_size #ensure particles stay within world
        rand_posy = random.gauss(bot.y,std_dev)
        rand_posy %= world_size
        rand_vel = bot.vel + random.gauss(0.0,fnoise) #gauss dist for initial vel
        rand_hdg = random.uniform(0.0,1.0)*2.0*np.pi # relative to x axis
        r = robot()
        r.set_params(N,world_size,landmarks)
        r.set_noise(fnoise,tnoise,snoise)
        r.set(rand_posx,rand_posy,rand_vel,rand_hdg)
        p.append(r)
    return p

#Function to calculate the effective sample size
def neff(weights):
        return 1. / np.sum(np.square(weights))

#function to estimate the state, not appropriate for multi-modal
def estimate(weights,particles):
        """ returns mean and variance """
        state_inter=[]

        for p in range(len(particles)):
            state_inter.append([particles[p].x,particles[p].y,particles[p].vel,particles[p].hdg])
            # print(state_inter[p])
        state = np.asarray(state_inter)

        #calculate the mean estimate of the state
        mu = np.average(state, weights=weights, axis=0) # should contain x,y, and heading
        cov = np.cov(state,rowvar=False)

        return mu, cov

def PRMSE(truth,mean_estimate):#pass the list of truth positions and
                               #list of lists containing mean estimates
    n = len(mean_estimate) #number of trials ran
    t = len(mean_estimate[0]) #total time steps


    PRMSE = []
    x_diffsq = [[] for i in range(n)]
    diffsq = [[] for i in range(t)]
    dist = [[] for i in range(n)]

    for i in range(n): #compare each set of mean estimates against the truth, sum, and average
        for j in range(t):
            dist[i].append(sqrt((mean_estimate[i][j][0]-truth[j+1][0])**2 + (mean_estimate[i][j][1]-truth[j+1][1])**2))

    for j in range(t):
        for i in range(n):
            #restructure for summation
            diffsq[j].append(dist[i][j])
        err_sum = 0
        err_sum = np.sum(diffsq[j])
        RMSE = sqrt(err_sum/n)
        PRMSE.append(RMSE)
    return PRMSE

#---------------------------RESAMPLING METHODS-------------------------------#

#Pass a single integer paramter correlating to the KVP in methods
def resample(N,weights,particles,select):
    """
        Function to map a dictionary of available resampling methods

        N: number of particles
        weights: normalized particle weights
        particles: list of particles
        select: key to dict of desired method
    """
    p_new = []
    if select == 1:
        p_new = systematic_resample(N,weights,particles)
    elif select == 2:
        p_new = RS_resample(N,weights,particles)
    else:
        print("Not a valid resampling method")

    return p_new

def systematic_resample(N,weights,particles):

    p_new = []
    index = int(random.random()*N)
    beta = 0.0
    maxw = max(weights)

    for i in range(N):
        beta += random.random()*2.0*maxw
        #print(index,weights[index],beta)

        while beta > weights[index]:
            beta -= weights[index]
            index = (index + 1)% N
        p_new.append(particles[index])
    particles = p_new

    return particles

# Residual systematic
def RS_resample(N,weights, particles):

    p_new = []
    index = [0]*N #Initialize index array
    U = random.random()/N #Generate a random number between 0 and 1/N
    i  = 0
    j = -1

    while j < N-1:
        j += 1
        Ns = floor(N*(weights[j]-U))+1
        counter = 1;
        while counter <= Ns:
            index[i] = j
            i += 1
            counter += 1
        U = U + Ns/N - weights[j]

    for i in range(len(index)):
        p_new.append(particles[index[i]])
    particles = p_new

    return particles

def ParticleState(particle):
    px,py,pvel,phdg = particle.x,particle.y,particle.vel,particle.hdg
    pState = np.asarray([px,py,pvel,phdg])
    return pState

#Generate pseudo time intervals
def GenerateLambda():

    delta_lambda_ratio = 1.2
    nLambda = 29 #number of exp spaced step sizes
    lambda_intervals = []
    for i in range(nLambda):
        lambda_intervals.append(i)

    lambda_1 = (1-delta_lambda_ratio)/(1-delta_lambda_ratio**nLambda)
    for i in range(nLambda):
        lambda_intervals[i] = lambda_1*(delta_lambda_ratio**i)

    return lambda_intervals

def caculate_flow_params(est,P,H,R,z,lam):
    """
    Calculate the values for A and b for the given particle

    params:
    ----------
    est: predicted mean state estimate
    P: Covariance matrix generated from particles
    H: Measurement matrix (numpy array)
    R: Noise error matrix (numpy array)
    z: current measurement vector(numpy array)
    lam: psuedo time interval

    returns:
    ----------
    A: Flow parameter
    b: Flow parameter
    """
    A = -0.5*dot3(dot(P,H.T),inv(lam*dot3(H,P,H.T)+R),H)
    b = dot((eye(4)+(2*lam*A)),dot3((eye(4)+(lam*A)),dot3(P,H.T,inv(R)),z)+dot(A,est))
    return A,b

def h_jacobian(est, landmarks):
    #Return the jacobian of the measurement matrix as a numpy arrray

    H = []

    for i in range(len(landmarks)):
        x_dif = est[0]-landmarks[i][0]
        y_dif = est[1]-landmarks[i][1]
        x_lin = x_dif/np.sqrt(x_dif**2+y_dif**2)
        y_lin = y_dif/np.sqrt(x_dif**2+y_dif**2)
        h_inter = [x_lin,y_lin,0,0]
        H.append(h_inter)
    H = np.asarray(H)
    print(H)
    input("observe jacobian")

    return H
