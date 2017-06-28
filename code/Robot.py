""" Robot simulation for particle filtering

Development sponsor: Air Force Research Laboratory ATR Program
Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: May 2017"""

import numpy as np
from numpy.random import randn, random, uniform
from math import *
import random
import matplotlib.pyplot as plt
import scipy.stats

# while True:
#     try:
#         world_size = int(input("Input size of world: "))
#
#     except ValueError:
#         print("ERROR: World size must be an integer")
#
#     else:
#         break

class robot:
    def __init__(self):
        self.x = 0 #random.random()*world_size
        self.y = 0 #random.random()*world_size
        self.orientation = 0 #random.random()*2.0*np.pi # relative to x axis
        self.world_size = 100
        self.landmarks = [[0,0]]
        self.N = 1
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set_params(self,new_N,new_world_size,new_landmarks):
        self.N = int(new_N)
        self.world_size = int(new_world_size)
        self.landmarks = new_landmarks

    def set(self,new_x,new_y,new_orientation): #place the robot at a given spot
        if new_x < 0 or new_x >= self.world_size:
            print("current world size: ", self.world_size)
            print("Bad X value: ",new_x)
            raise ValueError('X coordinate out of bounds')
        if new_y < 0 or new_y >= self.world_size:
            print("Bad Y value: ",new_y)
            raise ValueError('Y coordinate out of bounds')
        if new_orientation < 0 or new_orientation >= 2*np.pi:
            print("Bad hdg value: ",new_orientation)
            raise ValueError('Orientation must be in range [0,2*Pi]')

        self.x = float(new_x)
        self.y = float(new_y)
        self.orientation = float(new_orientation)

    def set_noise(self, new_forward_noise, new_turn_noise, new_sense_noise):
        """ Set the noise parameters, changing them is often useful in particle filters
        :param new_forward_noise: new noise value for the forward movement
        :param new_turn_noise:    new noise value for the turn
        :param new_sense_noise:  new noise value for the sensing
        """

        self.forward_noise = float(new_forward_noise)
        self.turn_noise = float(new_turn_noise)
        self.sense_noise = float(new_sense_noise)

    def sense(self):
        """ Sense the environment: calculate distances to landmarks
        :return measured distances to the known landmarks
        """

        z = []

        for i in range(len(self.landmarks)):
            dist = sqrt((self.x - self.landmarks[i][0]) ** 2 + (self.y - self.landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise) #0 mean noise, user defined standard deviation
            z.append(dist)

        return z #measure relative to each landmark

    def measurement_prob(self, measurement):
        #calculates how likely a measurement should be

        prob = 1.0
        for i in range(len(self.landmarks)):
            dist = sqrt((self.x - self.landmarks[i][0])**2 + (self.y-self.landmarks[i][1])**2)
            prob *= self.Gaussian(dist,self.sense_noise,measurement[i])
        return prob

    def move(self,turn,forward):
        if forward < 0:
            raise ValueError('Robot can only move forward')

        #turn, and add randomness to the command
        orientation = self.orientation + float(turn) + random.gauss(0.0,self.turn_noise)
        orientation %= 2*np.pi

        dist = float(forward) + random.gauss(0.0,self.forward_noise)
        #Define x and y motion based upon new bearing relative to the x axis
        x = self.x + (cos(orientation)*dist)
        y = self.y + (sin(orientation)*dist)
        x %= self.world_size #cyclic truncate
        y %= self.world_size

        #set particles
        res = robot()
        res.set_params(self.N,self.world_size,self.landmarks)
        res.set(x,y,orientation) # changes robot's position to the new location
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res

    def Gaussian(self,mu,sigma,x):
        #calculates the probability of x for 1-Dim Gaussian
        #with mean mu and variance sigma
        # gauss = scipy.stats.norm(mu,sigma)
        # return gauss
        return exp(-((mu-x)**2)/(sigma**2)/2.0)/sqrt(2.0*np.pi*(sigma **2))

    #----------------RESAMPLING METHODS--------------------------#

    def multinomial_resample(self,weights,particles):

        p = np.zeros((self.N, 3)) #second param needs to be the same as state params
        w = np.zeros(self.N)

        cumulative_sum = np.cumsum(weights)
        cumulative_sum[-1] = 1.
        for i in range(self.N):
            index = np.searchsorted(cumulative_sum, np.random.random(self.N))
            p[i] = particles[index]
            w[i] = weights[index]
        #resample according to indexes
        particles  = p
        weights = w/np.sum(weights)

        return weights, particles

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))

    def eval(self,r,p):
        sum = 0.0
        for i in range(len(p)):
            dx = (p[i].x - r.x + (self.world_size/2.0)) % self.world_size - (self.world_size/2.0)
            dy = (p[i].y - r.y + (self.world_size/2.0)) % self.world_size - (self.world_size/2.0)
            err = sqrt(dx*dx+dy*dy)
            sum+= err
        return sum/float(len(p))

def create_uniform_particles(N,fnoise,tnoise,snoise,world_size,landmarks):
    p = [] # list of particles

    for i in range(N): #create a list of particles (uniformly distributed)
        rand_posx =  random.random()*world_size
        rand_posy = random.random()*world_size
        rand_hdg = random.random()*2.0*np.pi # relative to x axis
        r = robot()
        r.set_params(N,world_size,landmarks)
        r.set_noise(fnoise,tnoise,snoise)
        r.set(rand_posx,rand_posy,rand_hdg)
        p.append(r)
    return p

#print("x = ", rand_posx, ", Y = ", rand_posy, ", Heading = ", rand_hdg)
# def create_gaussian_particles(N,fnoise,tnoise,snoise,world_size,landmarks,mean,var):
#     p = [] # list of particles
#
#     for i in range(N): #create a list of particles (uniformly distributed)
#         rand_posx =  mean[0] + randn(self.N)*var[0] #np.random.standard_normal()*world_size
#         rand_posy = np.random.standard_normal()*world_size
#         rand_hdg = np.random.standard_normal()*2.0*np.pi # relative to x axis
#         r = robot()
#         r.set_params(N,world_size,landmarks)
#         r.set_noise(fnoise,tnoise,snoise)
#         r.set(rand_posx,rand_posy,rand_hdg)
#
#         p.append(r)
#     return p
#
#     # self.particles[:, 0] = mean[0] + randn(self.N)*var[0]
#     # self.particles[:, 1] = mean[1] + randn(self.N)*var[1]
#     # self.particles[:, 2] = mean[2] + randn(self.N)*var[2]
#     # self.particles[:, 2] %= 2 * np.pi

#Function to calculate the effective sample size
def neff(weights):
        return 1. / np.sum(np.square(weights))

#function to estimate the state, not appropriate for multi-modal
def estimate(weights,particles):
        """ returns mean and variance """
        pos=[]
        for p in range(len(particles)):
            pos.append([particles[p].x,particles[p].y,particles[p].orientation])
            # pos[p,0] = particles[p].x#,particles[p].y,particles[p].orientation
            # pos[p,1] = particles[p].y
            # pos[p,2] = particles[p].orientation
        mu = np.average(pos, weights=weights, axis=0) # should contain x,y, and heading
        var = np.average((pos - mu)**2, weights=weights, axis=0)

        return mu, var

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
            # if j == t-1:
            #     print("estimate: (", mean_estimate[i][j][0],mean_estimate[i][j][1], ") truth: (", truth[j+1][0],truth[j+1][1],")")
            #     print("dist: ", dist[i][j])

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
    # print("Sampling Index: ",index)
    return particles
