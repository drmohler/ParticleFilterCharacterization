""" Robot simulation for particle filtering

Development sponsor: Air Force Research Laboratory ATR Program
Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: May 2017"""

import numpy as np
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
world_size = 100.0
landmarks = [[20.0,20.0], [20.0,80.0], [20.0,50.0],
            [50.0,20.0], [50.0,80.0], [80.0,80.0],
            [80.0,20.0], [80.0,50.0]]

class robot:
    def __init__(self):
        self.x = random.random()*world_size
        self.y = random.random()*world_size
        self.orientation = random.random()*2.0*np.pi # relative to x axis
        # self.world_size = world_size
        # self.landmarks = []
        self.forward_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0

    def set(self,new_x,new_y,new_orientation):
        if new_x < 0 or new_x >= world_size:
            raise ValueError('X coordinate out of bounds')
        if new_y < 0 or new_y >= world_size:
            raise ValueError('Y coordinate out of bounds')
        if new_orientation < 0 or new_orientation >= 2*np.pi:
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

        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0]) ** 2 + (self.y - landmarks[i][1]) ** 2)
            dist += random.gauss(0.0, self.sense_noise)
            z.append(dist)

        return z

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
        x %= world_size #cyclic truncate
        y %= world_size

        #set particles
        res = robot()
        res.set(x,y,orientation)
        res.set_noise(self.forward_noise, self.turn_noise, self.sense_noise)
        return res

    def Gaussian(self,mu,sigma,x):
        #calculates the probability of x for 1-Dim Gaussian
        #with mean mu and variance sigma
        # gauss = scipy.stats.norm(mu,sigma)
        # return gauss
        return exp(-((mu-x)**2)/(sigma**2)/2.0)/sqrt(2.0*np.pi*(sigma **2))



    def measurement_prob(self, measurement):
        #calculates how likely a measurement should be

        prob = 1.0
        for i in range(len(landmarks)):
            dist = sqrt((self.x - landmarks[i][0])**2 + (self.y-landmarks[i][1])**2)
            prob *= self.Gaussian(dist,self.sense_noise,measurement[i])
        return prob

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))

    def eval(self,r,p):
        sum = 0.0
        for i in range(len(p)):
            dx = (p[i].x - r.x + (world_size/2.0)) % world_size - (world_size/2.0)
            dy = (p[i].y - r.y + (world_size/2.0)) % world_size - (world_size/2.0)
            err = sqrt(dx*dx+dy*dy)
            sum+= err
        return sum/float(len(p))
