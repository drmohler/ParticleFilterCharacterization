""" Robot simulation for particle filtering

Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: Summer 2017"""

import numpy as np
from numpy.random import randn, random, uniform
from numpy import dot,eye
from numpy.linalg import inv
# from filterpy.common import dot3
from math import *
import random
import matplotlib.pyplot as plt
import scipy.stats

from cmath import rect, phase

class robot:
    def __init__(self, ):
        self.x = 0
        self.y = 0
        self.vel = 0.0
        self.hdg = 0.0
        self.world_size = 100
        self.landmarks = [[0,0]]
        self.model_noise = 0.0
        self.vel_noise = 0.0
        self.turn_noise = 0.0
        self.sense_noise = 0.0
    
    def copy(self):
        to_return = robot()
        to_return.x = self.x
        to_return.y = self.y
        to_return.vel = self.vel
        to_return.hdg = self.hdg
        to_return.world_size = self.world_size
        to_return.landmarks = self.landmarks
        to_return.model_noise = self.model_noise
        to_return.vel_noise = self.vel_noise
        to_return.turn_noise = self.turn_noise
        to_return.sense_noise = self.sense_noise
        return to_return


    def get_state_length(self):
        return 4

    def state(self):
        return np.array([self.x, self.y, self.vel, self.hdg])

    def get_meas_length(self):
        return len(self.landmarks)

    def set_params(self,new_world_size,new_landmarks):
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

    def set_noise(self, new_vel_noise, new_turn_noise, new_model_noise, new_sense_noise):
        """ Set the noise parameters, changing them is often useful in particle filters
        :param new_forward_noise: new noise value for the forward movement
        :param new_turn_noise:    new noise value for the turn
        :param new_sense_noise:  new noise value for the sensing
        """

        self.vel_noise = float(new_vel_noise)
        self.turn_noise = float(new_turn_noise)
        self.model_noise = float(new_model_noise)
        self.sense_noise = float(new_sense_noise)

    def sense(self, add_noise=None):
        """ Sense the environment: calculate distances to landmarks
        :return measured distances to the known landmarks
        : by default, add noise to these values
        """
        z = []
        for i in range(len(self.landmarks)):
            dist = sqrt((self.x - self.landmarks[i][0]) ** 2 + (self.y - self.landmarks[i][1]) ** 2)
            if (add_noise is None) or add_noise==True:
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

    def move(self, turn, accel):
        """
        turn: change in heading (radians)
        forward: robots present velocity
        """
        #turn, and add randomness to the command
        self.hdg = self.hdg + float(turn) + random.gauss(0.0,self.turn_noise)
        self.hdg %= 2*np.pi

        self.vel = self.vel + accel + random.gauss(0.0, self.vel_noise)
        if self.vel < 0:
            raise ValueError('Robot can only move forward')


        dist = self.vel


        #Define x and y motion based upon new bearing relative to the x axis
        self.x = self.x + (cos(self.hdg)*dist) + random.gauss(0.0,self.model_noise)
        self.y = self.y + (sin(self.hdg)*dist) + random.gauss(0.0,self.model_noise)
        
        #TODO:  Put a warning if it goes out of the world size...
        # x %= self.world_size #cyclic truncate
        # y %= self.world_size
        return self

    def __repr__(self):
        return '[x=%.6s y=%.6s vel=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.vel) ,str(self.hdg))

def create_uniform_particles(N,fnoise,tnoise,snoise,v_init,world_size,landmarks):
    """
    Params:
    ---------
    N: number of particles
    fnoise: forward noise
    tnoise: turn noise
    snoise: sensing noise
    v_init: initial velocity of the true robot
    world_size: size of the available area
    landmarks: measurement landmarks

    returns:
    p: list of particles
    """
    p = [] # list of particles
    for i in range(N): #create a list of particles (uniformly distributed)
        rand_posx =  random.uniform(0.0,1.0)*world_size
        rand_posy = random.uniform(0.0,1.0)*world_size
        rand_vel = v_init + random.gauss(0.0,fnoise) #gauss dist for initial vel
        rand_hdg = random.uniform(0.0,1.0)*2.0*np.pi # relative to x axis
        r = robot()
        r.set_params(world_size,landmarks)
        #TODO:  Make this follow the inputs.  The PF_main doesn't do this right now (it is hard coded) so I just make this the same for now
        r.set_noise(0.1,np.radians(2.5),.1,1.0)

#        r.set_noise(fnoise,tnoise,snoise)
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
        #Just testing...
        rand_hdg = random.uniform(0.0,2.0*np.pi)
        if rand_hdg < 0:
            rand_hdg += 2.0*np.pi
        #rand_hdg = random.uniform(0.0,1.0)*2.0*np.pi # relative to x axis
        r = robot()
        r.set_params(world_size,landmarks)
        #TODO:  Make this follow the inputs.  The PF_main doesn't do this right now (it is hard coded) so I just make this the same for now
        r.set_noise(0.1,np.radians(2.5),.1,1.0)
        r.set(rand_posx,rand_posy,rand_vel,rand_hdg)
        p.append(r)
    return p

#Function to calculate the effective sample size
def neff(weights):
        return 1. / np.sum(np.square(weights))

def mean_angle(rad, weights):
    """ Calculate the mean angle, accounting for roll over"""
    if weights==None:
        mean = phase(sum(rect(1, d) for d in rad)/len(rad))
    else:
        mean = phase(sum(rect(a,b) for a,b in zip(weights, rad)))
    #ensure positive mean angle measure
    while mean < 0:
        mean += 2*np.pi
    mean %= 2*np.pi
    return mean

def covariance(particles,mu):
    """
    Specific covariance calculations accounting for
    the fact that the heading is circular data and
    requires adjustment

    params:
    ------------
    particles: list of particles
    mu: current mean estimate of the state

    returns:
    P: covariance matrix
    """
    N = len(particles)
    mat_sum = np.zeros((4,4))
    for p in range(N):
        v = ParticleState(particles[p])
        v_shift = np.asarray(v-mu)
        if v_shift[3] > np.pi:
            v_shift[3] -= 2*(np.pi)
        if v_shift[3] < -np.pi:
            v_shift[3] += 2*(np.pi)
        v_shiftT = np.transpose(v_shift)
        mult = np.outer(v_shift,v_shiftT)
        mat_mults = np.asmatrix(mult)
        mat_sum += mat_mults
    cov = (1/(N-1))*mat_sum
    return cov

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
        mu[3] = mean_angle(state[:,3], weights)

        cov = covariance(particles,mu)

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
        p_new.append(particles[index].copy())
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
        p_new.append(particles[index[i]].copy())
    particles = p_new

    return particles

def ParticleState(particle):
    """Return the current particle state as numpy array"""
    px,py,pvel,phdg = particle.x,particle.y,particle.vel,particle.hdg
    pState = np.asarray([px,py,pvel,phdg])
    return pState

def GenerateLambda():
    """Exponentially distributed pseudo time steps"""
    delta_lambda_ratio = 1.2
    nLambda = 29 #number of exp spaced step sizes
    lambda_intervals = []
    for i in range(nLambda):
        lambda_intervals.append(i)

    lambda_1 = (1-delta_lambda_ratio)/(1-delta_lambda_ratio**nLambda)
    for i in range(nLambda):
        lambda_intervals[i] = lambda_1*(delta_lambda_ratio**i)

    return lambda_intervals

def h_jacobian(est, landmarks):
    """Return the jacobian of the measurement matrix as a numpy arrray
    note: specific to the distance style measurements used. If different
    measurements are used this will require modification. """
    H = []
    for i in range(len(landmarks)):
        x_dif = est[0]-landmarks[i][0]
        y_dif = est[1]-landmarks[i][1]
        x_lin = x_dif/np.sqrt(x_dif**2+y_dif**2)
        y_lin = y_dif/np.sqrt(x_dif**2+y_dif**2)
        h_inter = [x_lin,y_lin,0,0]
        H.append(h_inter)
    H = np.asarray(H)
    return H

def dot4(A,B,C,D):
    """ Returns the matrix multiplication of A*B*C*D"""
    return dot(A, dot(B, dot(C,D)))

def dot3(A,B,C):
    """ Returns the matrix multiplication of A*B*C"""
    return dot(A, dot(B, C))

def caculate_flow_params(est,P,H,R,z,pmeasure,lam):
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
    HPHt = dot3(H,P,H.T)
    A_inv = inv((lam*HPHt) + R)

    A = (-0.5)*dot4(P,H.T,A_inv,H)

    I2 = eye(4)+(2*lam*A)
    I1 = eye(4)+(lam*A)
    Ri = inv(R)

    IlamAHPHt = dot4(I1,P,H.T,Ri)
    err = pmeasure - dot(H,est)
    zdif = z - err
    b5 = dot(IlamAHPHt,zdif)
    Ax = dot(A,est)

    in_sum = b5 + Ax
    b = dot(I2,in_sum)
    return A,b

def EnKF_update(particles, meas):
    N=len(particles)
    pred_meas=[]

    #First, compute the predicted measurements
    for i in range(N):
        pred_meas.append(particles[i].sense(False))
    pred_meas_mean = np.mean(pred_meas, 0)
    #Second, compute K
    state_mean = estimate(None,particles)[0]
    PH = np.zeros((particles[i].get_state_length(), particles[i].get_meas_length() ))
    HPH = np.zeros((particles[i].get_meas_length(), particles[i].get_meas_length() ))
    for i in range(N): 
        de_robot = particles[i]
        meas_diff = pred_meas[i] - pred_meas_mean
        state_diff = de_robot.state() - state_mean
        if state_diff[3] > np.pi:
            state_diff[3] -= 2*np.pi
        if state_diff[3] < -np.pi:
            state_diff[3] += 2*np.pi
        PH += np.outer(de_robot.state() - state_mean, meas_diff)
        HPH += np.outer(meas_diff, meas_diff)
    PH = PH / (N-1)
    HPH = HPH / (N-1)
    K = np.dot(PH, inv(HPH + np.eye(len(meas))* de_robot.sense_noise* de_robot.sense_noise))
    # print('HPH is',HPH)
    # print('PH is',PH)
    # print('K is ',K)
    #Now to apply K to all the particles
    out_particles=[]
    for i in range(N):
        delta_state = np.dot(K, (meas-pred_meas[i] + particles[i].sense_noise * randn(len(meas))))
        # print('Curr state is',particles[i].state())
        # print("delta_state is ",delta_state)
        next_state = particles[i].state() + delta_state
        if (next_state[3] < 0):
            next_state[3] += 2*np.pi
        if (next_state[3] > 2*np.pi):
            next_state[3] -= 2*np.pi
        going_out = particles[i].copy()
        going_out.set( next_state[0], next_state[1], next_state[2], next_state[3] ) 
        out_particles.append(going_out)
    return out_particles