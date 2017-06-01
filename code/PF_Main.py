""" Robot localization simulation with landmarks

Development sponsor: Air Force Research Laboratory ATR Program
Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: May 2017"""

import Robot
import numpy as np
from math import *
import random
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.stats
#------------------------------------------------#
def visualize(robot, step, p , pr, weights):
    """
    robot: current robot object
    step: current step
    p: list of particles
    pr: list of resampled particles
    weights: particle weights  """

    plt.figure("Simple Robot", figsize=(15.,15.))
    plt.title('Particle Filter, step '+ str(step))

    grid = [0,world_size,0,world_size]
    plt.axis(grid)
    plt.grid(b=True, which='major',color='0.75', linestyle='--')
    plt.xticks([i for i in range(0,int(world_size),5)])
    plt.yticks([i for i in range(0,int(world_size),5)])

    #draw particles
    for ind in range(len(p)):

        #particles (ornage)
        circle = plt.Circle((p[ind].x,p[ind].y),1., facecolor='#ffb266', edgecolor='#994c00', alpha=0.5)
        plt.gca().add_patch(circle)

        #particles orientation
        arrow = plt.Arrow(p[ind].x,p[ind].y,2*cos(p[ind].orientation),2*sin(p[ind].orientation),
                            alpha=1., facecolor='#994c00', edgecolor='#994c00')
        plt.gca().add_patch(arrow)

    #draw resampled particles
    for ind in range(len(pr)):
        # particles (green)
        circle = plt.Circle((pr[ind].x, pr[ind].y), 1., facecolor='#66ff66', edgecolor='#009900', alpha=0.5)
        plt.gca().add_patch(circle)

        # particle's orientation
        arrow = plt.Arrow(pr[ind].x, pr[ind].y, 2*cos(pr[ind].orientation), 2*sin(pr[ind].orientation),
                            alpha=1., facecolor='#006600', edgecolor='#006600')
        plt.gca().add_patch(arrow)

    # fixed landmarks of known locations (red)
    for lm in Robot.landmarks:
        circle = plt.Circle((lm[0], lm[1]), 1., facecolor='#cc0000', edgecolor='#330000')
        plt.gca().add_patch(circle)

    # robot's location (blue)
    circle = plt.Circle((robot.x, robot.y), 1., facecolor='#6666ff', edgecolor='#0000cc')
    plt.gca().add_patch(circle)

    # robot's orientation
    arrow = plt.Arrow(robot.x, robot.y, 2*cos(robot.orientation), 2*sin(robot.orientation), alpha=0.5, facecolor='#000000', edgecolor='#000000')
    plt.gca().add_patch(arrow)


    plt.savefig("output/figure_" + str(step) + ".png")

    plt.close()
#--------------------------------------------------------------------------#


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
#         fnoise = float(input("Input desired forward noise parameter: "))
#         tnoise = float(input("Input desired turning noise parameter: "))
#         snoise = float(input("Input desired sensing noise parameter: "))
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
#
#     except ValueError:
#         print("ERROR: Number of iterations must be an integer")
#         #Possibly add exception for error not in the correct range of values?
#     else:
#         break

n = 1000
fnoise = 0.05
tnoise = 0.05
snoise = 5.0
steps = 10

p = [] # list of particles
world_size = Robot.world_size
Bot = Robot.robot()
Bot.set(50,50,np.pi/2)
z = Bot.sense()

for i in range(n): #create a list of particles (uniformly distributed)
    r = Robot.robot()
    r.set_noise(fnoise,tnoise,snoise)
    p.append(r)

for t in range(steps):
    #initialize the robot that we would like to track
    Bot = Bot.move(0.1,5.0)
    z = Bot.sense()

    p2=[]

    for i in range(n):
        p2.append(p[i].move(0.1,5.0))
    p = p2

    w = []
    #generate particle weights based on measurement
    for i in range(n):
        w.append(p[i].measurement_prob(z))

    #Resampling with a sample probability proportional to importance weight
    p3 = []

    index = int(random.random()*n)
    beta = 0.0
    mw=max(w)

    for i in range(n):
        beta += random.random()*2.0*mw

        while beta > w[index]:
            beta -= w[index]
            index = (index + 1) % n

        p3.append(p[index])

    p = p3

    print( 'Step =',t,', Evaluation = ', Bot.eval(Bot,p))
    #if (t%10) == 0:
    visualize(Bot,t,p2,p,w)
print('p = ',len(p) )
