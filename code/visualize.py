""" Particle Filter visualization

Development sponsor: Air Force Research Laboratory ATR Program
Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: May 2017"""


import numpy as np
from numpy.random import randn, random, uniform
from math import *
import random
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

#---------------------FUNCTIONS FOR DISPLAY ----------------------------------#
class vis:
    def __init__(self,world_size,landmarks):
        self.world_size = world_size
        self.landmarks = landmarks


    def visualize(self,robot, step, p , pr, weights,estimate):
        """
        robot: current robot object
        step: current step
        p: list of particles
        pr: list of resampled particles
        weights: particle weights  """

        plt.figure("Simple Robot", figsize=(15.,15.))
        plt.title('Particle Filter, step '+ str(step))

        grid = [0,self.world_size,0,self.world_size]
        plt.axis(grid)
        plt.grid(b=True, which='major',color='0.75', linestyle='--')
        plt.xticks([i for i in range(0,int(self.world_size),50)])
        plt.yticks([i for i in range(0,int(self.world_size),50)])

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
        for lm in self.landmarks:
            circle = plt.Circle((lm[0], lm[1]), 1., facecolor='#cc0000', edgecolor='#330000')
            plt.gca().add_patch(circle)

        # robot's location (blue)
        circle = plt.Circle((robot.x, robot.y), 1., facecolor='#6666ff', edgecolor='#0000cc')
        plt.gca().add_patch(circle)

        # robot's orientation
        arrow = plt.Arrow(robot.x, robot.y, 2*cos(robot.orientation), 2*sin(robot.orientation), alpha=0.5, facecolor='#000000', edgecolor='#000000')
        plt.gca().add_patch(arrow)

        #State Estimate (unimodal), will need modification for multi-modal
        circle = plt.Circle((estimate[0], estimate[1]), 1., facecolor='#505050', edgecolor='#000000')
        plt.gca().add_patch(circle)
        arrow = plt.Arrow(estimate[0],estimate[1], 2*cos(estimate[2]), 2*sin(estimate[2]), alpha=0.5, facecolor='#000000', edgecolor='#000000')
        plt.gca().add_patch(arrow)


        plt.savefig("output/figure_" + str(step) + ".png")

        plt.close()
    #-----------------------------------------------------------------------------#
