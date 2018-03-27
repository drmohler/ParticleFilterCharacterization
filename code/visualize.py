""" Particle Filter visualization

Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: Summer 2017"""


import numpy as np
from math import *
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats

#---------------------FUNCTIONS FOR DISPLAY ----------------------------------#
class vis:
    def __init__(self, world_size, landmarks):
        self.world_size = world_size
        self.landmarks = landmarks


    def visualize(self, robot, step, p, pr, weights, estimate):
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
        plt.grid(b=True, which='major', color='0.75', linestyle='--')
        plt.xticks([i for i in range(0,int(self.world_size),int(self.world_size/10))])
        plt.yticks([i for i in range(0,int(self.world_size),int(self.world_size/10))])

        #draw particles
        for ind in range(len(p)):

            #particles (ornage)
            circle = plt.Circle((p[ind].x, p[ind].y), 1., facecolor='#ffb266', edgecolor='#994c00', alpha=0.5)
            plt.gca().add_patch(circle)

            #particles hdg
            arrow = plt.Arrow(p[ind].x, p[ind].y,2*cos(p[ind].hdg),2*sin(p[ind].hdg),
                                alpha=1., facecolor='#994c00', edgecolor='#994c00')
            plt.gca().add_patch(arrow)

        #draw resampled particles
        for ind in range(len(pr)):
            # particles (green)
            circle = plt.Circle((pr[ind].x, pr[ind].y), 1., facecolor='#66ff66', edgecolor='#009900', alpha=0.5)
            plt.gca().add_patch(circle)

            # particle's hdg
            arrow = plt.Arrow(pr[ind].x, pr[ind].y, 2*cos(pr[ind].hdg), 2*sin(pr[ind].hdg),
                                alpha=1., facecolor='#006600', edgecolor='#006600')
            plt.gca().add_patch(arrow)

        # fixed landmarks of known locations (red)
        for lm in self.landmarks:
            circle = plt.Circle((lm[0], lm[1]), 1., facecolor='#cc0000', edgecolor='#330000')
            plt.gca().add_patch(circle)

        # robot's location (blue)
        circle = plt.Circle((robot.x, robot.y), 1., facecolor='#6666ff', edgecolor='#0000cc')
        plt.gca().add_patch(circle)

        # robot's hdg
        arrow = plt.Arrow(robot.x, robot.y, 2*cos(robot.hdg), 2*sin(robot.hdg), alpha=0.5, facecolor='#000000', edgecolor='#000000')
        plt.gca().add_patch(arrow)

        #State Estimate (unimodal), will need modification for multi-modal
        circle = plt.Circle((estimate[0], estimate[1]), 1., facecolor='#505050', edgecolor='#000000')
        plt.gca().add_patch(circle)
        arrow = plt.Arrow(estimate[0],estimate[1], 2*cos(estimate[3]), 2*sin(estimate[3]), alpha=0.5, facecolor='#000000', edgecolor='#000000')
        plt.gca().add_patch(arrow)


        plt.savefig("output/figure_" + str(step) + ".png")

        plt.close()

def plot_paths(true_pos,mean_estimate,save_file_name=None, plot_name=None):
    """
    Params:
    -----------------
    true_pos: true position of the robot
    mean_estimate: list of mean estimate coordinates from PF and PFPF
    """
    fig, ax = plt.subplots()
    trials = len(mean_estimate[0])

    #for x,y in true_pos:
    xt_pos = [i[0] for i in true_pos]
    yt_pos = [i[1] for i in true_pos]

    plt.plot(xt_pos,yt_pos,'-o', color="blue", markeredgecolor="blue", label="Truth")
    #TODO:  Should arrange colors so the same run has the same color, then have the markers (triangle, x, etc.) be the only difference and consistent across type
    for tr in range(trials):
        #Note that if we have more than 10 trials, will have the same plots over again
        color_string ="C"+str(tr%10)
        trial_label = "Flow Trial - " + str(tr+1)
        trial_label_std = "Std. Trial - " + str(tr+1)
        trial_label_enkf = "EnKF Trial - " + str(tr+1)
        trial_label_aux = "Aux Trial - " + str(tr+1)

        xe_pos_flow = [i[0] for i in mean_estimate[0][tr]]
        ye_pos_flow = [i[1] for i in mean_estimate[0][tr]]
        plt.plot(xe_pos_flow, ye_pos_flow, '-x', label=trial_label, color=color_string)

        if len(mean_estimate) > 1:
            xe_pos_std = [i[0] for i in mean_estimate[1][tr]]
            ye_pos_std = [i[1] for i in mean_estimate[1][tr]]
            plt.plot(xe_pos_std, ye_pos_std, '-^', label=trial_label_std, color=color_string)

        if len(mean_estimate) > 2:
            xe_pos_enkf = [i[0] for i in mean_estimate[2][tr]]
            ye_pos_enkf = [i[1] for i in mean_estimate[2][tr]]
            plt.plot(xe_pos_enkf,ye_pos_enkf,'-s' , label=trial_label_enkf, color=color_string)

        if len(mean_estimate) > 3:
            xe_pos_aux = [i[0] for i in mean_estimate[3][tr]]
            ye_pos_aux = [i[1] for i in mean_estimate[3][tr]]
            plt.plot(xe_pos_aux,ye_pos_aux,'-+' , label=trial_label_aux, color=color_string)

    plt.legend()
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    if plot_name!= None:
        plt.title(plot_name)
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('--')
    plt.show()
    if save_file_name!= None:
        plt.savefig(save_file_name)

def plot_RMSE(RMSE, save_file_name=None, plot_name=None):
    """
    Params:
    -----------------
    RMSE: List of lists containing PRMSE measurements from various
          Particle filter methods
    """
    fig, ax = plt.subplots()
    for m in range(len(RMSE)):
        if m == 0:
            method = "PFPF"
        elif m== 1:
            method = "SIR"
        elif m==2:
            method = "EnKF"
        else:
            method = "Aux"
        plt.errorbar(y=RMSE[m][0], x = range(len(RMSE[m][0])),
                     #yerr=[np.subtract(RMSE[m][0],RMSE[m][1]), np.subtract(RMSE[m][2],RMSE[m][0])], 
                     label=method)
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('RMSE (m)')
    if plot_name == None:
        plt.title("RMSE vs Time")
    else:
        plt.title(plot_name)
    ax.grid()
    gridlines = ax.get_xgridlines() + ax.get_ygridlines()
    for line in gridlines:
        line.set_linestyle('--')
    plt.show()
    if save_file_name!=None:
        plt.savefig(save_file_name)
