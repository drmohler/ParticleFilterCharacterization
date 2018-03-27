""" Top level file to run multiple particle filters and
    compare the results

Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: Summer 2017
"""
import math
import PF_Main
# import visualize

# #------------------------------USER INPUTS-------------------------------#
# #resample_methods = {1:"Systematic resample", 2:"Residual systematic resample"}

# #NOTE: KNOWN ISSUE WITH MULTIPLE RESAMPLING METHODS. FOR PROPER OPERATION
# #LIMIT TO A SINGLE METHOD.

# # print("--------------------------------------")
# # print("Available Resampling Methods: ")
# # print()
# # for key, value in resample_methods.items():
# #     print('{}: {}'.format(key, value))
# # print("--------------------------------------")
# # methods = []
# # method = input("Choose a resampling method or type 'all': ")
# #
# # if method == 'all':
# #     for i in resample_methods.keys():
# #         methods.append(i)
# #
# # if method != "" and method !='all':
# #     if int(method) in resample_methods and method != 'all':
# #         methods.append(int(method))
# #     else:
# #         print("Key not in dictionary")
# #
# # while method != "" and method != "all":
# #     method =  input("Choose another resampling method or press enter to continue: ")
# #     print()
# #     if method != "":
# #         if int(method) in resample_methods and int(method) not in methods:
# #             methods.append(int(method))
# #         else:
# #             print("Resampling method not available.")
# #             print()
# #     # if all()

# while True:
#     try:
#         n = int(input("Input desired number of particles: "))

#     except ValueError:
#         print("ERROR: Number of particles must be an integer")

#     else:
#         break

# # while True:
# #     try:
# #         fnoise = float(input("Input forward noise parameter: "))
# #         tnoise = float(input("Input turning noise parameter: "))
# #         tnoise = math.radians(tnoise)
# #         snoise = float(input("Input sensing noise parameter: "))

# #     except ValueError:
# #         print("ERROR: noise parameters must be an floating point values")

# #     else:
# #         break
# fnoise = 0.1
# tnoise = math.radians(2.5)
# snoise = 1.0

# while True:
#     try:
#         steps = int(input("Input desired iterations: "))
#         trials = int(input("Input desired trials (min. 1): "))

#     except ValueError:
#         print("ERROR: Number of iterations and trials must integers")
#         #Possibly add exception for error not in the correct range of values?
#     else:
#         break


# vis = input("Particle visualzations [y/n]?: ")
# graphics = bool(vis == 'y')

# #DEBUGGING VARIABLE VALUES
# # n = 500
# # fnoise = 0.1
# # tnoise = np.radians(2.5)
# # snoise = 1
# # steps = 100
# # trials = 2
# # graphics = False
# methods = [2]

# #run the particle filter using resampling methods
# # est,truth,PRMSE = PF_Main.ParticleFilt(n,fnoise,tnoise,snoise,steps,trials,methods,graphics)

# #Particle filter applying particle flow
# # est, truth,PRMSE = PF_Main.PFPF(n,fnoise,tnoise,snoise,steps,trials,graphics)

# #Need to write out to a file:
# # * True state
# # * Estimated state for each filter type and
# # ** each trial and
# # ** timestep

# #Should then have a class that reads in and computes/visualized desired metrics

# est, truth, PRMSE = PF_Main.two_filters(n, fnoise, tnoise, snoise, steps, trials, methods, graphics)
# visualize.plot_RMSE(PRMSE)
# visualize.plot_paths(truth, est)

###### This one is to try to find a case where the aux is doing significantly worse
#than the standard

#A new version that just runs through a loop, storing files as it goes.
#All visualization / analysis to come later
fnoise = 0.1
tnoise = math.radians(1)
snoise = 1.0
vnoise = 0.15 #velocity
methods = [2]
steps = 50
trials = 1
graphics = True
start_seed = 857601
# num_parts_set = [50, 100, 200, 400]
num_parts_set = [200]
look_at_list=[]
for n in num_parts_set:
    for start_loc in range(1) :
        save_file_name = None  #'output/Results6_bigHeadingNoise_'+str(start_loc)+'random_start_'+str(n)+'particles_'+str(trials)+'trials_'+str(steps)+'steps.npz'
        my_seed = start_seed + start_loc*13
        res=PF_Main.two_filters(n, fnoise, vnoise, tnoise, snoise, steps, 
                                trials, methods, 
                                graphics, 
                                result_save_filename=save_file_name, 
                                random_seed = my_seed)
        std_err = res[2][1][2]
        aux_err = res[2][3][2]
        print('With seed',my_seed,'std error is:',std_err[-1], 'aux error is:', aux_err[-1])
        if (aux_err[-1] > 5.0 and (aux_err[-1] > std_err[-1])):
            look_at_list.append(my_seed)
        print('std_err is:',std_err)
        print('aux_err is:',aux_err)
print('These may be some good seeds to look at...')
print(look_at_list)

import matplotlib
import matplotlib.pyplot as plt
plt.plot(aux_err)
plt.plot(std_err)
plt.show()

#Big turn noise = 10 degrees
#Normal equals 1?

########  How I was running things before

# #A new version that just runs through a loop, storing files as it goes.
# #All visualization / analysis to come later
# fnoise = 0.1
# tnoise = math.radians(10)
# snoise = 1.0
# vnoise = 0.15 #velocity
# methods = [2]
# steps = 150
# trials = 1
# graphics = True
# # num_parts_set = [50, 100, 200, 400]
# num_parts_set = [400]

# for n in num_parts_set:
#     for start_loc in range(1) :
#         save_file_name = 'output/Results6_bigHeadingNoise_'+str(start_loc)+'random_start_'+str(n)+'particles_'+str(trials)+'trials_'+str(steps)+'steps.npz'
#         PF_Main.two_filters(n, fnoise, vnoise, tnoise, snoise, steps, 
#                             trials, methods, graphics, save_file_name)

# #Big turn noise = 10 degrees
# #Normal equals 1?