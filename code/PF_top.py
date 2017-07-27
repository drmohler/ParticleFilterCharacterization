""" Top level file to run multiple particle filters and
    compare the resutlst

Development sponsor: Air Force Research Laboratory ATR Summer Program
Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: June 2017
"""

import PF_Main
import numpy as np
import visualize



#------------------------------USER INPUTS-------------------------------#
resample_methods = {1:"Systematic resample", 2:"Residual systematic resample"}

print("--------------------------------------")
print("Available Resampling Methods: ")
print()
for key, value in resample_methods.items():
    print('{}: {}'.format(key, value))
print("--------------------------------------")

methods = []
# method = input("Choose a resampling method or type 'all': ")
#
# if method == 'all':
#     for i in resample_methods.keys():
#         methods.append(i)
#
# if method != "" and method !='all':
#     if int(method) in resample_methods and method != 'all':
#         methods.append(int(method))
#     else:
#         print("Key not in dictionary")
#
# while method != "" and method != "all":
#     method =  input("Choose another resampling method or press enter to continue: ")
#     print()
#     if method != "":
#         if int(method) in resample_methods and int(method) not in methods:
#             methods.append(int(method))
#         else:
#             print("Resampling method not available.")
#             print()
#     # if all()

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
#         fnoise = float(input("Input forward noise parameter: "))
#         tnoise = float(input("Input turning noise parameter: "))
#         snoise = float(input("Input sensing noise parameter: "))
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
#         trials = int(input("Input desired trials (min. 1): "))
#
#     except ValueError:
#         print("ERROR: Number of iterations and trials must integers")
#         #Possibly add exception for error not in the correct range of values?
#     else:
#         break

#DEBUGGING VARIABLE VALUES
n = 500
fnoise = 0.1
tnoise = np.radians(2.5)
snoise = 1
steps = 100
trials = 2
graphics = False
methods = [2]

#run the particle filter for each of the chosen resampling methods
# est,truth,PRMSE = PF_Main.ParticleFilt(n,fnoise,tnoise,snoise,steps,trials,methods,graphics)
# est2,true2,RMSE2 = PF_Main.ParticleFilt(n,fnoise,tnoise,snoise,steps,trials,methods,graphics)

# PRMSE = [RMSE,RMSE2]

# est, truth,PRMSE = PF_Main.PFPF(n,fnoise,tnoise,snoise,steps,trials,graphics)

# PF_Main.PFPF(n,fnoise,tnoise,snoise,steps,trials,methods,graphics)
est,truth,PRMSE = PF_Main.two_filters(n,fnoise,tnoise,snoise,steps,trials,methods,graphics)
visualize.plot_RMSE(PRMSE)
visualize.plot_paths(truth,est)
