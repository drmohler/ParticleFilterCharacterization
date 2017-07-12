""" Top level file to run multiple particle filters and
    compare the resutlst

Development sponsor: Air Force Research Laboratory ATR Summer Program
Mentor: Clark Taylor Ph.D.
Developer: David R. Mohler
Developed: June 2017
"""

import PF_Main
import numpy as np

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
n = 1000
fnoise = 0.25
tnoise = np.radians(5)
snoise = 0.25
steps = 126
trials = 5
graphics = False
methods = [2]


#MODIFY MAIN TO TAKE IN ALL PARAMETERS USED ABOVE, AS WELL AS TO SAVE ALL TRIALS
#PROBABLY WANT THIS SUCH THAT IT RUNS ON THE SAME DATA FOR ALL METHODS...
#REALLY BAD NEWS FOR COMPUTE TIME...

#run the particle filter for each of the chosen resampling methods
PF_Main.ParticleFilt(n,fnoise,tnoise,snoise,steps,trials,methods,graphics)