from visualize import *
import matplotlib.pyplot as plt

import numpy as np
import sys
import pickle

def press(event):
    # print('press', event.key)
    # sys.stdout.flush()
    conv_lst.append(event.key)


start_loc=1 # typically goes 0 - 4
n_set=[50,100,200,400]

#These next two are usually fixed across runs, but are in the filename.
trials = 10
steps = 150

from Robot import PRMSE


fig, ax = plt.subplots()

fig.canvas.mpl_connect('key_press_event', press)

res_list=[]

for n in n_set:
    conv_lst = []
    aux_mse=[]
    pfpf_mse=[]
    std_mse=[]
    enkf_mse=[]
    for start_loc in range(10):
        input_filename = 'output/Results5_bigHeadingNoise_'+str(start_loc)+'random_start_'+str(n)+'particles_'+str(trials)+'trials_'+str(steps)+'steps.npz'
        file_stuff = np.load(input_filename)

        # #This is the code to plot the paths for this file:
        # #plot_paths assumes [0] is pfpf, [1] is std, [2] is enkf, [3] is aux
        # combined_ests = [file_stuff['pfpf_est'], file_stuff['std_est'], \
        #                  file_stuff['enkf_mean'], file_stuff['aux_mean']]
        # plot_paths(file_stuff['truth'], combined_ests, plot_name = 'Test1')                 

        #Now for some code that plots the errors for each method
        #First, find the RMSE for a given type of particle filter
        for j in range(trials):
            tmp = file_stuff['std_est'][j]
            tmp_shape = tmp.shape 
            tmp.shape = [1, tmp_shape[0], tmp_shape[1]]
            std_rmse = PRMSE(file_stuff['truth'], tmp)
            tmp = file_stuff['pfpf_est'][j]
            tmp_shape = tmp.shape 
            tmp.shape = [1, tmp_shape[0], tmp_shape[1]]
            pfpf_rmse = PRMSE(file_stuff['truth'], tmp)
            tmp = file_stuff['aux_mean'][j]
            tmp_shape = tmp.shape 
            tmp.shape = [1, tmp_shape[0], tmp_shape[1]]
            aux_rmse = PRMSE(file_stuff['truth'], tmp)
            tmp = file_stuff['enkf_mean'][j]
            tmp_shape = tmp.shape 
            tmp.shape = [1, tmp_shape[0], tmp_shape[1]]
            enkf_rmse = PRMSE(file_stuff['truth'], tmp)
            rmse_list=[pfpf_rmse, std_rmse, enkf_rmse, aux_rmse ]

            my_plot_name = 'RMSE, run '+str(start_loc)+', trial '+str(j)+', '+str(n)+' particles'
            ax.set_title(my_plot_name)
            ax.plot(pfpf_rmse[0], label="PFPF")
            ax.plot(std_rmse[0], label='SIR')
            ax.plot(aux_rmse[0], label='Aux')
            ax.plot(enkf_rmse[0], label='EnKF')
            ax.legend()
            plt.draw()
            plt.waitforbuttonpress()
            plt.cla()
        std_mse.append(np.sum(np.multiply(std_rmse[0],std_rmse[0])))
        pfpf_mse.append(np.sum(np.multiply(pfpf_rmse[0],pfpf_rmse[0])))
        aux_mse.append(np.sum(np.multiply(aux_rmse[0],aux_rmse[0])))
        enkf_mse.append(np.sum(np.multiply(enkf_rmse[0],enkf_rmse[0])))

    print('At',n,'particles:')
    print('SIR RMSE is',sqrt(np.mean(std_mse)))
    print('PFPF RMSE is',sqrt(np.mean(pfpf_mse)))    
    print('Aux RMSE is',sqrt(np.mean(aux_mse)))
    res_list.append(conv_lst)

print(res_list)
fb = open ('Convergence_junker.pkl','wb')
pickle.dump(res_list,fb)
# #This was the original, find the RMSE across the whole set code...
# for n in n_set:
#     aux_mse=[]
#     pfpf_mse=[]
#     std_mse=[]
#     enkf_mse=[]
#     for start_loc in range(10):
#         input_filename = 'output/Results5_regularNoise_'+str(start_loc)+'random_start_'+str(n)+'particles_'+str(trials)+'trials_'+str(steps)+'steps.npz'
#         file_stuff = np.load(input_filename)

#         # #This is the code to plot the paths for this file:
#         # #plot_paths assumes [0] is pfpf, [1] is std, [2] is enkf, [3] is aux
#         # combined_ests = [file_stuff['pfpf_est'], file_stuff['std_est'], \
#         #                  file_stuff['enkf_mean'], file_stuff['aux_mean']]
#         # plot_paths(file_stuff['truth'], combined_ests, plot_name = 'Test1')                 

#         #Now for some code that plots the errors for each method
#         #First, find the RMSE for a given type of particle filter
#         std_rmse = PRMSE(file_stuff['truth'], file_stuff['std_est'])
#         pfpf_rmse = PRMSE(file_stuff['truth'], file_stuff['pfpf_est'])
#         aux_rmse = PRMSE(file_stuff['truth'], file_stuff['aux_mean'])
#         enkf_rmse = PRMSE(file_stuff['truth'], file_stuff['enkf_mean'])
#         rmse_list=[pfpf_rmse, std_rmse, enkf_rmse, aux_rmse ]
#         my_plot_name = 'RMSE, run '+str(start_loc)+', '+str(n)+' particles'
#         plot_RMSE(rmse_list, plot_name = my_plot_name)
#         std_mse.append(np.mean(np.multiply(std_rmse[0],std_rmse[0])))
#         pfpf_mse.append(np.mean(np.multiply(pfpf_rmse[0],pfpf_rmse[0])))
#         aux_mse.append(np.mean(np.multiply(aux_rmse[0],aux_rmse[0])))
#         enkf_mse.append(np.mean(np.multiply(enkf_rmse[0],enkf_rmse[0])))

#     print('At',n,'particles:')
#     print('SIR RMSE is',sqrt(np.mean(std_mse)))
#     print('PFPF RMSE is',sqrt(np.mean(pfpf_mse)))    
#     print('Aux RMSE is',sqrt(np.mean(aux_mse)))
#     print('EnKF RMSE is',sqrt(np.mean(enkf_mse)))