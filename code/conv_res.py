import pickle
import numpy as np
from Robot import PRMSE

fb3 = open('Convergence_test_bigHeading1.pkl','rb')
conv_list=pickle.load(fb3)
trials=10
steps=150

n_set=[50,100,200,400]

for i in range(len(n_set)):
    print('With',n_set[i],'particles')
    aux_se=[]
    sir_se=[]
    pfpf_se=[]
    enkf_se=[]
    for start_loc in range(10):
        input_filename = 'output/Results5_bigHeadingNoise_'+str(start_loc)+'random_start_'+str(n_set[i])+'particles_'+str(trials)+'trials_'+str(steps)+'steps.npz'
        file_stuff = np.load(input_filename)
        for j in range(trials):
            tmp = file_stuff['std_est'][j]
            tmp_shape = tmp.shape 
            tmp.shape = [1, tmp_shape[0], tmp_shape[1]]
            std_rmse = PRMSE(file_stuff['truth'], tmp)[0]
            if (conv_list[i][start_loc*10+j]=='b' or conv_list[i][start_loc*10+j]==';'):
                sir_se.append(std_rmse[100:-1])
            tmp = file_stuff['pfpf_est'][j]
            tmp_shape = tmp.shape 
            tmp.shape = [1, tmp_shape[0], tmp_shape[1]]
            pfpf_rmse = PRMSE(file_stuff['truth'], tmp)[0]
            pfpf_se.append(pfpf_rmse[100:-1])
            tmp = file_stuff['aux_mean'][j]
            tmp_shape = tmp.shape 
            tmp.shape = [1, tmp_shape[0], tmp_shape[1]]
            aux_rmse = PRMSE(file_stuff['truth'], tmp)[0]
            if (conv_list[i][start_loc*10+j]=='b' or conv_list[i][start_loc*10+j]=='a'):
                aux_se.append(aux_rmse[100:-1])
            tmp = file_stuff['enkf_mean'][j]
            tmp_shape = tmp.shape 
            tmp.shape = [1, tmp_shape[0], tmp_shape[1]]
            enkf_rmse = PRMSE(file_stuff['truth'], tmp)[0]
            enkf_se.append(enkf_rmse[100:-1])
    print('SIR conververged on',len(sir_se),'runs')
    print('aux conververged on',len(aux_se),'runs')
    print('pfpf conververged on',len(pfpf_se),'runs')
    print('enkf conververged on',len(enkf_se),'runs')
    print('SIR RMSE is',np.sqrt(np.mean(np.multiply(sir_se,sir_se))))
    print('Aux RMSE is',np.sqrt(np.mean(np.multiply(aux_se,aux_se))))
    print('PFPF RMSE is',np.sqrt(np.mean(np.multiply(pfpf_se,pfpf_se))))
    print('EnKF RMSE is',np.sqrt(np.mean(np.multiply(enkf_se,enkf_se))))
    
