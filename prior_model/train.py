import torch
from torch import nn
import numpy as np
import pandas as pd
import time
from itertools import chain
import sys
import os
sys.path.append('/home/xg23/scratch/DynAE/scripts')
import prior_model as prior

default_device = torch.device("cpu")

# load training/testing data set
b = np.arange(0.5,2.1,0.2)
det = 1    #data loading freq
data_list = []
for rb in b:
    betaT = np.round(rb,2)
    df = pd.read_csv(f'/home/xg23/scratch/DynAE/langevin/DW_beta_{betaT}_traj.csv', header=0,)
    traj_z = torch.from_numpy(df.Pos.values).float().to(default_device)
    betaT = torch.tensor(betaT, dtype=torch.float32).to(default_device)
    data0, data1, dataT = prior.data_pair_with_betaT(traj_z, betaT)
    data_list += [prior.data_split_train_test(data0, data1, dataT)]

train_data0 = torch.cat([data_list[i][0] for i in range(len(b))], dim=0)[::det]
test_data0 = torch.cat([data_list[i][1] for i in range(len(b))], dim=0)[::det]
train_data1 = torch.cat([data_list[i][2] for i in range(len(b))], dim=0)[::det]
test_data1 = torch.cat([data_list[i][3] for i in range(len(b))], dim=0)[::det]
train_dataT = torch.cat([data_list[i][4] for i in range(len(b))], dim=0)[::det]
test_dataT = torch.cat([data_list[i][5] for i in range(len(b))], dim=0)[::det]
        
# load extra inference data set
b = np.arange(0.6,2.1,0.2)
infer_data_list = []
for rb in b:
    betaT = np.round(rb,2)
    df = pd.read_csv(f'/home/xg23/scratch/DynAE/langevin/DW_beta_{betaT}_traj.csv', header=0,)    
    traj_z = torch.from_numpy(df.Pos.values).float().to(default_device)
    betaT = torch.tensor(betaT, dtype=torch.float32).to(default_device)
                         
    infer_data_list += [prior.data_pair_with_betaT(traj_z, betaT)]
    
infer_data0 = torch.cat([infer_data_list[i][0] for i in range(len(b))], dim=0)[::det]
infer_data1 = torch.cat([infer_data_list[i][1] for i in range(len(b))], dim=0)[::det]
infer_dataT = torch.cat([infer_data_list[i][2] for i in range(len(b))], dim=0)[::det]

beta_MSEs = [2, 3, 4, 5, 6, 7, 8, 9]
batch_sizes = [32,64,128,256,512]
lr = 0.01
for beta_mse in beta_MSEs:
    for bsize in batch_sizes:
        My_model = []
        My_model = prior.prior_model(z_dim=1, device=default_device, ConstantDiffusionPrior=False)
        My_model.to(default_device)
        My_model.train()
    
        My_model.train_model(train_data0, test_data0, train_data1, test_data1, train_dataT, test_dataT, 
                            infer_data0, infer_data1, infer_dataT, prior_learning_rate=lr, batch_size=bsize, max_epochs=200, 
                            output_path=f'/home/xg23/scratch/DynAE/DW_stepLR_mse_betamse{beta_mse}_batchsize{bsize}', 
                            log_interval=20, SaveTrainingProgress=True, lr_scheduler_step_size=10, lr_scheduler_gamma=0.8, beta_MSE=beta_mse)


