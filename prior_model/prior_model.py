import torch
from torch import nn
import numpy as np
import os
import time
from itertools import chain
import utils

# data_loader
def data_pair_with_betaT(data_set, betaT):
    data0 = data_set[:-1]
    data1 = data_set[1:]
    dataT = torch.full((len(data0), 1), betaT)
    assert len(data0) == len(data1)
    assert len(data0) == len(dataT)
    
    return data0.view((len(data0),-1)), data1.view((len(data1),-1)), dataT    
    
def data_split_train_test(data0, data1, dataT):
    n_data = len(data0)
    p = np.random.permutation(n_data)
    data0 = data0[p]
    data1 = data1[p]
    dataT = dataT[p]
    
    train_set0 = data0[: (8*n_data)//10]
    test_set0 = data0[(8*n_data)//10 :]
    
    train_set1 = data1[: (8*n_data)//10]
    test_set1 = data1[(8*n_data)//10 :]
    
    train_setT = dataT[: (8*n_data)//10]
    test_setT = dataT[(8*n_data)//10 :]
    
    return train_set0, test_set0, train_set1, test_set1, train_setT, test_setT
    
def sample_mini_batch(data0, data1, dataT, indices, device):
    mini_data0 = data0[indices].to(device)
    mini_data1 = data1[indices].to(device)
    mini_dataT = dataT[indices].to(device)
    
    return mini_data0, mini_data1, mini_dataT


# architecture
class prior_model(nn.Module):
    def __init__(self, z_dim, device, ConstantDiffusionPrior=False, reduced_force=True, neuron_num=32, bias_factor=3):
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.neuron_num = neuron_num
        self.bias_factor = bias_factor
        self.ConstantDiffusionPrior = ConstantDiffusionPrior
        self.reduced_force = reduced_force
        
        if ConstantDiffusionPrior:
            print("Constant Diffusion!!")
            self.constant_logM = nn.Parameter(torch.randn(1, z_dim))
        
        self.logA_net = nn.Sequential(
            nn.Linear(z_dim, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, z_dim))
        
        self.ea_net = nn.Sequential(
            nn.Linear(z_dim, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, z_dim),
            nn.ReLU())
        
        self.force_net = nn.Sequential(
            nn.Linear(z_dim+1, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, neuron_num),
            nn.Tanh(),
            nn.Linear(neuron_num, z_dim))
    
    def prior_logA(self, z):
        return self.logA_net(z)
    
    def prior_EA(self, z):
        return self.ea_net(z)
    
    def prior_force(self, zwt):
        return self.force_net(zwt)
    
    def prior_loss(self, z0, z1, betaT):
        z0 = z0.detach()
        z0.requires_grad = True
        if self.reduced_force:
            z0wt = torch.cat((z0, betaT), dim=1)
        else:
            z0wt = torch.cat((z0, torch.zeros_like(betaT)), dim=1)
        force = self.prior_force(z0wt)
        
        if self.ConstantDiffusionPrior:
            logM = self.constant_logM
            M = torch.exp(logM)
            prior_loss = 0.5*torch.sum(logM + 0.5*betaT*torch.pow(z1 - z0 - M*force, 2)/M, dim=1)        
            
        else:
            logA = self.prior_logA(z0)
            EA = self.prior_EA(z0)

            D_factor = torch.exp(-betaT*EA)
            A = torch.exp(logA)
            M = A*D_factor
            logM = logA - betaT*EA

            logA_grad = []
            for i in range(self.z_dim):
                logA_i = logA[:,i]
                logA_grad += [torch.autograd.grad(logA_i.sum(),z0,retain_graph=True)[0][:,i]]
            A_grad = torch.stack(logA_grad, dim=-1) * A

            EA_grad = []
            for i in range(self.z_dim):
                EA_i = EA[:,i]
                EA_grad += [torch.autograd.grad(EA_i.sum(),z0,retain_graph=True)[0][:,i]]
            EA_grad = torch.stack(EA_grad, dim=-1)

            M_grad = D_factor*A_grad - betaT * M * EA_grad

            prior_loss = 0.5*torch.sum(logM + 0.5*betaT*torch.pow(z1 - z0 - M*force+ M_grad/betaT, 2)/M, dim=1)
        
        return prior_loss
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std * torch.randn_like(mu)
    
    def Langevin_Forward(self, z0, betaT, dt_infer):
        z0 = z0.detach()
        z0.requires_grad = True        
        if dt_infer != 1:
            raise NotImplementedError
        if self.reduced_force:
            z0wt = torch.cat((z0, betaT), dim=1)
        else:
            z0wt = torch.cat((z0, torch.zeros_like(betaT)), dim=1)
 
        force = self.prior_force(z0wt)
        
        if self.ConstantDiffusionPrior:
            logM = self.constant_logM
            M = torch.exp(logM) 
            z1 = z0 + self.reparameterize(M*force, logM-np.log(betaT/2))
            
        else:
            logA = self.prior_logA(z0)
            EA = self.prior_EA(z0)        

            D_factor = torch.exp(-betaT*EA)
            A = torch.exp(logA)
            M = A*D_factor
            logM = logA - betaT*EA

            logA_grad = []
            for i in range(self.z_dim):
                logA_i = logA[:,i]
                logA_grad += [torch.autograd.grad(logA_i.sum(),z0,retain_graph=True)[0][:,i]]
            A_grad = torch.stack(logA_grad, dim=-1) * A

            EA_grad = []
            for i in range(self.z_dim):
                EA_i = EA[:,i]
                EA_grad += [torch.autograd.grad(EA_i.sum(),z0,retain_graph=True)[0][:,i]]
            EA_grad = torch.stack(EA_grad, dim=-1)

            M_grad = D_factor*A_grad - betaT * M * EA_grad    

            z1 = z0 + self.reparameterize(M*force+ M_grad/betaT, logM-np.log(betaT/2))
        
        return z1

    def evolve_latent_dynamics(self, z_init, betaT_infer, infer_steps, dt_infer=1):
        traj = []
        z0 = z_init.clone()
        z0 = z0.view((-1,self.z_dim))
        betaT = torch.full((len(z0),1), betaT_infer)
        for istep in range(infer_steps):
            if istep % 1000 == 0:
                print('Step:', istep, 'Temp:', betaT_infer)
            z1 = self.Langevin_Forward(z0, betaT, dt_infer)
            z0 = z1.detach()
            traj.append(z0)
            
        return traj       


    @torch.no_grad()
    def get_cluster_centers(self, train_input_data, test_input_data, infer_input_data, save_centers=False, log_path=None, batch_size=128):
        # This function generates the cluster centers from regular space clustering

        if log_path!=None:
            start_time = time.time()

#        # obtain the latent representation
#        train_all_z = []
#        for i in range(0, len(train_input_data), batch_size):
#            batch_inputs = train_input_data[i:i + batch_size].to(self.device)
#
#            # pass through VAE
#            z = self.encode(batch_inputs)
#
#            train_all_z += [z.cpu()]
#
#        train_all_z = torch.cat(train_all_z, dim=0)
#
#        test_all_z = []
#        for i in range(0, len(test_input_data), batch_size):
#            batch_inputs = test_input_data[i:i + batch_size].to(self.device)
#
#            # pass through VAE
#            z = self.encode(batch_inputs)
#            test_all_z += [z.cpu()]
#
#        test_all_z = torch.cat(test_all_z, dim=0)
#
	# for training latent prior model only
        train_all_z = train_input_data
        test_all_z = test_input_data
        infer_all_z = infer_input_data

        # dicretize the latent space into bins
        cluster_centers = utils.RegSpaceClustering(train_all_z)
	

        # obtain the cluster labels
        train_distance_matrix = torch.sqrt((torch.square(train_all_z.unsqueeze(1) - cluster_centers.unsqueeze(0))).sum(dim=-1))
        train_cluster_labels = torch.argmin(train_distance_matrix, dim=1)

        test_distance_matrix = torch.sqrt((torch.square(test_all_z.unsqueeze(1) - cluster_centers.unsqueeze(0))).sum(dim=-1))
        test_cluster_labels = torch.argmin(test_distance_matrix, dim=1)

        infer_distance_matrix = torch.sqrt((torch.square(infer_all_z.unsqueeze(1) - cluster_centers.unsqueeze(0))).sum(dim=-1))
        infer_cluster_labels = torch.argmin(infer_distance_matrix, dim=1)


        if log_path!=None:
            elapsed_time = time.time() - start_time
            print('Finished after ' + str(elapsed_time) + 's')
            print('%i cluster centers detected' % len(cluster_centers) + '\n')

            print('Finished after ' + str(elapsed_time) + 's', file=open(log_path, 'a'))
            print('%i cluster centers detected' % len(cluster_centers) + '\n', file=open(log_path, 'a'))

        if save_centers:
            return train_cluster_labels, test_cluster_labels, infer_cluster_labels, cluster_centers

        else:
            return train_cluster_labels, test_cluster_labels, infer_cluster_labels

    def resampling(self, train_past_data0, test_past_data0, infer_past_data0, save_centers, output_path, log_path, index=0, batch_size=32):
        '''
        Uniformly discretizing the latent space and resampling the dataset based on a well-tempered distribution.
            Args:
                train_past_data0: ndarray containing (n,d)-shaped float data for training
                test_past_data0: ndarray containing (n,d)-shaped float data for test
                save_centers: bool, whether to save cluster centers
                output_path: str
                log_path: str
                index: int, random seed number

            Returns:
                train_indices: the resampled indices of training dataset
                test_indices the resampled indices of test dataset
        '''

        # discretize the latent space into bins using regular clustering
        output_variables = self.get_cluster_centers(train_past_data0, test_past_data0, infer_past_data0, save_centers=save_centers, log_path=log_path)

        train_cluster_labels, test_cluster_labels, infer_cluster_labels = output_variables[0], output_variables[1], output_variables[2]

        # output z cluster centers
        if save_centers:
            cluster_centers = output_variables[3]
            z_cluster_center_path = output_path + '_z_cluster_centers' + str(index) + '.npy'
            np.save(z_cluster_center_path, cluster_centers.cpu().data.numpy())

        num_cluster = int(torch.max(train_cluster_labels).cpu().numpy()) + 1
        # draw samples based on the bias factor (>=1)
        # n_k' ~ n_k^(1/bias_factor)
        cluster_weights = []
        total_weights = 0
        train_cluster_indices = []
        test_cluster_indices = []
        infer_cluster_indices = []

        total_effective_samples = 0

        for k in range(num_cluster):
            train_cluster_indices += [torch.nonzero(train_cluster_labels == k, as_tuple=True)[0]]
            test_cluster_indices += [torch.nonzero(test_cluster_labels == k, as_tuple=True)[0]]
            infer_cluster_indices += [torch.nonzero(infer_cluster_labels == k, as_tuple=True)[0]]

            if len(train_cluster_indices[k]) > batch_size:
                total_effective_samples += len(train_cluster_indices[k])

            weight = np.power(len(train_cluster_indices[k]), 1/self.bias_factor)
            cluster_weights += [weight]
            total_weights += weight

        if total_effective_samples < train_past_data0.shape[0]*0.8:
            print(1.0*total_effective_samples/train_past_data0.shape[0])
            print("Too few samples in each bin! Please increase dmin!")
            raise ValueError

        # create better dataset by resampling from each bin
        train_dataset_indices = []
        test_dataset_indices = []
        infer_dataset_indices = []

        for k in range(num_cluster):
            train_dataset_size = int(train_past_data0.shape[0] * cluster_weights[k] / total_weights / batch_size + 1) * batch_size
            test_dataset_size = int(test_past_data0.shape[0] * cluster_weights[k] / total_weights / batch_size + 1) * batch_size
            infer_dataset_size = int(infer_past_data0.shape[0] * cluster_weights[k] / total_weights / batch_size + 1) * batch_size

            if len(train_cluster_indices[k]) > train_dataset_size:
                train_dataset_indices += [train_cluster_indices[k][
                                              torch.randperm(len(train_cluster_indices[k]))[
                                              :train_dataset_size]]]
#            elif len(train_cluster_indices[k]) > batch_size:
#                size = train_cluster_indices[k].shape[0] // batch_size * batch_size
            else:
#                print((train_dataset_size) // train_cluster_indices[k].shape[0], train_dataset_size, train_cluster_indices[k].shape[0])
                for i in range((train_dataset_size) // train_cluster_indices[k].shape[0]):
                    train_dataset_indices += [
                        train_cluster_indices[k][
                            torch.randperm(len(train_cluster_indices[k]))]]#[:size]]]

            if len(test_cluster_indices[k]) > test_dataset_size:
                test_dataset_indices += [test_cluster_indices[k][
                                             torch.randperm(len(test_cluster_indices[k]))[
                                             :test_dataset_size]]]
#            elif len(test_cluster_indices[k]) > batch_size:
#                size = test_cluster_indices[k].shape[0] // batch_size * batch_size
            elif test_cluster_indices[k].shape[0] > 0:
                for i in range(test_dataset_size // test_cluster_indices[k].shape[0]):
                    test_dataset_indices += [
                        test_cluster_indices[k][torch.randperm(len(test_cluster_indices[k]))]]#[:size]]]


            if len(infer_cluster_indices[k]) > infer_dataset_size:
                infer_dataset_indices += [infer_cluster_indices[k][
                                             torch.randperm(len(infer_cluster_indices[k]))[
                                             :infer_dataset_size]]]
            elif infer_cluster_indices[k].shape[0] > 0:
                for i in range(infer_dataset_size // infer_cluster_indices[k].shape[0]):
                    infer_dataset_indices += [
                        infer_cluster_indices[k][torch.randperm(len(infer_cluster_indices[k]))]]


        train_dataset_indices = torch.cat(train_dataset_indices, dim=0)#.reshape((-1, batch_size))
        test_dataset_indices = torch.cat(test_dataset_indices, dim=0)#.reshape((-1, batch_size))
        infer_dataset_indices = torch.cat(infer_dataset_indices, dim=0)

        train_indices = train_dataset_indices[
            torch.randperm((train_dataset_indices).shape[0])].flatten()
        test_indices = test_dataset_indices[
            torch.randperm((test_dataset_indices).shape[0])].flatten()
        infer_indices = infer_dataset_indices[
            torch.randperm((infer_dataset_indices).shape[0])].flatten()

        return train_indices, test_indices, infer_indices
 
    
    def train_model(self, train_set0, test_set0, train_set1, test_set1, train_setT, test_setT, 
                    infer_set0, infer_set1, infer_setT, prior_learning_rate, batch_size, max_epochs, 
                    output_path, log_interval, SaveTrainingProgress, lr_scheduler_step_size, lr_scheduler_gamma, beta_MSE):
        self.train()

        step = 0        # steps of model updates
        start = time.time()
        os.makedirs(output_path,exist_ok=True)
        log_path = output_path + '/log.txt'
        model_path = output_path + '/prior_model'
        os.makedirs(model_path,exist_ok=True)

        epoch = 0        # cycles of training data set 

        #setup optimizer
        prior_optimizer = torch.optim.Adam(self.parameters(),lr=prior_learning_rate,)
        
        prior_scheduler = torch.optim.lr_scheduler.StepLR(prior_optimizer, step_size=lr_scheduler_step_size,
                                                    gamma=lr_scheduler_gamma)
        
        train_resample, test_resample, infer_resample = self.resampling(train_set0, test_set0, infer_set0, save_centers=False, output_path=output_path, log_path=log_path) 
        while epoch < max_epochs:
            if epoch == 0:
                train_permutation = torch.randperm(train_set0.shape[0])
                test_permutation = torch.randperm(test_set0.shape[0])
                infer_permutation = torch.randperm(infer_set0.shape[0])
            else:
                train_permutation = train_resample[torch.randperm((train_resample).shape[0])] 
                test_permutation = test_resample[torch.randperm((test_resample).shape[0])]
                infer_permutation = infer_resample[torch.randperm((infer_resample).shape[0])]


            for i in range(0, len(train_permutation), batch_size):
                step += 1
                if (i+batch_size) > len(train_permutation):
                    print(i+batch_size, len(train_permutation))
                    break
                
                train_indices = train_permutation[i:(i+batch_size)]
                z0, z1, temp = sample_mini_batch(train_set0, train_set1, train_setT, train_indices, self.device)  # call function from utils
                train_loss = self.prior_loss(z0, z1, temp).mean()
                
                #MSE reconstruction loss
                if beta_MSE > 0:
                    z1_sampled = self.Langevin_Forward(z0, temp, dt_infer=1)
                    train_loss += beta_MSE*torch.sum(torch.square(z1_sampled - z1).flatten(start_dim=1),dim=1).mean()                
                
                if (torch.isnan(train_loss).any()):
                    print("NAN in training loss")
                    print(train_loss)
                    return True
                
                prior_optimizer.zero_grad()
                train_loss.backward()
                prior_optimizer.step()
                
                if step % log_interval == 0:  #output log every 500 steps
                    train_time = time.time() - start
                    
                    print(f"Iteration {step}:\tTime {train_time} s\nPrior loss (train) {train_loss}")
                    print(f"Iteration {step}:\tTime {train_time} s\nPrior loss (train) {train_loss}", 
                          file=open(log_path,'a'))
                    
                    j = i%len(test_permutation)
                    if (j+batch_size) > len(test_permutation):
                        j = len(test_permutation) - batch_size
                        
                    test_indices = test_permutation[j:(j+batch_size)]
                    z0, z1, temp = sample_mini_batch(test_set0, test_set1, test_setT, test_indices, self.device)  # call function from utils
                    test_loss = self.prior_loss(z0, z1, temp).mean()
                    # MSE loss
                    if beta_MSE > 0:
                        z1_sampled = self.Langevin_Forward(z0, temp, dt_infer=1)
                        test_loss += beta_MSE*torch.sum(torch.square(z1_sampled - z1).flatten(start_dim=1),dim=1).mean()
                    
                    print(f"Prior loss (test) {test_loss}")
                    print(f"Prior loss (test) {test_loss}", file=open(log_path,'a'))
                    
                    k = i%len(infer_permutation)
                    if (k+batch_size) > len(infer_permutation):
                        k = len(infer_permutation) - batch_size
                        
                    infer_indices = infer_permutation[k:(k+batch_size)]
                    z0, z1, temp = sample_mini_batch(infer_set0, infer_set1, infer_setT, infer_indices, self.device)  # call function from utils
                    infer_loss = self.prior_loss(z0, z1, temp).mean()
                    # MSE loss
                    if beta_MSE > 0:
                        z1_sampled = self.Langevin_Forward(z0, temp, dt_infer=1)
                        infer_loss += beta_MSE*torch.sum(torch.square(z1_sampled - z1).flatten(start_dim=1),dim=1).mean()
                    
                    print(f"Prior loss (infer) {infer_loss}")
                    print(f"Prior loss (infer) {infer_loss}", file=open(log_path,'a'))                   
                    
                
            epoch += 1
            prior_scheduler.step()
            if prior_scheduler.gamma < 1:
                print("Update lr to %f" % (prior_optimizer.param_groups[0]['lr']))
                print("Update lr to %f" % (prior_optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))
            
            if SaveTrainingProgress:
                if epoch % 10 == 0:
                    # self.eval()
                    # for i in range(len(input_data_list)):
                    #     self.save_traj_results(input_data_list[i], batch_size, output_path + '_epoch%d' % epoch, False, i, index)
                    # self.train()
                    torch.save({'epoch': epoch,
                                'state_dict': self.state_dict()}, model_path + f'/model_{epoch}_cpt.pt')
                    
            print(f"Epoch: {epoch}\n")
            print(f"Epoch: {epoch}\n", file=open(log_path, 'a'))
                


        total_training_time = time.time() - start
        print(f"Total training time: {total_training_time} s")
        print(f"Total training time: {total_training_time} s", file=open(log_path, 'a'))
        torch.save({'epoch': epoch,
            'state_dict': self.state_dict()}, model_path + f'/model_final_cpt.pt')
        
        return False
        
    def output_result(self, train_set0, test_set0, train_set1, test_set1, train_setT, test_setT,
                    infer_set0, infer_set1, infer_setT, beta_MSE, outputfile, batch_size=32):

        train_resample, test_resample, infer_resample = self.resampling(train_set0, test_set0, infer_set0, save_centers=False, output_path=None, log_path=outputfile)
        train_permutation = train_resample[torch.randperm((train_resample).shape[0])]
        test_permutation = test_resample[torch.randperm((test_resample).shape[0])]
        infer_permutation = infer_resample[torch.randperm((infer_resample).shape[0])]

        train_prior_loss = []
        test_prior_loss = []
        infer_prior_loss = []
        train_mse_loss = []
        test_mse_loss = []
        infer_mse_loss = []
        for i in range(0, len(train_permutation), batch_size):
            if (i+batch_size) > len(train_permutation):
                print(i+batch_size, len(train_permutation))
                break

            train_indices = train_permutation[i:(i+batch_size)]
            z0, z1, temp = sample_mini_batch(train_set0, train_set1, train_setT, train_indices, self.device)  # call function from utils
            train_prior_loss += [self.prior_loss(z0, z1, temp).mean().detach().cpu().numpy()]

            #MSE reconstruction loss
            if beta_MSE > 0:
                z1_sampled = self.Langevin_Forward(z0, temp, dt_infer=1)
                train_mse_loss += [torch.sum(torch.square(z1_sampled - z1).flatten(start_dim=1),dim=1).mean().detach().cpu().numpy()]
        train_prior_loss = np.mean(train_prior_loss)
        train_mse_loss = np.mean(train_mse_loss)

        for i in range(0, len(test_permutation), batch_size):
            if (i+batch_size) > len(test_permutation):
                print(i+batch_size, len(test_permutation))
                break

            test_indices = test_permutation[i:(i+batch_size)]
            z0, z1, temp = sample_mini_batch(test_set0, test_set1, test_setT, test_indices, self.device)  # call function from utils
            test_prior_loss += [self.prior_loss(z0, z1, temp).mean().detach().cpu().numpy()]

            #MSE reconstruction loss
            if beta_MSE > 0:
                z1_sampled = self.Langevin_Forward(z0, temp, dt_infer=1)
                test_mse_loss += [torch.sum(torch.square(z1_sampled - z1).flatten(start_dim=1),dim=1).mean().detach().cpu().numpy()]
        test_prior_loss = np.mean(test_prior_loss)
        test_mse_loss = np.mean(test_mse_loss)


        for i in range(0, len(infer_permutation), batch_size):
            if (i+batch_size) > len(infer_permutation):
                print(i+batch_size, len(infer_permutation))
                break

            infer_indices = infer_permutation[i:(i+batch_size)]
            z0, z1, temp = sample_mini_batch(infer_set0, infer_set1, infer_setT, infer_indices, self.device)  # call function from utils
            infer_prior_loss += [self.prior_loss(z0, z1, temp).mean().detach().cpu().numpy()]

            #MSE reconstruction loss
            if beta_MSE > 0:
                z1_sampled = self.Langevin_Forward(z0, temp, dt_infer=1)
                infer_mse_loss += [torch.sum(torch.square(z1_sampled - z1).flatten(start_dim=1),dim=1).mean().detach().cpu().numpy()]
        infer_prior_loss = np.mean(infer_prior_loss)
        infer_mse_loss = np.mean(infer_mse_loss)    

        print(f"beta_MSE = {beta_MSE}", file=open(outputfile, 'a'))

        print(f"train_prior_loss: {train_prior_loss}", file=open(outputfile, 'a'))
        print(f"test_prior_loss: {test_prior_loss}", file=open(outputfile, 'a'))
        print(f"infer_prior_loss: {infer_prior_loss}", file=open(outputfile, 'a'))

        print(f"train_mse_loss: {train_mse_loss}", file=open(outputfile, 'a'))
        print(f"test_mse_loss: {test_mse_loss}", file=open(outputfile, 'a'))
        print(f"infer_mse_loss: {infer_mse_loss}", file=open(outputfile, 'a'))

        return False

