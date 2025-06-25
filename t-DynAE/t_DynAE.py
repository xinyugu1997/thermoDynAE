import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import os
import time
from itertools import chain
import utils
import architectures

class t_DynAE(nn.Module):
    def __init__(self, z_dim, output_shape, data_shape, device, reduced_force_T_noise=0, neuron_num1=16, neuron_num2=16, 
                 projection_num=50, ConstantDiffusionPrior=True, reduced_force=True, embed_dim=64, embed_scale=10.0, 
                 neuron_num=32, bias_factor=3, min_centers=200):
        super().__init__()
        self.z_dim = z_dim
        self.output_shape = output_shape
        self.data_shape = data_shape
        self.device = device
        self.projection_num = projection_num
        self.embed_dim = embed_dim
        self.embed_scale = embed_scale
        self.min_centers = min_centers
        self.neuron_num = neuron_num
        self.neuron_num1 = neuron_num1
        self.neuron_num2 = neuron_num2
        self.bias_factor = bias_factor
        self.ConstantDiffusionPrior = ConstantDiffusionPrior
        self.reduced_force = reduced_force
        self.reduced_force_T_noise = reduced_force_T_noise
        

        self.model_encoder = architectures.fc_encoder(z_dim, data_shape, neuron_num1)
        self.model_decoder = architectures.fc_decoder(z_dim, output_shape, neuron_num2)
        self.model_prior = architectures.Langevin_prior(z_dim, device, ConstantDiffusionPrior, reduced_force, reduced_force_T_noise, embed_dim, embed_scale, neuron_num)


    def encode(self, inputs):
        h = self.model_encoder.encoder_input_layer(inputs)
        enc = self.model_encoder.encoder(h)
        z = self.model_encoder.encoder_output_layer(enc)
        return z

    def decode(self, z):
        h = self.model_decoder.decoder_input_layer(z)
        dec = self.model_decoder.decoder(h)
        outputs = self.model_decoder.decoder_output_layer(dec)
        return outputs

    def forward(self, data):
        z = self.encode(data)
        outputs = self.decode(z)
        return outputs, z

    
    def prior_loss(self, z0, z1, betaT, beta_MSE=0):
        z0 = z0.detach()
        z1 = z1.detach()
        z0.requires_grad = True
        if self.reduced_force:
            embed_t = betaT+self.reduced_force_T_noise*torch.randn_like(betaT)
        else:
            embed_t = torch.zeros_like(betaT)
        force = self.model_prior.prior_force(z0, embed_t)
        
        if self.ConstantDiffusionPrior:
            #logM = self.model_prior.constant_logM
            #M = torch.exp(logM)
            #prior_loss = 0.5*torch.sum(logM + 0.5*betaT*torch.pow(z1 - z0 - M*force, 2)/M, dim=1)        
            prior_loss = 0.5*torch.sum(0.5*betaT*torch.pow(z1 - z0 - force, 2), dim=1)
            
        else:
            logA = self.model_prior.prior_logA(z0)
            EA = self.model_prior.prior_EA(z0)

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

        #MSE reconstruction loss
        if beta_MSE > 0:
            z1_sampled = self.Langevin_Forward(z0, betaT, dt_infer=1)
            MSE_loss = torch.sum(torch.square(z1_sampled - z1).flatten(start_dim=1),dim=1)
            return prior_loss.mean() + beta_MSE * MSE_loss.mean()
        else:
            return prior_loss.mean()


    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        return mu + std * torch.randn_like(mu)
    
    def Langevin_Forward(self, z0, betaT, dt_infer):
        z0 = z0.detach()
        z0.requires_grad = True       
        if self.reduced_force:
            embed_t = betaT    #+self.reduced_force_T_noise*torch.randn_like(betaT)
        else:
            embed_t = torch.zeros_like(betaT)
        force = self.model_prior.prior_force(z0, embed_t)
        
        if self.ConstantDiffusionPrior:
            #logM = self.model_prior.constant_logM
            #M = torch.exp(logM) 
            #z1 = z0 + self.reparameterize(M*force*dt_infer, logM-np.log(betaT/2/dt_infer))
            z1 = z0 + self.reparameterize(force*dt_infer, -np.log(betaT/2/dt_infer))
            
        else:
            logA = self.model_prior.prior_logA(z0)
            EA = self.model_prior.prior_EA(z0)        

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

            z1 = z0 + self.reparameterize(M*force*dt_infer + M_grad*dt_infer/betaT, logM-np.log(betaT/2/dt_infer))
        
        return z1

    def evolve_latent_dynamics(self, z_init, betaT_infer, infer_steps, dt_infer=1):
        traj = []
        z0 = z_init.clone()
        z0 = z0.view((-1,self.z_dim))
        betaT = torch.full((len(z0),1), betaT_infer)
        for istep in range(infer_steps):
           if istep % 10000 == 0:
                print('Step:', istep, 'Temp:', betaT_infer, 'z:', z0.cpu().data.numpy())
           z1 = self.Langevin_Forward(z0, betaT, dt_infer)
           z0 = z1.detach()
           traj += [z0]
        traj = torch.cat(traj, dim=0) 
            
        return traj       

    def calculate_loss(self, betaT, input_data0, input_data1, target_data0, target_data1, 
                       betaT_bins, beta=1.0):
        batch_size = input_data0.shape[0]
        data = torch.cat([input_data0, input_data1], dim=0)
        out, z = self.forward(data)
         
        output_data0 = out[:batch_size]
        output_data1 = out[batch_size:]

        z0 = z[:batch_size]
        z1 = z[batch_size:]

        encoded_samples = z1 - z0
        prior_samples = self.Langevin_Forward(z0, betaT, dt_infer=1).detach() - z0.detach()

        reconstruction_error = torch.sum( torch.pow((output_data0 - target_data0), 2), dim=1 ).mean() + \
                               torch.sum( torch.pow((output_data1 - target_data1), 2), dim=1 ).mean()

        sw_loss = utils.sliced_wasserstein_distance(encoded_samples, prior_samples, betaT, betaT_bins, 
                                                    self.projection_num, device=self.device)

        loss = reconstruction_error + beta * sw_loss

        # detach the graph from encoder
        z0_detached = z0.detach()
        z1_detached = z1.detach()

        prior_loss = self.prior_loss(z0_detached, z1_detached, betaT)


        return loss, reconstruction_error.detach(), sw_loss.detach(), prior_loss 

    @torch.no_grad()
    def get_cluster_centers(self, train_input_data, test_input_data, save_centers=False, log_path=None, batch_size=128):
        # This function generates the cluster centers from regular space clustering

        if log_path!=None:
            start_time = time.time()

        # obtain the latent representation
        train_all_z = []
        for i in range(0, len(train_input_data), batch_size):
            batch_inputs = train_input_data[i:i + batch_size].to(self.device)

            # pass through VAE
            z = self.encode(batch_inputs)

            train_all_z += [z.cpu()]

        train_all_z = torch.cat(train_all_z, dim=0)

        test_all_z = []
        for i in range(0, len(test_input_data), batch_size):
            batch_inputs = test_input_data[i:i + batch_size].to(self.device)

            # pass through VAE
            z = self.encode(batch_inputs)
            test_all_z += [z.cpu()]

        test_all_z = torch.cat(test_all_z, dim=0)

        # dicretize the latent space into bins
        cluster_centers = utils.RegSpaceClustering(train_all_z, min_centers=self.min_centers)
	

        # obtain the cluster labels
        train_distance_matrix = torch.sqrt((torch.square(train_all_z.unsqueeze(1) - cluster_centers.unsqueeze(0))).sum(dim=-1))
        train_cluster_labels = torch.argmin(train_distance_matrix, dim=1)

        test_distance_matrix = torch.sqrt((torch.square(test_all_z.unsqueeze(1) - cluster_centers.unsqueeze(0))).sum(dim=-1))
        test_cluster_labels = torch.argmin(test_distance_matrix, dim=1)


        if log_path!=None:
            elapsed_time = time.time() - start_time
            print('Finished after ' + str(elapsed_time) + 's')
            print('%i cluster centers detected' % len(cluster_centers) + '\n')

            print('Finished after ' + str(elapsed_time) + 's', file=open(log_path, 'a'))
            print('%i cluster centers detected' % len(cluster_centers) + '\n', file=open(log_path, 'a'))

        if save_centers:
            return train_cluster_labels, test_cluster_labels, cluster_centers

        else:
            return train_cluster_labels, test_cluster_labels 


    def resampling(self, train_past_data0, test_past_data0, save_centers, output_path, log_path, index=0, batch_size=128):
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
        output_variables = self.get_cluster_centers(train_past_data0, test_past_data0, save_centers=save_centers, log_path=log_path)

        train_cluster_labels, test_cluster_labels = output_variables[0], output_variables[1]

        # output z cluster centers
        if save_centers:
            cluster_centers = output_variables[2]
            z_cluster_center_path = output_path + '_z_cluster_centers' + str(index) + '.npy'
            np.save(z_cluster_center_path, cluster_centers.cpu().data.numpy())

        num_cluster = int(torch.max(train_cluster_labels).cpu().numpy()) + 1
        # draw samples based on the bias factor (>=1)
        # n_k' ~ n_k^(1/bias_factor)
        cluster_weights = []
        total_weights = 0
        train_cluster_indices = []
        test_cluster_indices = []

        total_effective_samples = 0

        for k in range(num_cluster):
            train_cluster_indices += [torch.nonzero(train_cluster_labels == k, as_tuple=True)[0]]
            test_cluster_indices += [torch.nonzero(test_cluster_labels == k, as_tuple=True)[0]]

            if len(train_cluster_indices[k]) > batch_size:
                total_effective_samples += len(train_cluster_indices[k])

            weight = np.power(len(train_cluster_indices[k]), 1/self.bias_factor)
            cluster_weights += [weight]
            total_weights += weight

        if total_effective_samples < train_past_data0.shape[0]*0.7:
            print(1.0*total_effective_samples/train_past_data0.shape[0])
            print("Too few samples in each bin! Please decrease min_centers!")
            #raise ValueError

        # create better dataset by resampling from each bin
        train_dataset_indices = []
        test_dataset_indices = []

        for k in range(num_cluster):
            train_dataset_size = int(train_past_data0.shape[0] * cluster_weights[k] / total_weights / batch_size + 1) * batch_size
            test_dataset_size = int(test_past_data0.shape[0] * cluster_weights[k] / total_weights / batch_size + 1) * batch_size

            if len(train_cluster_indices[k]) >= train_dataset_size:
                train_dataset_indices += [train_cluster_indices[k][
                                              torch.randperm(len(train_cluster_indices[k]))[
                                              :train_dataset_size]]]
            else:
                train_dataset_indices += [train_cluster_indices[k][
                                              torch.randint(0, len(train_cluster_indices[k]), (train_dataset_size,))]]

            if len(test_cluster_indices[k]) >= test_dataset_size:
                test_dataset_indices += [test_cluster_indices[k][
                                             torch.randperm(len(test_cluster_indices[k]))[
                                             :test_dataset_size]]]
            elif test_cluster_indices[k].shape[0] > 0:
                test_dataset_indices += [test_cluster_indices[k][
                                             torch.randint(0, len(test_cluster_indices[k]), (test_dataset_size,))]]


        train_dataset_indices = torch.cat(train_dataset_indices, dim=0).reshape((-1, batch_size))
        test_dataset_indices = torch.cat(test_dataset_indices, dim=0).reshape((-1, batch_size))

        train_indices = train_dataset_indices[
            torch.randperm((train_dataset_indices).shape[0])].flatten()
        test_indices = test_dataset_indices[
            torch.randperm((test_dataset_indices).shape[0])].flatten()

        return train_indices, test_indices
 
    
    def train_model(self, train_input0, train_input1, train_target0, train_target1, train_setT, 
			test_input0, test_input1, test_target0, test_target1, test_setT, betaT_bins, 
			beta, lr, lr_scheduler_step_size, lr_scheduler_gamma, prior_learning_rate, 
			batch_size, max_epochs, output_path, log_interval, SaveTrainingProgress):
        self.train()

        step = 0        # steps of model updates
        start = time.time()
        os.makedirs(output_path,exist_ok=True)
        log_path = output_path + '/log.txt'
        model_path = output_path + '/tDynAE_model'
        os.makedirs(model_path,exist_ok=True)

        epoch = 0        # cycles of training data set 

        # setup optimizer
        # small learning rate and beta for the first epoch
        beta_current = beta/100
        optimizer = torch.optim.Adam(chain(self.model_encoder.parameters(), self.model_decoder.parameters()), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)

        prior_optimizer = torch.optim.Adam(self.model_prior.parameters(),lr=prior_learning_rate,)        
        
        while epoch < max_epochs:
            if epoch == 0:
                train_permutation = torch.randperm(train_input0.shape[0])
                test_permutation = torch.randperm(test_input0.shape[0])

            elif epoch == 1:
                beta_current = beta
                optimizer = torch.optim.Adam(chain(self.model_encoder.parameters(), self.model_decoder.parameters()), lr=lr)
                scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_scheduler_step_size, gamma=lr_scheduler_gamma)
                train_permutation, test_permutation = self.resampling(train_input0, test_input0, batch_size=batch_size, save_centers=False, output_path=output_path, log_path=log_path)

            else:
                train_permutation, test_permutation = self.resampling(train_input0, test_input0, batch_size=batch_size, save_centers=False, output_path=output_path, log_path=log_path)

            for i in range(0, len(train_permutation), batch_size):
                step += 1
                if (i+batch_size) > len(train_permutation):
                    print(i+batch_size, len(train_permutation))
                    break
                
                train_indices = train_permutation[i:(i+batch_size)]
                temp, input0, input1, target0, target1 = utils.sample_pairwise_minibatch(train_setT, train_input0, train_input1, 
							train_target0, train_target1, train_indices, self.device)

                loss, reconstruction_error, sw_loss, prior_loss = self.calculate_loss(temp, input0, input1, 
							target0, target1, betaT_bins=betaT_bins, beta=beta_current)
                
                if (torch.isnan(loss).any()):
                    print("NAN in training loss")
                    return True

                optimizer.zero_grad()
                prior_optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (torch.isnan(prior_loss).any()):
                    print("NAN in priot loss")
                    return True
                
                optimizer.zero_grad()
                prior_optimizer.zero_grad()
                prior_loss.backward()
                prior_optimizer.step()

                
                if step % 100 == 0:  #output log every 100 steps
                    train_time = time.time() - start
                    
                    print(f"Iteration {step}:\tTime {train_time} s\n loss (train) {loss}\t \
			 reconstruction_err {reconstruction_error}\t sw_loss {sw_loss}\t prior_loss {prior_loss}")
                    print(f"Iteration {step}:\tTime {train_time} s\n loss (train) {loss}\t \
			reconstruction_err {reconstruction_error}\t sw_loss {sw_loss}\t prior_loss {prior_loss}", 
                          file=open(log_path,'a'))
                    
                    j = i%len(test_permutation)
                    if (j+batch_size) > len(test_permutation):
                        j = len(test_permutation) - batch_size
                        
                    test_indices = test_permutation[j:(j+batch_size)]
                    temp, input0, input1, target0, target1 = utils.sample_pairwise_minibatch(test_setT, test_input0, test_input1, 
								test_target0, test_target1, test_indices, self.device)

                    test_loss, test_reconstruction_error, test_sw_loss, test_prior_loss  = self.calculate_loss(temp, input0, input1, 
								target0, target1, betaT_bins=betaT_bins, beta=beta_current)

                    print(f" loss (test) {test_loss}\t reconstruction_err {test_reconstruction_error}\t \
				sw_loss {test_sw_loss}\t prior_loss {test_prior_loss}")
                    print(f" loss (test) {test_loss}\t reconstruction_err {test_reconstruction_error}\t \
				sw_loss {test_sw_loss}\t prior_loss {test_prior_loss}",
                          file=open(log_path,'a'))

                    
                
            epoch += 1
            scheduler.step()
            if scheduler.gamma < 1:
                print("Update lr to %f" % (optimizer.param_groups[0]['lr']))
                print("Update lr to %f" % (optimizer.param_groups[0]['lr']), file=open(log_path, 'a'))
            
            if SaveTrainingProgress:
                if epoch % log_interval == 0:
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

    @torch.no_grad()    
    def output_result(self, train_input0, train_input1, train_target0, train_target1, train_setT, 
			test_input0, test_input1, test_target0, test_target1, test_setT, 
			output_path, beta_current, betaT_bins, batch_size=1024, index=0):

        log_path = output_path + f'/result_log_{index}.txt'
        outputfile = output_path + f'/result_loss_{index}.txt'
        train_permutation, test_permutation = self.resampling(train_input0, test_input0, batch_size=batch_size, save_centers=False, output_path=output_path, log_path=log_path)

        train_prior_loss = []
        train_loss = []
        train_reconstruction_error = []
        train_sw_loss = []

        for i in range(0, len(train_permutation), batch_size):
            if (i+batch_size) > len(train_permutation):
                print(i+batch_size, len(train_permutation))
                break

            train_indices = train_permutation[i:(i+batch_size)]
            temp, input0, input1, target0, target1 = utils.sample_pairwise_minibatch(train_setT, train_input0, train_input1,
                                                        train_target0, train_target1, train_indices, self.device)

            loss, reconstruction_error, sw_loss, prior_loss  = self.calculate_loss(temp, input0, input1,
                                                        target0, target1, betaT_bins=betaT_bins, beta=beta_current)

            train_loss += [loss.detach().cpu().numpy()]
            train_reconstruction_error += [reconstruction_error.detach().cpu().numpy()]
            train_sw_loss += [sw_loss.detach().cpu().numpy()]
            train_prior_loss += [prior_loss.detach().cpu().numpy()]

        train_prior_loss = np.mean(train_prior_loss)
        train_loss = np.mean(train_loss)
        train_reconstruction_error = np.mean(train_reconstruction_error)
        train_sw_loss = np.mean(train_sw_loss)


        test_prior_loss = []
        test_loss = []
        test_reconstruction_error = []
        test_sw_loss = []

        for i in range(0, len(test_permutation), batch_size):
            if (i+batch_size) > len(test_permutation):
                print(i+batch_size, len(test_permutation))
                break

            test_indices = test_permutation[i:(i+batch_size)]
            temp, input0, input1, target0, target1 = utils.sample_pairwise_minibatch(test_setT, test_input0, test_input1,
                                                        test_target0, test_target1, test_indices, self.device)

            loss, reconstruction_error, sw_loss, prior_loss  = self.calculate_loss(temp, input0, input1,
                                                                target0, target1, betaT_bins=betaT_bins, beta=beta_current)


            test_loss += [loss.detach().cpu().numpy()]
            test_reconstruction_error += [reconstruction_error.detach().cpu().numpy()]
            test_sw_loss += [sw_loss.detach().cpu().numpy()]
            test_prior_loss += [prior_loss.detach().cpu().numpy()]

        test_prior_loss = np.mean(test_prior_loss)
        test_loss = np.mean(test_loss)
        test_reconstruction_error = np.mean(test_reconstruction_error)
        test_sw_loss = np.mean(test_sw_loss)

        print(f"train_loss: {train_loss}", file=open(outputfile, 'a'))
        print(f"test_loss: {test_loss}", file=open(outputfile, 'a'))

        print(f"train_reconstruction_error: {train_reconstruction_error}", file=open(outputfile, 'a'))
        print(f"test_reconstruction_error: {test_reconstruction_error}", file=open(outputfile, 'a'))

        print(f"train_sw_loss: {train_sw_loss}", file=open(outputfile, 'a'))
        print(f"test_sw_loss: {test_sw_loss}", file=open(outputfile, 'a'))

        print(f"train_prior_loss: {train_prior_loss}", file=open(outputfile, 'a'))
        print(f"test_prior_loss: {test_prior_loss}", file=open(outputfile, 'a'))

        return False

    @torch.no_grad()
    def save_traj(self, input0, save_path, savefile="save_traj.npy", batch_size=128):
        all_out = []
        all_z = []

        for i in range(0, len(input0), batch_size):
              if (i+batch_size) > len(input0):
                 batch_input = input0[i:].to(self.device)
              else:
                 batch_input = input0[i:i+batch_size].to(self.device)
              out, z = self.forward(batch_input)
              all_out += [out.cpu()]
              all_z += [z.cpu()]

        all_out = torch.cat(all_out, dim=0).data.numpy()
        all_z = torch.cat(all_z, dim=0).data.numpy()

        np.save(save_path+"/out_"+savefile, all_out)
        np.save(save_path+"/z_"+savefile, all_z)

        return False

    @torch.no_grad()
    def evolve_full_dynamics(self, z_init, betaT_infer, infer_steps, save_path, savefile="infer_traj.npy", dt_infer=1, batch_size=128):
        z_traj = self.evolve_latent_dynamics(z_init, betaT_infer, infer_steps, dt_infer) 

        all_out = []
        for i in range(0, len(z_traj), batch_size):
              if (i+batch_size) > len(z_traj):
                 batch_input = z_traj[i:].to(self.device)
              else:
                 batch_input = z_traj[i:i+batch_size].to(self.device)
              out = self.decode(batch_input)
              all_out += [out.cpu()]

        out_traj = torch.cat(all_out, dim=0).data.numpy()

        np.save(f"{save_path}/out_{savefile}", out_traj)
        np.save(f"{save_path}/z_{savefile}", z_traj.cpu().data.numpy())

        return False
