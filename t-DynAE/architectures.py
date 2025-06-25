"""
adopted from Dedi's DynAE work, modified by Xinyu

"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class fc_encoder(nn.Module):
    def __init__(self, z_dim, data_shape, neuron_num1=16):
        super().__init__()
        self.z_dim = z_dim
        self.data_shape = data_shape
        self.neuron_num1 = neuron_num1

        self.encoder_input_layer = nn.Sequential(
             nn.Linear(np.prod(data_shape), self.neuron_num1), 
             nn.ReLU())

        modules = []
        for _ in range(2):
            modules += [nn.Linear(self.neuron_num1, self.neuron_num1)]
            modules += [nn.ReLU()]
        self.encoder = nn.Sequential(*modules)
   
        self.encoder_output_layer = nn.Linear(self.neuron_num1, z_dim)

class fc_decoder(nn.Module):
    def __init__(self, z_dim, output_shape, neuron_num2=16):
        super().__init__()
        self.z_dim = z_dim
        self.output_shape = output_shape
        self.neuron_num2 = neuron_num2

        self.decoder_input_layer = nn.Sequential(
             nn.Linear(self.z_dim, self.neuron_num2),
             nn.ReLU())

        modules = []
        for _ in range(2):
            modules += [nn.Linear(self.neuron_num2, self.neuron_num2)]
            modules += [nn.ReLU()]
        self.decoder = nn.Sequential(*modules)

        self.decoder_output_layer = nn.Linear(self.neuron_num2, self.output_shape)


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=10.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = x * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)

class Langevin_prior(nn.Module):
    def __init__(self, z_dim, device, ConstantDiffusionPrior=True, reduced_force=True, 
                 reduced_force_T_noise=0, embed_dim=64, embed_scale=10., neuron_num=32):
        super().__init__()
        self.z_dim = z_dim
        self.device = device
        self.neuron_num = neuron_num
        self.ConstantDiffusionPrior = ConstantDiffusionPrior
        self.reduced_force = reduced_force
        self.reduced_force_T_noise = reduced_force_T_noise

        if ConstantDiffusionPrior:
            print("Constant Diffusion Matrix!!")
            # ConstantDiffusion, without loss of generality, matrix M_ij = delta_ij
            #self.constant_logM = nn.Parameter(torch.randn(1, z_dim))
        else:
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

        if self.reduced_force:
            print(f"reduced_force, temperature dependent!! T_noise during training:{reduced_force_T_noise}.")
            print(f"temperature embed_dim:{embed_dim} embed_scale:{embed_scale}")
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim, scale=embed_scale),
            nn.Linear(embed_dim, embed_dim))

        self.act = lambda x: x * torch.sigmoid(x)
        self.linear1 = nn.Linear(z_dim, neuron_num)
        self.dense1 = Dense(embed_dim, neuron_num)
        self.linear2 = nn.Linear(neuron_num, neuron_num)
        self.dense2 = Dense(embed_dim, neuron_num)
        self.linear3 = nn.Linear(neuron_num, z_dim)


    def prior_logA(self, z):
        return self.logA_net(z)

    def prior_EA(self, z):
        return self.ea_net(z)

    def prior_force(self, z, betaT):
        embed = self.act(self.embed(betaT))
        h = self.linear1(z)
        ## Incorporate information from betaT
        h += self.dense1(embed)
        h = F.tanh(h)
        h = self.linear2(h)
        h += self.dense2(embed)
        h = F.tanh(h)
        h = self.linear3(h)
        return h


        
