"""
Adopted from DynAE, modified by Xinyu

DynamicsAE: A deep learning-based framework to uniquely identify an uncorrelated,
isometric and meaningful latent representation. Code maintained by Dedi.

Read and cite the following when using this method:
https://arxiv.org/abs/2209.00905
"""
import torch
import numpy as np
import pyemma

def TICA_prep(traj_data, lag=100):
    # traj_data should be (Nframes, dim_X) shaped
    traj_meaned = traj_data - np.mean(traj_data, axis=0)
    tica = pyemma.coordinates.tica(traj_meaned, lag, var_cutoff=1)
    eigenvectors = tica.eigenvectors

    # output mean_free traj_data and column-wise tica eigenvectors
    return traj_meaned, eigenvectors   


def data_init(traj_data, traj_target, traj_betaT, t0=0, dt=0):
    # This function generates the datasets for training
    # pairing X0 and X1, together w. effective inverse temperature betaT = T0/T

    assert len(traj_data) == len(traj_target)
    assert len(traj_data) == len(traj_betaT)

    # skip the first t0 data
    betaT_data0 = traj_betaT[t0:(len(traj_data) - dt - 1)]
    past_data0 = traj_data[t0:(len(traj_data) - dt - 1)]
    past_data1 = traj_data[(t0 + 1):(len(traj_data) - dt)]

    target0 = traj_target[(t0 + dt):(len(traj_data) - 1)]
    target1 = traj_target[(t0 + dt + 1):len(traj_data)]

    # data shape
    data_shape = past_data0.shape[1:]

    n_data = len(past_data0)

    # 90% random test/train split
    p = np.random.permutation(n_data)
    betaT_data0 = betaT_data0[p]
    past_data0 = past_data0[p]
    past_data1 = past_data1[p]
    target0 = target0[p]
    target1 = target1[p]

    train_betaT_data0 = betaT_data0[0: (8 * n_data) // 10]
    test_betaT_data0 = betaT_data0[(8 * n_data) // 10:]
    
    train_past_data0 = past_data0[0: (8 * n_data) // 10]
    test_past_data0 = past_data0[(8 * n_data) // 10:]

    train_past_data1 = past_data1[0: (8 * n_data) // 10]
    test_past_data1 = past_data1[(8 * n_data) // 10:]

    train_target_data0 = target0[0: (8 * n_data) // 10]
    test_target_data0 = target0[(8 * n_data) // 10:]

    train_target_data1 = target1[0: (8 * n_data) // 10]
    test_target_data1 = target1[(8 * n_data) // 10:]

    return data_shape, train_betaT_data0, train_past_data0, train_past_data1, train_target_data0, train_target_data1, \
           test_betaT_data0, test_past_data0, test_past_data1, test_target_data0, test_target_data1

def sample_pairwise_minibatch(betaT_data, past_data0, past_data1, target_data0, target_data1, indices, device):
    sample_betaT_data = betaT_data[indices].to(device)
    sample_past_data0 = past_data0[indices].to(device)
    sample_past_data1 = past_data1[indices].to(device)
    sample_target_data0 = target_data0[indices].to(device)
    sample_target_data1 = target_data1[indices].to(device)

    return sample_betaT_data, sample_past_data0, sample_past_data1, sample_target_data0, sample_target_data1


def D_KL(dist1, dist2, epsilon=1e-12):
    if len(dist1) != len(dist2):
       raise ValueError('two distributions must share the same bins')
    dist1 = dist1/np.sum(dist1)
    dist2 = dist2/np.sum(dist2)
    dist1 = np.clip(dist1, epsilon, 1)
    dist2 = np.clip(dist2, epsilon, 1)
    return np.sum(dist1*np.log(dist1/dist2))
	

def rand_projections(z_dim, num_samples=50):
    # This function generates `num_samples` random samples from the latent space's unit sphere
    projections = [w / np.sqrt((w**2).sum())
                   for w in np.random.normal(size=(num_samples, z_dim))]
    projections = torch.from_numpy(np.array(projections)).float()
    return projections

# Only used for unweighted samples
def sliced_wasserstein_distance(encoded_samples, prior_samples, betaT_samples, betaT_bins, projection_num=50, p=2, device='cpu'):
    # This function calculates the sliced-Wasserstein distance between the encoded samples and prior samples

    # derive latent space dimension size from random samples drawn from latent prior distribution
    z_dim = prior_samples.size(-1)

    # generate random projections in latent space
    projections = rand_projections(z_dim, projection_num).to(device)
    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
    # calculate projections through the prior distribution random samples
    prior_projections = prior_samples.matmul(projections.transpose(0, 1))
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection # for each betaT bin
    
    wasserstein_distance = []
    #ratio_betaT_bins = []
    for ibeta in range(1, len(betaT_bins)):
        T_index = np.where((betaT_samples < betaT_bins[ibeta]) & (betaT_samples >= betaT_bins[ibeta-1]))[0]
        #ratio_betaT_bins.append(len(T_index))
        if len(T_index) > 0:
            wasserstein_diff = (torch.sort(encoded_projections[T_index], dim=0)[0] -
                                    torch.sort(prior_projections[T_index], dim=0)[0])
            # distance between latent space prior and encoded distributions
            # power of 2 by default for Wasserstein-2
            wasserstein_diff = torch.pow(wasserstein_diff, p)
            wasserstein_distance += [wasserstein_diff]

    wasserstein_distance = torch.cat(wasserstein_distance, dim=0)
    if len(wasserstein_distance) < ( 0.9 * len(betaT_samples) ):
        raise ValueError('Please extend the range of betaT bins!!')
    
    #print("ratio_betaT_bins:", np.array(ratio_betaT_bins)/len(encoded_samples))
 
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()


# If using TICA mode, input X for each traj must be mean-free for TICA and training
def proj_TICA(X, b, vb, mTICA):
    """
    Args:
        X:      (N, m)      full coord batch
        b:      (N, 1)      betaT batch
        vb:     (L)         reference betaT
        mTICA:  (L, m, m)   per-betaT TICA transformation matrices, torch.stack([ev,ev,...,ev],dim=0), ev: columnwise eigenvectors

    Returns:
        (N, m) TICA projection batch
    """
    # scaled sharp softmax to find reference betaT-TICA
    if len(vb) == 1:   # single training temperature
       scale = 1
    else:
       scale = (torch.max(vb) - torch.min(vb)) /100
    diff_sq = ((b - vb)/scale) ** 2  
    weights = torch.softmax(-diff_sq, dim=1)
 
    # find reference betaT-TICA (batch_size × m × m)
    TICAs = torch.einsum('nl,lmk->nmk', weights.float(), mTICA)  # (N, m, m)
    # projecting
    out = torch.einsum('nk,nkm->nm', X, TICAs) # (N, m)

    return out


def rand_normal(batch_size, dim):
    """ This function generates 2D samples from a uniform distribution in a 2-dimensional space

        Args:
            batch_size (int): number of batch samples
            dim (int): dimension of the Gaussian

        Return:
            torch.Tensor: tensor of size (batch_size, dim)
    """
    z =  np.random.normal(size=(batch_size, dim))
    return torch.from_numpy(z).type(torch.FloatTensor)

def rand_padding(input_data, f_dim, sigma, cum=True):
    """ This function pads input_data along dimension 1 to reach f_dim with a Gaussian distribution 

        Args:
            input_data (tensor): tensor to be padded
            f_dim (int): final value of input_data.shape[1]
            sigma (float): variance of the Gaussian
            cum (bool): padding with Brownian motion if True; True by default            

        Return:
            torch.Tensor: tensor of size (batch_size, f_dim)
    """
    a_dim = int(f_dim - input_data.shape[1])
    #pad = sigma * rand_normal(input_data.shape[0], a_dim)
    pad = sigma * np.random.normal(size=(input_data.shape[0], a_dim)) 
    if cum:
        pad = np.cumsum(pad, axis=0)
        pad = (pad - np.mean(pad, axis=0))/np.std(pad, axis=0)  #whitening

    pad = torch.from_numpy(pad).type(torch.FloatTensor)
    return torch.cat((input_data, pad), dim=1)


def rand_uniform(batch_size, dim):
    """ This function generates 2D samples from a uniform distribution in a 2-dimensional space

        Args:
            batch_size (int): number of batch samples
            dim (int): dimension of the uniform distribution

        Return:
            torch.Tensor: tensor of size (batch_size, 2)
    """
    z = 2 * (np.random.uniform(size=(batch_size, dim)) - 0.5)
    return torch.from_numpy(z).type(torch.FloatTensor)

@torch.no_grad()
def RegSpaceClustering(z_data, min_centers=200, batch_size=128, dist_decay=0.9):
    '''
    Regular space clustering.
        Args:
            z_data: ndarray containing (n,d)-shaped float data
            min_centers: the minimum number of cluster centers to be determined, integer greater than 0 required

        Returns:
            cluster_centers: ndarray containing the cluster centers
    '''
    n_samples, z_dim = z_data.shape    
    cluster_centers = z_data[0:1,:].clone()
    # initialize min_dist based on data varience
    min_dist = torch.sqrt(torch.var(z_data, dim=0).sum())/2

    while len(cluster_centers) < min_centers:
        cluster_centers = z_data[0:1,:].clone()
        i = 1
        while i < n_samples:
            batch = z_data[i:min(i+batch_size, n_samples)]
            dist = torch.sqrt(torch.square(batch.unsqueeze(1)-cluster_centers.unsqueeze(0)).sum(-1))
            indices = torch.nonzero(torch.all(dist>min_dist, dim=-1), as_tuple=True)[0]
            if len(indices) > 0:
                cluster_centers = torch.cat((cluster_centers, batch[indices[0]].reshape(1,z_dim)), dim=0)
                i += indices[0]
            else:
                i += batch_size                
        
        min_dist = min_dist*dist_decay
        
    return cluster_centers 
