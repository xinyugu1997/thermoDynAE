import numpy as np
import pyemma

def TPT_TH_detour(data,t_lag=1):
    A = np.loadtxt("TPT_TH_stateA.id")
    B = np.loadtxt("TPT_TH_stateB.id")
    X = np.loadtxt("TPT_TH_stateX.id")
    cluster = pyemma.load('kmeans200.pyemma', model_name='cluster200')
    # estimate MSM
    msm = pyemma.msm.estimate_markov_model(cluster.assign(X=data), lag=t_lag)
    # TPT
    A_active = np.where(np.isin(msm.active_set, A))[0]
    B_active = np.where(np.isin(msm.active_set, B))[0]
    X_active = np.where(np.isin(msm.active_set, X))[0]
    flux = pyemma.msm.tpt(msm, A_active, B_active)

    # paths cross X are assigned into detour paths; compute detour percentage
    paths, path_fluxes = flux.pathways(fraction=1.0, maxiter=5000)    
    P = path_fluxes/np.sum(path_fluxes)
    detour = 0
    for i, path in enumerate(paths):
        if bool(set(X_active) & set(path)): 
            detour += P[i]
    return detour
