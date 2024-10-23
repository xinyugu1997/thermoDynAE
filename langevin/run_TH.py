import numpy as np
import pandas as pd
import baoab_langevin as ld 
import sys

#b = np.arange(0.5,2.1,0.1)
#for beta in b:
#    beta = np.round(beta,2)
beta = float(sys.argv[1])
df = pd.DataFrame()
traj_t, traj_z, traj_v = ld.baoab(ld.three_hole_force, ini_z=np.array([0.0, 0.0]), ini_v=np.array([0.0, 0.0]), beta=beta, steps=100000000, outfreq=1000)
df['Time'] = traj_t
df['zx'] = traj_z[:,0]
df['zy'] = traj_z[:,1]
df['vx'] = traj_v[:,0]
df['vy'] = traj_v[:,1]
df.to_csv(f'TH_beta_{beta}_traj.csv', header=True, index=False,)
