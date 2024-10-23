import numpy as np

def double_well(z, a=1, beta_0=1):
    return ((z-a)*(z+a))**2/beta_0/a**4

def double_well_force(z, a=1, beta_0=1):
    return 4*(z/a**2-z**3/a**4)/beta_0

#def three_hole(x, y, beta_0=1, sigma=10):
#    return -1/beta_0*(3*np.exp(-x**2/sigma**2)*(np.exp(-(y-5*sigma/3)**2/sigma**2) - np.exp(-(y-sigma/3)**2/sigma**2))+(4*np.exp(-y**2/sigma**2)*(np.exp(-(x-sigma)**2/sigma**2) + np.exp(-(x+sigma)**2/sigma**2))))
#
#def three_hole_force(z, beta_0=1,sigma=10):
#    z = np.reshape(z, (-1,2))
#    x = z[:, 0]
#    y = z[:, 1]
#    force_x = 1/beta_0*(3*np.exp(-x**2/sigma**2)*(-2*x/sigma**2)*(np.exp(-(y-5*sigma/3)**2/sigma**2) - np.exp(-(y-sigma/3)**2/sigma**2))+(4*np.exp(-y**2/sigma**2)*(np.exp(-(x-sigma)**2/sigma**2)*(-2*(x-sigma)/sigma**2) + np.exp(-(x+sigma)**2/sigma**2)*(-2*(x+sigma)/sigma**2))))
#    force_y = 1/beta_0*(3*np.exp(-x**2/sigma**2)*(np.exp(-(y-5*sigma/3)**2/sigma**2)*(-2*(y-5*sigma/3)/sigma**2) - np.exp(-(y-sigma/3)**2/sigma**2)*(-2*(y-sigma/3)/sigma**2))+(4*np.exp(-y**2/sigma**2)*(-2*y/sigma**2)*(np.exp(-(x-sigma)**2/sigma**2) + np.exp(-(x+sigma)**2/sigma**2))))
#    return np.array([force_x, force_y]).T

def three_hole(x, y, beta_0=1, sigma=1):
    return -1/beta_0*(-x**2-(y-2*sigma/3)**2+6*np.exp(-x**2/sigma**2)*(np.exp(-(y-5*sigma/3)**2/sigma**2) - np.exp(-(y-sigma/3)**2/sigma**2))+(8*np.exp(-y**2/sigma**2)*(np.exp(-(x-sigma)**2/sigma**2) + np.exp(-(x+sigma)**2/sigma**2))))

def three_hole_force(z, beta_0=1,sigma=1):
    z = np.reshape(z, (-1,2))
    x = z[:, 0]
    y = z[:, 1]
    force_x = 1/beta_0*(-2*x+6*np.exp(-x**2/sigma**2)*(-2*x/sigma**2)*(np.exp(-(y-5*sigma/3)**2/sigma**2) - np.exp(-(y-sigma/3)**2/sigma**2))+(8*np.exp(-y**2/sigma**2)*(np.exp(-(x-sigma)**2/sigma**2)*(-2*(x-sigma)/sigma**2) + np.exp(-(x+sigma)**2/sigma**2)*(-2*(x+sigma)/sigma**2))))
    force_y = 1/beta_0*(-2*(y-2*sigma/3) + 6*np.exp(-x**2/sigma**2)*(np.exp(-(y-5*sigma/3)**2/sigma**2)*(-2*(y-5*sigma/3)/sigma**2) - np.exp(-(y-sigma/3)**2/sigma**2)*(-2*(y-sigma/3)/sigma**2))+(8*np.exp(-y**2/sigma**2)*(-2*y/sigma**2)*(np.exp(-(x-sigma)**2/sigma**2) + np.exp(-(x+sigma)**2/sigma**2))))
    return np.array([force_x, force_y]).T

def update_position(z, v, dt):
    return z+v*dt/2

def update_velocity_from_U(v, force, m, dt):
    return v+force*dt/2/m

def update_velocity_from_OU(v, gamma, dt, beta, m):
    c = np.exp(-gamma*dt)
    return v*c + np.sqrt((1-c*c)/m/beta)*np.random.randn(*v.shape)

def baoab(force, ini_z, ini_v, beta, dt=0.001, gamma=10, m=1, steps=10000000, outfreq=1000):
    traj_z = []
    traj_v = []
    traj_t = []
    
    z = ini_z
    v = ini_v
    
    f = np.squeeze(force(z))
    for istep in range(1, steps+1):        
        # B
        v = update_velocity_from_U(v, f, m, dt)
        
        # A
        z = update_position(z, v, dt)
        
        # O
        v = update_velocity_from_OU(v, gamma, dt, beta, m)
        
        # A
        z = update_position(z, v, dt)
        
        f = np.squeeze(force(z))
        # B
        v = update_velocity_from_U(v, f, m, dt)
        
        if istep % outfreq == 0:
            traj_t.append(dt*istep)
            traj_z.append(z)
            traj_v.append(v)  
            
        if istep % (100*outfreq) == 0:
            print(f'step:{istep}  beta:{beta}  z:{z}', file=open(f'log_beta{beta}.txt','a'))
    
    return np.array(traj_t), np.array(traj_z), np.array(traj_v)
