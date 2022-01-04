import numpy as np
from time import sleep
from os.path import exists as ospe
from numba import njit, prange
from mpi4py import MPI
comm = MPI.COMM_WORLD


def apply_velocity_model(Par,rho_itp,v_itp):
    
    @njit(parallel=True,  cache=True)
    def apply_velocity_model_ini(rho_itp,rho_0,targeted_rms,v_itp,alpha,folder_saving):
        delta_plus_one    = rho_itp/rho_0 
        var_tgt           = targeted_rms**2
        var_v_itp         = np.var(v_itp)/100**2
        var_rho_itp_alpha = np.mean(delta_plus_one**alpha)
        beta              = (var_tgt - var_v_itp)/var_rho_itp_alpha
        return beta
    
    @njit(parallel=True,  cache=True)
    def apply_velocity_model_4all(rho_itp,rho_0,v_itp,alpha,beta):
        dispersions    = np.zeros_like(rho_itp)
        for p in prange(len(rho_itp)):
            dispersions[p] = np.sqrt(beta*(rho_itp[p]/rho_0)**alpha)
        return v_itp/100,dispersions
    
    if Par['velocity']:
        
        if comm.Get_rank() == 0 and not ospe(Par['folder_beta']+'beta'):
            beta = apply_velocity_model_ini(rho_itp,Par['rho_0'],Par['targeted_rms'],v_itp,Par['alpha'],Par['folder_beta'])
            np.savetxt(Par['folder_beta']+'beta',[beta])
        while not ospe(Par['folder_beta']+'beta'): sleep(10)
        beta = np.loadtxt(Par['folder_beta']+'beta')
        v_itp,dispersions = apply_velocity_model_4all(rho_itp,Par['rho_0'],v_itp,Par['alpha'],beta)
        v_cat             = np.random.normal(v_itp,dispersions)
    
    else: v_cat = 'nope'
        
    return v_cat

def apply_RSD(xc,yc,zc,vx,vy,vz,z_snap,Omega_m,L,PPA=True): # plane parallel approximation here
    def Eofz(z,Omega_m): 
            return np.sqrt(Omega_m*(1+z)**3 + (1 - Omega_m))
        
    if PPA: vp = vz
    
    else:
        rc = np.sqrt(xc**2+yc**2+zc**2)
        vp = vx*xc/rc + vy*yc/rc + vz*zc/rc
    
    factor = vp * (1+z_snap)/(100*Eofz(z_snap,Omega_m)*np.sqrt(1+z_snap))
    
    if PPA: 
        xo = xc ; yo = yc ; zo = zc + factor
        out_right = ((zo >  L/2) == True)
        out_left  = ((zo < -L/2) == True)
        
        zo[out_right] = zo[out_right] - L
        zo[out_left]  = zo[out_left]  + L
        
    else  : 
        xo = xc + xc/rc * factor ; yo = yc + yc/rc * factor ; zo = zc + zc/rc * factor
        out_right_x = ((xo >  L/2) == True)
        out_left_x  = ((xo < -L/2) == True)
        
        out_right_y = ((yo >  L/2) == True)
        out_left_y  = ((yo < -L/2) == True)
        
        out_right_z = ((zo >  L/2) == True)
        out_left_z  = ((zo < -L/2) == True)
        
        xo[out_right] = xo[out_right] - L
        xo[out_left]  = xo[out_left]  + L
        
        yo[out_right] = yo[out_right] - L
        yo[out_left]  = yo[out_left]  + L
        
        zo[out_right] = zo[out_right] - L
        zo[out_left]  = zo[out_left]  + L
        
    return xo,yo,zo;