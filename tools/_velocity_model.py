import numpy as np
from time import sleep
from os.path import exists as ospe
from os.path import join as ospj
from numba import njit, prange
from mpi4py import MPI
from glob import glob
comm = MPI.COMM_WORLD

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def from_globalcomm_to_intranode_comm():
    '''
    generate a mpi communicator between processes sharing the same RAM, usefull when sharing common numpy arrays
    '''
    processor_name     = MPI.Get_processor_name()
    processor_name_all = comm.gather(processor_name, root=0)
    
    if rank == 0 :
        unique = np.unique(processor_name_all)
    else : unique = None
    
    unique = comm.scatter([unique for i in range(size)], root=0)
    
    billy_color = np.arange(unique.size)
    billy_index = np.where(unique==processor_name)[0][0]
    local_billy = billy_color[billy_index]
        
    intracomm = MPI.Comm.Split(comm,local_billy)
    return intracomm


def apply_velocity_model(Par,rho_itp,v_itp):
    if Par['velocity']:
        
        intracomm = from_globalcomm_to_intranode_comm() #need node synchronisation! 

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
    
        if not ospe(Par['folder_beta']+'beta'):
            beta = apply_velocity_model_ini(rho_itp,Par['rho_0'],Par['targeted_rms'],v_itp,Par['alpha'],Par['folder_beta'])
            np.savetxt(Par['folder_beta']+'beta_rank_%i'%comm.Get_rank(),[beta])
            comm.Barrier()
            sleep(1)
            if comm.Get_rank() == 0:
                beta_names = glob(ospj(Par['folder_beta'],'*'))
                beta_all = []
                for i in range(len(beta_names)):
                    beta_all.append(np.loadtxt(beta_names[i]))
                np.savetxt(Par['folder_beta']+'beta',[np.mean(beta)])
                sleep(2)
            comm.Barrier()
            if Par['total_number_of_cat'] >= 1000 :  sleep(intracomm.Get_rank()*180)
            else :                                   sleep(intracomm.Get_rank()*60)
            sleep(comm.Get_rank()/2)
            
        while not ospe(Par['folder_beta']+'beta'):
            print('waiting for beta estimate',flush=True)
            sleep(3)
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