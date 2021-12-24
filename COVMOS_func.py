import os
import numpy as np
import sys
from mpi4py import MPI

intracomm = from_globalcomm_to_intranode_comm()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
    
def Eofz(z,Omega_m): 
    return np.sqrt(Omega_m*(1+z)**3 + (1 - Omega_m))

def generate_and_clean_repertories_if_necessary(folder_saving,total_number_of_cat,key_name_of_file,savecat,RSD,savePkcat,rank):
    if rank == 0: 
        if not os.path.exists(folder_saving):        os.makedirs(folder_saving)
        if not os.path.exists(folder_saving+'_job'): os.makedirs(folder_saving+'_job')
        
        if savecat:
            for i in range(total_number_of_cat):
                if os.path.exists(folder_saving+'_job/_%i'%(i+1)) and not os.path.exists(folder_saving+key_name_of_file+'_%i.data'%(i+1)):
                    os.remove(folder_saving+'_job/_%i'%(i+1))
                if os.path.exists(folder_saving+key_name_of_file+'_%i.data'%(i+1)) and not os.path.exists(folder_saving+'_job/_%i'%(i+1)):
                    np.savetxt(folder_saving+'_job/_%i'%(i+1),[1])

        if not savecat and savePkcat:
            for i in range(total_number_of_cat):
                if not RSD:
                    if os.path.exists(folder_saving+'_job/_%i'%(i+1)) and not os.path.exists(folder_saving+key_name_of_file+'_%i.data_Pk'%(i+1)):
                        os.remove(folder_saving+'_job/_%i'%(i+1))
                        if os.path.exists(folder_saving+key_name_of_file+'_%i.data'%(i+1)):
                            os.remove(folder_saving+key_name_of_file+'_%i.data'%(i+1))
                    if os.path.exists(folder_saving+key_name_of_file+'_%i.data_Pk'%(i+1)) and not os.path.exists(folder_saving+'_job/_%i'%(i+1)):
                        np.savetxt(folder_saving+'_job/_%i'%(i+1),[1])

                    if os.path.exists(folder_saving+'_job/_%i'%(i+1)) and os.path.exists(folder_saving+key_name_of_file+'_%i.data_Pk'%(i+1)):
                        if os.path.exists(folder_saving+key_name_of_file+'_%i.data'%(i+1)):
                            os.remove(folder_saving+key_name_of_file+'_%i.data'%(i+1))

                if RSD:
                    if os.path.exists(folder_saving+'_job/_%i'%(i+1)) and not os.path.exists(folder_saving+key_name_of_file+'_%i.data_PkRSD'%(i+1)):
                        os.remove(folder_saving+'_job/_%i'%(i+1))
                        if os.path.exists(folder_saving+key_name_of_file+'_%i.data'%(i+1)):
                            os.remove(folder_saving+key_name_of_file+'_%i.data'%(i+1))
                    if os.path.exists(folder_saving+key_name_of_file+'_%i.data_PkRSD'%(i+1)) and not os.path.exists(folder_saving+'_job/_%i'%(i+1)):
                        np.savetxt(folder_saving+'_job/_%i'%(i+1),[1])
                        
                    if os.path.exists(folder_saving+'_job/_%i'%(i+1)) and os.path.exists(folder_saving+key_name_of_file+'_%i.data_PkRSD'%(i+1)):
                        if os.path.exists(folder_saving+key_name_of_file+'_%i.data'%(i+1)):
                            os.remove(folder_saving+key_name_of_file+'_%i.data'%(i+1))
    return

def find_a_sim_number_to_run(total_number_of_cat,folder_saving,key_name_of_file):
    np.random.seed()
    ramdom_sim      = int(np.random.random()*total_number_of_cat+1)
    name_of_the_sim = folder_saving+key_name_of_file+'_%i'%ramdom_sim
    name_of_the_job = folder_saving+'_job/'+'_%i'%ramdom_sim
    return name_of_the_sim,name_of_the_job

@njit(parallel = True, cache=True)
def inverse_div_theta(complex1,byprod_pk_velocity,k_3D_2):
    array = np.zeros_like(complex1)
    sampling = array.shape[0]
    for i in prange(sampling):
        for j in prange(sampling):
            for k in prange(sampling):
                if k_3D_2[i,j,k] != 0:
                    array[i,j,k] = -1j*(complex1[i,j,k]*byprod_pk_velocity[i,j,k])/k_3D_2[i,j,k]
    return array 

@njit(parallel=True,cache=True)
def array_times_array(array1,array2):
    array_ = np.zeros_like(array1)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                array_[i,j,k] = array1[i,j,k]*array2[i,j,k]
    return array_

@njit(parallel=True,cache=True)
def from_delta_to_rho_times_a3(delta,rho_0,a3):
    array_ = np.zeros_like(delta)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                    array_[i,j,k] = (delta[i,j,k]+1)*rho_0 * a3
    return array_

@njit(parallel=True, cache=True)
def mean_delta_times_a3(delta,a3,rho_0):
    rho_       = np.zeros_like(delta)
    mean_grid_ = np.zeros_like(delta)
    sampling = mean_grid_.shape[0]
    
    for i in prange(sampling):
        for j in prange(sampling):
            for k in prange(sampling):
                rho_[i,j,k] = (delta[i,j,k]+1)*rho_0
                
    for i in prange(sampling):
        for j in prange(sampling):
            for k in prange(sampling):
                ip1 = (i != (sampling-1)) * (i+1)
                jp1 = (j != (sampling-1)) * (j+1)
                kp1 = (k != (sampling-1)) * (k+1)
                
                mean_grid_[i,j,k] = a3 *(rho_[i,j,k] + rho_[ip1,j,k] + rho_[i,jp1,k] + rho_[i,j,kp1] + rho_[ip1,jp1,k] + rho_[ip1,j,kp1] + rho_[i,jp1,kp1] + rho_[ip1,jp1,kp1]) / 8
    return rho_,mean_grid_

def apply_velocity_model(rho_itp,rho_0,targeted_rms,v_itp,alpha,RSD,comm,folder_saving):
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
    
    if RSD:
        if comm.Get_rank() == 0 and not os.path.exists(folder_saving+'beta'):
            beta = apply_velocity_model_ini(rho_itp,rho_0,targeted_rms,v_itp,alpha,folder_saving)
            np.savetxt(folder_saving+'beta',[beta])
        while not os.path.exists(folder_saving+'beta'):
            time.sleep(10)
        beta = np.loadtxt(folder_saving+'beta')
        v_itp,dispersions = apply_velocity_model_4all(rho_itp,rho_0,v_itp,alpha,beta)
        v_cat             = np.random.normal(v_itp,dispersions)
    else:  
        v_cat = 'nope'
    return v_cat


def save_and_or_analyse_cat(savecat,redshift,name_of_the_sim,RSD,tot_obj,cat,v_cat,detached_Pk_estimate,L,savePkcat,comm):
    if savecat: save_catalogue(redshift,name_of_the_sim,RSD,tot_obj,cat,v_cat)
    if savePkcat:
        if detached_Pk_estimate :
            if not savecat : save_catalogue(redshift,name_of_the_sim,RSD,tot_obj,cat,v_cat) # the cat must be saved for detached Pk estimate
            Pk_poles_estimate_detached(name_of_the_sim+'.data',RSD,redshift,savecat,L)
        if not detached_Pk_estimate:
            Pk_poles_estimate(name_of_the_sim,RSD,redshift,savecat,tot_obj,cat,v_cat,L,comm)
    return 

def Pk_poles_estimate_detached(filename,RSD,true_redshift,savecat,L):
    pid = spr.Popen(['ssh', '-o', 'StrictHostKeyChecking=no',socket.gethostname(),'python','-W','ignore','/renoir/baratta/COVMOS/Pk_estimate.py',filename,str(RSD),str(true_redshift),str(L),str(savecat),'-c','&'])
    return

def Pk_poles_estimate(name_of_the_sim,RSD,true_redshift,savecat,tot_obj,cat,v_cat,L,comm):
    
    from nbodykit.source.catalog import ArrayCatalog
    from nbodykit.lab import *
    from nbodykit import use_mpi, CurrentMPIComm
    
    
    singlecomm = MPI.COMM_SELF # compute on a single thread without MPI
    
    def save_Pk_in_right_way(Pknbk,RSD_here,filename,L):
        poles = Pknbk.poles
        nk      = int(512/2 - 1)
        k_new   = np.zeros(nk-1)
        num_new = np.zeros(nk-1)
        Pk_new  = np.zeros((nk-1)*len((0,2,4)))
        kF      = (2*np.pi)/L
        for i in range(1,poles['k'].size) :
            norm = poles['k'][i]
            if norm <= ((nk-1)*kF + kF/2.) :
                k_new[int(np.floor((norm-kF/2.)/kF))] += norm*poles['modes'][i]/2.
                num_new[int(np.floor((norm-kF/2.)/kF))] += poles['modes'][i]/2.
                for p in range(len((0,2,4))) :
                    Pk_new[int(np.floor((norm-kF/2.)/kF)) + (nk-1)*p] += poles['power_'+str((0,2,4)[p])][i].real*poles['modes'][i]/2.
        k_new /= num_new
        for p in range(len((0,2,4))): Pk_new[p*k_new.size : k_new.size*(p+1)] /= num_new
        Pk_new[0 : k_new.size] -= poles.attrs['shotnoise']
        if RSD_here : extension = '_PkRSD'
        else        : extension = '_Pk'
        np.savetxt(filename+extension,np.transpose(np.vstack((k_new,np.ones(len(k_new))*tot_obj,Pk_new[0 : k_new.size],Pk_new[k_new.size : k_new.size*2],Pk_new[k_new.size*2 : k_new.size*3]))))
        return

    #comoving space
    data = np.empty(tot_obj, dtype=[('Position', (np.float32, 3))])
    data['Position'][:,0] = cat[0,:] ; data['Position'][:,1] = cat[1,:] ; data['Position'][:,2] = cat[2,:]
    f = ArrayCatalog({'Position' : data['Position']},comm=singlecomm) ; del data
    mesh_comov = f.to_mesh(Nmesh=512,BoxSize=L,dtype=np.float32,interlaced=True,compensated=True,resampler='pcs',position='Position') ; del f

    #redshift space
    if RSD:
        GADGET = 100*np.sqrt(1+true_redshift)
        cat[0,:],cat[1,:],cat[2,:] = apply_RSD(cat[0,:],cat[1,:],cat[2,:],v_cat[0,:]*GADGET,v_cat[1,:]*GADGET,v_cat[2,:]*GADGET,true_redshift,L)

        data = np.empty(tot_obj, dtype=[('Position', (np.float32, 3))])
        data['Position'][:,0] = cat[0,:] ; data['Position'][:,1] = cat[1,:] ; data['Position'][:,2] = cat[2,:]
        f = ArrayCatalog({'Position' : data['Position']},comm=singlecomm) ; del data
        mesh_redshift = f.to_mesh(Nmesh=512,BoxSize=L,dtype=np.float32,interlaced=True,compensated=True,resampler='pcs',position='Position') ; del f

    #comobile PK        
    Pknbk_comob = FFTPower(mesh_comov, mode='2d', poles = (0,2,4), dk=0, kmin=2*np.pi/L) ; del mesh_comov
    save_Pk_in_right_way(Pknbk_comob,False,name_of_the_sim+'.data',L)

    #redshift PK  
    if RSD:
        Pknbk_red = FFTPower(mesh_redshift, mode='2d', poles = (0,2,4), dk=0, kmin=2*np.pi/L) ; del mesh_redshift
        save_Pk_in_right_way(Pknbk_red,True,name_of_the_sim+'.data',L)
    return

def apply_RSD(xc,yc,zc,vx,vy,vz,z_snap,L,PPA=True):
    '''
    correct the coordinate of particles given peculiar velocities (Doppler effect) in plane parallel approximation or not
    '''
    if PPA: vp = vz
    
    else:
        rc = np.sqrt(xc**2+yc**2+zc**2)
        vp = vx*xc/rc + vy*yc/rc + vz*zc/rc
    
    factor = vp * (1+z_snap)/(100*Eofz(z_snap)*np.sqrt(1+z_snap))
    
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
