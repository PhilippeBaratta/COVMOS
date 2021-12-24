'''
This code can be run if and only if the code COVMOS_ini.py has already been executed, associated to a .ini file.
You can run it in mpi (or not) using 'mpiexec -f machinefile -n 10 python /renoir/baratta/COVMOS_public/COVMOS/COVMOS_sim.py setting_example'
This code is divided into five parts: 
- NETWORK INITIALISATION identifies the properties of the MPI network
- PARAMETER INITIALISATION reads the input parameters set by the user, defines some constants and creates output repertories 
- LOADING/SHARING ARRAYS loads the files computed by COVMOS_ini.py and some relevant arrays for the pipeline. These arrays are shared through processes belonging to the same machine
- COORD ASSIGNMENT DEFINITIONS is the parallelised function allowing the coordinate assignment of particles
- MAIN LOOP ON CATALOGUES stands for the main process: it generates first the Gaussian field, transforms it to obtain delta, Poissonizes it, assignes coordinates, generates the velocity field, assignes it to particles, saves and/or run NBodyKit to estimate the Pk multipoles
'''


#####################################################################################################################
########################################## NETWORK INITIALISATION ###################################################
#####################################################################################################################
from mpi4py import MPI
from multiprocessing import cpu_count
import pyfftw
from COVMOS_func import *

comm = MPI.COMM_WORLD
intracomm = from_globalcomm_to_intranode_comm()
pyfftw.interfaces.cache.enable()
pyfftw.interfaces.cache.set_keepalive_time(3600)
pyfftw.config.NUM_THREADS = cpu_count()

#####################################################################################################################
########################################## PARAMETER INITIALISATION #################################################
#####################################################################################################################
from initialization_funcs import *
import sys

Par = read_parameters(str(sys.argv[1]))

if estimate_Pk_multipoles == 'stopandrun' : 
    detached_Pk_estimate = False ; savePkcat = True
elif estimate_Pk_multipoles == 'detached'   : 
    detached_Pk_estimate = True  ; savePkcat = True
else : savePkcat = False

a    = L/N_sample
unit = -100 * Eofz(redshift,Omega_m)/(1+redshift)
folder_saving = output_dir + (not output_dir[-1]=='/')*'/' + project_name + '/outputs/' 
generate_and_clean_repertories_if_necessary(folder_saving,total_number_of_cat,project_name,save_catalogue,velocity,savePkcat,comm.Get_rank())

#####################################################################################################################
############################################ LOADING/SHARING ARRAYS #################################################
#####################################################################################################################
import numpy as np

file = np.load(output_dir + (not output_dir[-1]=='/')*'/' + project_name + '/inputs/Byprod_Gaussian_spectra.npz')
x_nu = file['arr_0']  ; L_of_nu = file['arr_1']

grid_shape = (N_sample,N_sample,N_sample)

byprod_pk_density  = sharing_array_throw_MPI(grid_shape,intracomm,'float64')
if intracomm.rank == 0: byprod_pk_density[:,:,:] = file['arr_2']
        
if velocity: 
    byprod_pk_velocity = sharing_array_throw_MPI(grid_shape,intracomm,'float32')
    kz                 = sharing_array_throw_MPI(grid_shape,intracomm,'float32')
    k_3D_2             = sharing_array_throw_MPI(grid_shape,intracomm,'float32')
    
    if intracomm.rank == 0:
        kz[:,:,:],k_3D_2[:,:,:] = Fouriermodes(L,N_sample,mode=2)
        byprod_pk_velocity[:,:,:]   = (file['arr_3']).astype(np.float32)
    
grid_pos = sharing_array_throw_MPI((3,N_sample**3),intracomm,'float32')
if intracomm.rank == 0: grid_pos[:,:] = grid_positions(N_sample,L)
if assign_sheme == 'trilinear': 
    vertex = sharing_array_throw_MPI((3,N_sample**3),intracomm,'int16')
    if intracomm.rank == 0: vertex[:,:] = compute_vertex_indices(N_sample)
else: vertex = np.array([],dtype=np.int16)
    
    

#####################################################################################################################
########################################## COORD ASSIGNMENT DEFINITIONS #############################################
#####################################################################################################################
from numba import njit, prange

@njit(parallel=True,cache=True)
def discrete_assignment(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,rho,v_grid):
    if assign_sheme == 'tophat':
        for i in prange(len(non_0_)):
            non_0         = non_0_[i]
            N_cell        = Nbr [non_0]
            cumu_non_0    = cumu[non_0]
            cumu_non_0_p1 = cumu[non_0+1]
            les_randoms_  = les_randoms[:,cumu_non_0:cumu_non_0_p1]
            grille        = grid_pos[:,non_0]
            cat[0,cumu_non_0:cumu_non_0_p1] = les_randoms_[0]*a + grille[0]
            cat[1,cumu_non_0:cumu_non_0_p1] = les_randoms_[1]*a + grille[1]
            cat[2,cumu_non_0:cumu_non_0_p1] = les_randoms_[2]*a + grille[2]

    elif assign_sheme == 'trilinear':
        for i in prange(len(non_0_)):
            non_0         = non_0_[i]
            N_cell        = Nbr [non_0]
            cumu_non_0    = cumu[non_0]
            cumu_non_0_p1 = cumu[non_0+1]
            les_randoms_  = les_randoms[:,cumu_non_0:cumu_non_0_p1]
            grille        = grid_pos[:,non_0]
            
            vtx_0 = vertex[0,non_0]                     ; vtx_1 = vertex[1,non_0]                     ; vtx_2 = vertex[2,non_0]
            vtx_3 = (vtx_0 != (N_sample-1)) * (vtx_0+1) ; vtx_4 = (vtx_1 != (N_sample-1)) * (vtx_1+1) ; vtx_5 = (vtx_2 != (N_sample-1)) * (vtx_2+1)
            
            rho_000 = rho[vtx_0,vtx_1,vtx_2] ; rho_a0a = rho[vtx_3,vtx_1,vtx_5] ; rho_0a0 = rho[vtx_0,vtx_4,vtx_2] ; rho_aa0 = rho[vtx_3,vtx_4,vtx_2]
            rho_a00 = rho[vtx_3,vtx_1,vtx_2] ; rho_0aa = rho[vtx_0,vtx_4,vtx_5] ; rho_00a = rho[vtx_0,vtx_1,vtx_5] ; rho_aaa = rho[vtx_3,vtx_4,vtx_5]
            
            C1 = rho_000+rho_0a0+rho_00a+rho_0aa
            C2 = -rho_000+rho_a00-rho_0a0-rho_00a+rho_a0a-rho_0aa+rho_aa0+rho_aaa
            C3 = rho_000+rho_a00+rho_0a0+rho_00a+rho_a0a+rho_0aa+rho_aa0+rho_aaa
            x  = (a*(-C1+np.sqrt(C1**2+C2*C3*les_randoms_[0])))/C2
            
            C1 = a*(-rho_000-rho_00a+rho_0a0+rho_0aa)+x*(rho_000+rho_00a-rho_0a0-rho_0aa-rho_a00-rho_a0a+rho_aa0+rho_aaa)
            C2 = 2*a**2 *(rho_000+rho_00a)-2*a*x*(rho_000+rho_00a-rho_a00-rho_a0a)
            C3 = a**2*(a*(rho_000+rho_00a+rho_0a0+rho_0aa)+x*(-rho_000-rho_00a-rho_0a0-rho_0aa+rho_a00+rho_a0a+rho_aa0+rho_aaa))
            y  = -(C2-np.sign(C1)*np.sqrt(C2**2+4*C1*C3*les_randoms_[1])*np.sign(C1))/(2*C1)
            
            C1 = a*(a*(-rho_000+rho_00a)+x*(rho_000-rho_00a-rho_a00+rho_a0a)+y*(rho_000-rho_00a-rho_0a0+rho_0aa))+x*y*(-rho_000+rho_00a+rho_0a0-rho_0aa+rho_a00-rho_a0a-rho_aa0+rho_aaa)
            C2 = 2*a*(a*(a*rho_000+x*(-rho_000+rho_a00)+y*(-rho_000+rho_0a0))+x*y*(rho_000-rho_0a0-rho_a00+rho_aa0))
            C3 = a**2*(a**2*(rho_000+rho_00a)+a*(y*(-rho_000-rho_00a+rho_0a0+rho_0aa)+x*(-rho_000-rho_00a+rho_a00+rho_a0a))+x*y*(rho_000+rho_00a-rho_0a0-rho_0aa-rho_a00-rho_a0a+rho_aa0+rho_aaa))
            z  = -(C2-np.sign(C1)*np.sqrt(C2**2+4*C1*C3*les_randoms_[2])*np.sign(C1))/(2*C1)
            
            cat[0,cumu_non_0:cumu_non_0_p1] = x + grille[0]
            cat[1,cumu_non_0:cumu_non_0_p1] = y + grille[1]
            cat[2,cumu_non_0:cumu_non_0_p1] = z + grille[2]
            
            if velocity:
                vx_000 = v_grid[0,vtx_0,vtx_1,vtx_2] ; vy_000 = v_grid[1,vtx_0,vtx_1,vtx_2] ; vz_000 = v_grid[2,vtx_0,vtx_1,vtx_2] 
                vx_a00 = v_grid[0,vtx_3,vtx_1,vtx_2] ; vy_a00 = v_grid[1,vtx_3,vtx_1,vtx_2] ; vz_a00 = v_grid[2,vtx_3,vtx_1,vtx_2] 
                vx_0a0 = v_grid[0,vtx_0,vtx_4,vtx_2] ; vy_0a0 = v_grid[1,vtx_0,vtx_4,vtx_2] ; vz_0a0 = v_grid[2,vtx_0,vtx_4,vtx_2] 
                vx_00a = v_grid[0,vtx_0,vtx_1,vtx_5] ; vy_00a = v_grid[1,vtx_0,vtx_1,vtx_5] ; vz_00a = v_grid[2,vtx_0,vtx_1,vtx_5] 
                vx_a0a = v_grid[0,vtx_3,vtx_1,vtx_5] ; vy_a0a = v_grid[1,vtx_3,vtx_1,vtx_5] ; vz_a0a = v_grid[2,vtx_3,vtx_1,vtx_5] 
                vx_0aa = v_grid[0,vtx_0,vtx_4,vtx_5] ; vy_0aa = v_grid[1,vtx_0,vtx_4,vtx_5] ; vz_0aa = v_grid[2,vtx_0,vtx_4,vtx_5] 
                vx_aa0 = v_grid[0,vtx_3,vtx_4,vtx_2] ; vy_aa0 = v_grid[1,vtx_3,vtx_4,vtx_2] ; vz_aa0 = v_grid[2,vtx_3,vtx_4,vtx_2] 
                vx_aaa = v_grid[0,vtx_3,vtx_4,vtx_5] ; vy_aaa = v_grid[1,vtx_3,vtx_4,vtx_5] ; vz_aaa = v_grid[2,vtx_3,vtx_4,vtx_5] 
                
                amx = a-x ; amy = a-y ; amz = a-z ; a3 = a**3
                
                v_of_part_x = vx_000*amx*amy*amz+vx_a00*x*amy*amz+vx_0a0*amx*y*amz+vx_00a*amx*amy*z+vx_a0a*x*amy*z+vx_0aa*amx*y*z+vx_aa0*x*y*amz+vx_aaa*x*y*z
                v_of_part_y = vy_000*amx*amy*amz+vy_a00*x*amy*amz+vy_0a0*amx*y*amz+vy_00a*amx*amy*z+vy_a0a*x*amy*z+vy_0aa*amx*y*z+vy_aa0*x*y*amz+vy_aaa*x*y*z
                v_of_part_z = vz_000*amx*amy*amz+vz_a00*x*amy*amz+vz_0a0*amx*y*amz+vz_00a*amx*amy*z+vz_a0a*x*amy*z+vz_0aa*amx*y*z+vz_aa0*x*y*amz+vz_aaa*x*y*z
                rho_trip    = rho_000*amx*amy*amz+rho_a00*x*amy*amz+rho_0a0*amx*y*amz+rho_00a*amx*amy*z+rho_a0a*x*amy*z+rho_0aa*amx*y*z+rho_aa0*x*y*amz+rho_aaa*x*y*z
                
                v_itp[0,cumu_non_0:cumu_non_0_p1] = v_of_part_x/a3 * unit
                v_itp[1,cumu_non_0:cumu_non_0_p1] = v_of_part_y/a3 * unit
                v_itp[2,cumu_non_0:cumu_non_0_p1] = v_of_part_z/a3 * unit
                rho_itp[cumu_non_0:cumu_non_0_p1] = rho_trip   /a3
    
    return cat,rho_itp,v_itp


#####################################################################################################################
########################################### MAIN LOOP ON CATALOGUES #################################################
#####################################################################################################################
from os import path as osp
from glob import glob
from time import sleep
from shutil import rmtree

number_in_folder = len(glob(osp.join(folder_saving+'_job/', '*'))) ; comm.Barrier() ; sleep(comm.Get_rank())
sleep(intracomm.Get_rank()*60)

while number_in_folder < total_number_of_cat:
    name_of_the_sim,name_of_the_job = find_a_sim_number_to_run(total_number_of_cat,folder_saving,project_name)
    
    if not osp.exists(name_of_the_job):
        np.savetxt(name_of_the_job,[1])
        if fixed_Rdm_seed == False : np.random.seed()
        else                       : np.random.seed(0)
        
        v_grid = np.array([],dtype=np.float32) ; rho_itp = np.array([],dtype=np.float64) ; v_itp = np.array([],dtype=np.float32)
        
        real1           = pyfftw.empty_aligned(grid_shape, dtype='float64')   
        real1   [:,:,:] = np.random.normal(loc=0.,scale=1.,size=grid_shape)
        complex1        = pyfftw.empty_aligned(grid_shape, dtype='complex128')
        complex1[:,:,:] = pyfftw.interfaces.numpy_fft.fftn(real1,axes=(0,1,2)) ; del real1
        
        if velocity:
            complex2        = pyfftw.empty_aligned(grid_shape, dtype='complex64')
            complex2[:,:,:] = inverse_div_theta(complex1,byprod_pk_velocity,k_3D_2)
            v_grid          = pyfftw.empty_aligned((3,N_sample,N_sample,N_sample), dtype='float32')
            v_grid[0,:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(array_times_array(complex2,kz.transpose(2,1,0)),axes=(0,1,2)) 
            v_grid[1,:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(array_times_array(complex2,kz.transpose(0,2,1)),axes=(0,1,2))
            v_grid[2,:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(array_times_array(complex2,kz)                 ,axes=(0,1,2)) ; del complex2
            
        complex1[:,:,:] = array_times_array(complex1,byprod_pk_density)
        real1           = pyfftw.empty_aligned(grid_shape, dtype='float64')
        real1[:,:,:]    = pyfftw.interfaces.numpy_fft.ifftn(complex1,axes=(0,1,2)) ; del complex1  
        real1[:,:,:]    = np.interp(real1,x_nu,L_of_nu)
                
        if assign_sheme == 'tophat':  
            rho = from_delta_to_rho_times_a3(real1,rho_0,a**3) ; del real1
            Nbr = np.reshape(np.random.poisson(rho),N_sample**3)
            rho = np.array([],dtype=np.float64) 
        
        if assign_sheme == 'trilinear':
            rho,rho_mean_times_a3 = mean_delta_times_a3(real1,a**3,rho_0)      ; del real1
            Nbr = np.reshape(np.random.poisson(rho_mean_times_a3),N_sample**3) ; del rho_mean_times_a3
              
        non_0_  = np.where(Nbr!=0)[0] ; tot_obj = np.sum(Nbr) ; cumu = np.insert(np.cumsum(Nbr),0,0)
        rho_itp = np.zeros(tot_obj,dtype='float64') ; cat = np.zeros((3,tot_obj),dtype='float32') ; v_itp = np.zeros((3,tot_obj),dtype='float32')
                
        les_randoms = np.random.random((3,tot_obj))
        cat,rho_itp,v_itp = discrete_assignment(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,rho,v_grid) ; del les_randoms,v_grid,cumu,Nbr,rho,non_0_
        v_cat = apply_velocity_model(rho_itp,rho_0,targeted_rms,v_itp,alpha,velocity,comm,folder_saving) ; del rho_itp,v_itp
        save_and_or_analyse_cat(save_catalogue,redshift,name_of_the_sim,velocity,tot_obj,cat,v_cat,detached_Pk_estimate,L,savePkcat,comm) ; del cat,v_cat
    
    number_in_folder = len(glob(osp.join(folder_saving+'_job/', '*')))

comm.Barrier()

if comm.Get_rank() == 0 : 
    rmtree(folder_saving+'_job/')