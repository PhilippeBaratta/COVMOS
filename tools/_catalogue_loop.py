import numpy as np
from os import path as osp
from glob import glob
from time import sleep
from shutil import rmtree
from numba import njit, prange
import warnings
from sys import stdout
import time

warnings.catch_warnings()
warnings.simplefilter("ignore")

from mpi4py import MPI
comm = MPI.COMM_WORLD

from tools._networking import *
intracomm = from_globalcomm_to_intranode_comm()

import pyfftw as FFTw
from multiprocessing import cpu_count
FFTw.interfaces.cache.enable()
FFTw.interfaces.cache.set_keepalive_time(3600)
FFTw.config.NUM_THREADS = cpu_count()

from tools._velocity_model import *
from tools._save_analyse_catalogues import *
from tools._numba_functions import *
from tools._assignment_schemes import *
from tools._COVMOS_covariance import *
from tools.fast_interp.fast_interp.fast_interp import interp1d


def generate_analyse_catalogues(Par,Ary):
    if not Par['PDF_d_file'] == 'gaussian':
        x_nu    = Ary['x_nu']   [Ary['x_nu']<20]
        L_of_nu = Ary['L_of_nu'][Ary['x_nu']<20]
        mapping = interp1d(a=np.amin(x_nu),b=np.amax(x_nu),h=x_nu[1]-x_nu[0],f=L_of_nu,k=1)
    
    number_in_folder = len(glob(osp.join(Par['folder_job'], '*')))
    comm.Barrier()
    if number_in_folder < Par['total_number_of_cat'] : sleep(intracomm.Get_rank()*20)
    
    if Par['verbose'] and rank == 0: print('\n_____________________________ START CATALOGUES SIMULATION ____________________________\n',flush=True)
    while number_in_folder < Par['total_number_of_cat']:
        if Par['verbose'] and rank == 0:  stdout.write("\rloop on catalogues: %i / %i, %i%%" %(number_in_folder,Par['total_number_of_cat'],(number_in_folder/Par['total_number_of_cat'])*100)) ; stdout.flush()
            
        sim_ref = find_a_sim_number_to_run(Par)
        if not osp.exists(sim_ref['job_name']):
            
            np.savetxt(sim_ref['job_name'],[1])
            if Par['fixed_Rdm_seed'] : np.random.seed(0)
            else                     : np.random.seed()

            v_grid = np.array([],dtype=np.float32) ; rho_itp = np.array([],dtype=np.float64) ; v_itp = np.array([],dtype=np.float32)

            real1           = FFTw.empty_aligned(Par['grid_shape'], dtype='float64')   
            real1   [:,:,:] = np.random.normal(loc=0.,scale=1.,size=Par['grid_shape'])
            complex1        = FFTw.empty_aligned(Par['grid_shape'], dtype='complex128')
            complex1[:,:,:] = FFTw.interfaces.numpy_fft.fftn(real1,axes=(0,1,2))                          ; del real1

            if Par['velocity']:
                complex2        = FFTw.empty_aligned(Par['grid_shape'], dtype='complex64')
                complex2[:,:,:] = inverse_div_theta(complex1,Ary['byprod_pk_velocity'])
                v_grid          = FFTw.empty_aligned((3,Par['N_sample'],Par['N_sample'],Par['N_sample']), dtype='float32')
                
                v_grid[0,:,:,:] = FFTw.interfaces.numpy_fft.ifftn(array_times_array(complex2,Ary['kz'].transpose(2,1,0)),axes=(0,1,2)) 
                v_grid[1,:,:,:] = FFTw.interfaces.numpy_fft.ifftn(array_times_array(complex2,Ary['kz'].transpose(0,2,1)),axes=(0,1,2))
                v_grid[2,:,:,:] = FFTw.interfaces.numpy_fft.ifftn(array_times_array(complex2,Ary['kz'])                 ,axes=(0,1,2))
                del complex2

            complex1[:,:,:] = array_times_array(complex1,Ary['byprod_pk_density'])
            real1           = FFTw.empty_aligned(Par['grid_shape'], dtype='float64')
            real1[:,:,:]    = FFTw.interfaces.numpy_fft.ifftn(complex1,axes=(0,1,2))                      ; del complex1  
            if not Par['PDF_d_file'] == 'gaussian':
                real1[:,:,:] = mapping(real1)
                
            if Par['assign_scheme'] == 'tophat':  
                rho = from_delta_to_rho_times_a3(real1,Par['rho_0']*Par['a']**3)                          ; del real1
                Nbr = np.random.poisson(rho).flatten()
                rho = np.array([],dtype=np.float64) 

            if Par['assign_scheme'] == 'trilinear':
                rho,rho_mean_times_a3 = mean_delta_times_a3(real1,Par['a']**3,Par['rho_0'])               ; del real1
                Nbr = np.random.poisson(rho_mean_times_a3).flatten()                                      ; del rho_mean_times_a3

            non_0_  = np.where(Nbr!=0)[0] ; tot_obj = np.sum(Nbr) ; cumu = np.insert(np.cumsum(Nbr),0,0)
            rho_itp = np.zeros(tot_obj,dtype='float64')
            cat = np.zeros((3,tot_obj),dtype='float32')
            v_itp = np.zeros((3,tot_obj),dtype='float32')

            rdm_spl = np.random.random((3,tot_obj))
            cat,rho_itp,v_itp = discrete_assignment(cat,rho_itp,v_itp,non_0_,Nbr,cumu,rdm_spl,rho,v_grid,Par,Ary) ; del rdm_spl,v_grid,cumu,Nbr,rho,non_0_
            v_cat = apply_velocity_model(Par,rho_itp,v_itp)                                               ; del rho_itp,v_itp
            save_and_or_analyse_cat(Par,sim_ref,tot_obj,cat,v_cat)                                        ; del cat,v_cat

        number_in_folder = len(glob(osp.join(Par['folder_job'], '*')))
    comm.Barrier()
    if Par['verbose'] and rank == 0:  stdout.write("\rloop on catalogues: %i / %i, %i%%" %(number_in_folder,Par['total_number_of_cat'],(number_in_folder/Par['total_number_of_cat'])*100)) ; stdout.flush()

    if rank == 0 :
        if Par['verbose']:  
            print('\n',flush=True)
            if Par['compute_covariance']: print('waiting for the power spectra...',flush=True)
        if not Par['velocity']:
            while not len(glob(osp.join(Par['folder_Pk']+'*'))) == Par['total_number_of_cat']: sleep(10)
        else:
            while not len(glob(osp.join(Par['folder_Pk_RSD']+'*'))) == Par['total_number_of_cat']: sleep(10)
        if Par['verbose'] and Par['compute_covariance']: print('computing the unbiased covariance in',Par['folder_cov'],flush=True)
            
        compute_COVMOS_covariance(Par)
        

def find_a_sim_number_to_run(Par):
    np.random.seed()
    sim_ref = {}
    ramdom_sim = int(np.random.random()*Par['total_number_of_cat']+1)
    sim_ref['sim_name'] = Par['file_sim']+'_%i'%ramdom_sim
    sim_ref['job_name'] = Par['folder_job']+'_%i'%ramdom_sim
    sim_ref['number']   = '_%i'%ramdom_sim
    return sim_ref    
