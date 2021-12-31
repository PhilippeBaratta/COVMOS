import numpy as np
import warnings
from glob import glob
from os import makedirs
import scipy.special as SS

import pyfftw
from multiprocessing import cpu_count
pyfftw.config.NUM_THREADS = cpu_count()

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from tools._shell_averaging import *
from tools._numba_functions import *

intracomm = from_globalcomm_to_intranode_comm()
intrarank = intracomm.rank

def compute_2PS_predictions(Par,density_field,PDF_map,k_1D):
    Ns = Par['N_sample'] ; L = Par['L'] ; k_F = 2.*np.pi/L ; a = L/Ns
    
    if rank == 0 and (Par['compute_Pk_prediction'] or Par['compute_2pcf_prediction']):
        
        if Par['verbose']: print('\n__________________________ COMPUTING TWO-P STAT PREDICTIONS __________________________\n',flush=True)
        
        xi_nu     = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')
        Pk_target = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')
        
        if Par['verbose']: print('compute the 3D grid, output power spectrum',flush=True)
            
        if not Par['PDF_d_file'] == 'gaussian':    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xi_nu[:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(density_field['Pk_nu'],axes=(0,1,2))*(Ns*k_F)**3
                xi_delta     = np.interp(xi_nu,PDF_map['Xi_G_template'],PDF_map['Xi_NG_template'])          ; del xi_nu
                Pk_target[:,:,:] = pyfftw.interfaces.numpy_fft.fftn(xi_delta,axes=(0,1,2))/(Ns*k_F)**3
        else: 
            Pk_target[:,:,:] = density_field['Pk_nu']                                                      ; del xi_nu
            xi_delta         = np.real(pyfftw.interfaces.numpy_fft.ifftn(Pk_target,axes=(0,1,2))*(Ns*k_F)**3)

        if not Par['compute_2pcf_prediction']: del xi_delta

        if Par['compute_Pk_prediction']: Pk_target_grid_1d = fast_shell_averaging(Par,Pk_target)
        
        ref     = np.arange(Ns)
        norm_1d = np.concatenate((ref[ref<Ns/2] *k_F,(ref[ref >= Ns/2] - Ns)*k_F))
        kz      = np.array([[norm_1d,]*Ns]*Ns)
                    
        if Par['verbose']: print('computing W0, the window function related to the',Par['assign_scheme'],'assignment scheme',flush=True)
            
        kx_ademi = array_times_scalar(kz.transpose(2,1,0),a/2)
        ky_ademi = array_times_scalar(kz.transpose(0,2,1),a/2)
        kz_ademi = array_times_scalar(kz,a/2)                                                        ; del kz
        
        W0 = array_times_array_times_array(SS.spherical_jn(0,kx_ademi),SS.spherical_jn(0,ky_ademi),SS.spherical_jn(0,kz_ademi))
        del kx_ademi, ky_ademi, kz_ademi

        if Par['assign_scheme'] == 'trilinear': W0 = array_times_array(W0,W0)
        
        if Par['verbose']: print('applying it to the grid power spectrum to obtain the 3D catalogue power spectrum',flush=True)
            
        Pkpoisson3D = array_times_arraypow2(Pk_target,abs(W0))                                       ; del W0,Pk_target
        
        if Par['compute_Pk_prediction']: 
            if Par['verbose']: print('shell-average it',flush=True)
            Pkpoisson1D = fast_shell_averaging(Par,Pkpoisson3D)
            if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/grid_k_Pk_prediction.txt')
            np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/grid_k_Pk_prediction.txt',np.transpose(np.vstack((k_1D,Pk_target_grid_1d))))
            if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/catalogue_k_Pk_prediction.txt')
            np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/catalogue_k_Pk_prediction.txt',np.transpose(np.vstack((k_1D,Pkpoisson1D))))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Pk_target = np.exp(np.interp(np.log(k_1D),np.log(density_field['Pk_1D_dd'][0]),np.log(density_field['Pk_1D_dd'][1]),right=1e-10))
            if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/target_k_Pk.txt')
            np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/target_k_Pk.txt',np.transpose(np.vstack((k_1D,Pk_target))))
    
    if Par['compute_2pcf_prediction']:
        if Par['verbose'] and rank == 0:  print('\nnow working on two-point correlation functions',flush=True)
        
        if rank == 0: 
            twopcfP        = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                twopcfP[:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(Pkpoisson3D,axes=(0,1,2))*(Ns*k_F)**3 ; del Pkpoisson3D
                
        shell_avg_file_2pcf = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'shellaveraging_trick_arrays_2PCF_L%s_Ns%i'%(L,Ns)
        
        size_small_scales = len(glob(shell_avg_file_2pcf+'/smallscales*')) 
        size_large_scales = len(glob(shell_avg_file_2pcf+'/largescales*'))
        width_shell = 2 # in Mpc/h
        
        if (size_small_scales == 0 or size_large_scales == 0):
            normpos = shellaveraging_trick_funtion_2pcf(Par,width_shell)
            
        else: 
            if rank == 0:
                zzz    = np.zeros((Ns,Ns,Ns))
                zzz[:] = np.linspace(0,L,Ns)
                normpos = fast_norm(np.transpose(zzz,(2,1,0)),np.transpose(zzz,(1,2,0)),zzz)    ; del zzz

        if rank == 0 :
            SA_SmallScales = [] ; SA_LargeScales = []

            r_to_share = np.arange(100)
            centre_shells_tot  = np.arange(0,int(L))+width_shell/2

            for r in r_to_share:
                file = np.load(shell_avg_file_2pcf +'/smallscales'+ str(r)+'.npy')
                SA_SmallScales += [file]
            for r in centre_shells_tot:
                file = np.load(shell_avg_file_2pcf+'/largescales'+ str(r)+'.npy')
                SA_LargeScales += [file]
                        
            R_P,corr_P = twoPCF_Shell_Averaging(SA_SmallScales,SA_LargeScales,normpos,twopcfP,Par)
            R_G,corr_G = twoPCF_Shell_Averaging(SA_SmallScales,SA_LargeScales,normpos,xi_delta,Par)

            if Par['compute_2pcf_prediction']:
                if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/grid_r_Xi_prediction.txt',flush=True)
                np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/grid_r_Xi_prediction.txt',np.transpose(np.vstack((R_G,corr_G))))
                if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/catalogue_r_Xi_prediction.txt',flush=True)
                np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/catalogue_r_Xi_prediction.txt',np.transpose(np.vstack((R_P,corr_P))))

    return