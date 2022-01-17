import numpy as np
from itertools import product as iterProd
from time import time
from sys import stdout
import warnings
from numba import njit, prange


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from tools._numba_functions import *
from tools._shell_averaging import *
from tools.fast_interp.fast_interp.fast_interp import interp1d

def aliasing(density_field,Par,k_1D,k_3D):
    '''
    computes the 3D, aliased version (following an aliasing_order parameter) of a 1D power spectrum
    '''
    aliasOrd = Par['aliasing_order'] ; L = Par['L'] ; Ns = Par['N_sample'] ; a = L/Ns ; k_N = np.pi/a ; k_F = 2*np.pi/L
    
    if (not Par['Pk_dd_file'][-4:] == '.npy'):
        
        with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logk_1D_filt       = np.log(density_field['Pk_1D_dd_filtered'][0])
                logPk_theo_1D_filt = np.log(density_field['Pk_1D_dd_filtered'][1])

        logk_1D_filt       = logk_1D_filt      [np.isinf(logPk_theo_1D_filt) == False]
        logPk_theo_1D_filt = logPk_theo_1D_filt[np.isinf(logPk_theo_1D_filt) == False]
            
        interpolater = interp1d(a=logk_1D_filt[0],b=logk_1D_filt[-1],h=logk_1D_filt[1]-logk_1D_filt[0],f=logPk_theo_1D_filt,k=1,e=100)
            
        if aliasOrd == 0: 
            if Par['verbose'] and rank == 0: print('Since aliasing_order = 0, the target power spectrum is simply interpolated on 3D Fourier modes',flush=True)
            if rank == 0:
                with warnings.catch_warnings():
                    log_k3D = np.log(k_3D)
                density_field['Pk_3D_dd_alias'] = expo(interpolater(log_k3D))
            else:
                density_field['Pk_3D_dd_alias'] = 0
        else:
            
            ref     = np.arange(Ns)
            norm_1d = np.concatenate((ref[ref<Ns/2] *k_F,(ref[ref >= Ns/2] - Ns)*k_F))
            kz      = np.array([[norm_1d,]*Ns]*Ns)
            kx      = kz.transpose(2,1,0)
            ky      = kz.transpose(0,2,1)
            
            #computing the n1,n2,n3 arrays given aliasing_order
            n1_arr = [] ; n2_arr = [] ; n3_arr = []
            
            m = np.arange(2*aliasOrd+1) - aliasOrd
            
            for n1 in m:
                for n2 in m:
                    for n3 in m:
                        n1_arr.append(n1) ; n2_arr.append(n2) ; n3_arr.append(n3)
            n1_arr = np.array(n1_arr) ; n2_arr = np.array(n2_arr) ; n3_arr = np.array(n3_arr)
            
            ns1_to_distribute,ns2_to_distribute,ns3_to_distribute = spliting(n1_arr,n2_arr,n3_arr,size)
            my_ns1 = ns1_to_distribute[rank] ; my_ns2 = ns2_to_distribute[rank] ; my_ns3 = ns3_to_distribute[rank]
            
            if Par['verbose'] and rank == 0: print('start aliasing the theoretical power spectrum' + (size != 1)* ' in MPI',flush=True)
            
            Pk_3d_part    = np.zeros((Ns,Ns,Ns))    
            
            start_time  = time()
            total_loops = (2*aliasOrd+1)**3 / size
            
            @njit(parallel=True,cache=True)
            def lognorm_kalias(kx,ky,kz,m1,m2,m3):
                array_ = np.zeros_like(kx)
                for i in prange(array_.shape[0]):
                    for j in prange(array_.shape[1]):
                        for k in prange(array_.shape[2]):
                                array_[i,j,k] = np.log(np.sqrt((kx[i,j,k]-2.*m1*k_N)**2 + (ky[i,j,k]-2.*m2*k_N)**2 + (kz[i,j,k]-2.*m3*k_N)**2))
                return array_
            
            @njit(parallel=True,cache=True)
            def addexp(array1,array2):
                array_ = np.zeros_like(array1)
                for i in prange(array_.shape[0]):
                    for j in prange(array_.shape[1]):
                        for k in prange(array_.shape[2]):
                                array_[i,j,k] = array1[i,j,k] + np.exp(array2[i,j,k])
                return array_

            i=1
            for n in range(len(my_ns1)):
                n1 = my_ns1[n] ; n2 = my_ns2[n] ; n3 = my_ns3[n]
                logk_alias  = lognorm_kalias(kx,ky,kz,n1,n2,n3)
                Pk_k_alias  = interpolater(logk_alias) ; del logk_alias
                Pk_3d_part  = addexp(Pk_3d_part,Pk_k_alias) ; del Pk_k_alias
                
                time_since_start  = (time() - start_time)
                rmn = (time_since_start * total_loops/ i - time_since_start)/60
                percent = 100*i/total_loops
                if Par['verbose'] and rank == 0: 
                    stdout.write("\restimated remaining time: %.1f minutes, %.0f %%" %(rmn,percent)) ; stdout.flush()
                i+=1
            
            if Par['verbose'] and rank == 0: print('\n')
            del kx,ky,kz
            
            if rank == 0:  totals = np.zeros_like(Pk_3d_part)
            else:          totals = None
            
            comm.Barrier()
            comm.Reduce( [Pk_3d_part, MPI.DOUBLE], [totals, MPI.DOUBLE], op = MPI.SUM,root = 0)
            
            if rank == 0 :
                assert len(np.where(np.isnan(totals))[0])==0, 'aliasing failed, nan detected in the resulting 3D power spectrum'
                density_field['Pk_3D_dd_alias'] = totals
            else: density_field['Pk_3D_dd_alias'] = 0
                
        if rank == 0 and Par['debug']: 
            pk_aliased_1d = fast_shell_averaging(Par,density_field['Pk_3D_dd_alias'])
            np.savetxt(Par['output_dir_project'] + '/debug_files/k_pk_theo_aliased_1d.txt',np.transpose(np.vstack((k_1D,pk_aliased_1d))))        
    comm.Barrier()
    return density_field