import numpy as np
from itertools import product as iterProd
from time import time
from sys import stdout
import warnings

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from tools._numba_functions import *
from tools._shell_averaging import *

def aliasing(density_field,Par,k_1D,k_3D):
    '''
    computes the 3D, aliased version (following an aliasing_order parameter) of a 1D power spectrum
    '''
    aliasOrd = Par['aliasing_order'] ; L = Par['L'] ; Ns = Par['N_sample'] ; a = L/Ns ; k_N = np.pi/a ; k_F = 2*np.pi/L
    
    if (not Par['Pk_dd_file'][-4:] == '.npy'):
        if aliasOrd == 0: 
            if Par['verbose'] and rank == 0: print('Since aliasing_order = 0, the target power spectrum is simply interpolated on 3D Fourier modes',flush=True)
            density_field['Pk_3D_dd_alias'] = np.exp(np.interp(np.log(k_3D),np.log(density_field['Pk_1D_dd_filtered'][0]),np.log(density_field['Pk_1D_dd_filtered'][1]),right=1e-10))
        
        else:
            '''kx = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float32')
            ky = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float32')
            kz = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float32')
            if intrarank == 0: 
                kz[:,:,:] = Fouriermodes(Par,mode=3)
                kx[:,:,:] = kz.transpose(2,1,0)
                ky[:,:,:] = kz.transpose(0,2,1)''' #this array sharing is not compatible with MPI.SUM
            
            ref     = np.arange(Ns)
            norm_1d = np.concatenate((ref[ref<Ns/2] *k_F,(ref[ref >= Ns/2] - Ns)*k_F))
            kz      = np.array([[norm_1d,]*Ns]*Ns)
            kx      = kz.transpose(2,1,0)
            ky      = kz.transpose(0,2,1)
            
            #computing the n1,n2,n3 arrays given aliasing_order
            n1_arr = [] ; n2_arr = [] ; n3_arr = []
            for n1,n2,n3 in iterProd(range(2*aliasOrd+1),range(2*aliasOrd+1),range(2*aliasOrd+1)):
                n1_arr.append(n1) ; n2_arr.append(n2) ; n3_arr.append(n3)
            n1_arr = np.array(n1_arr) ; n2_arr = np.array(n2_arr) ; n3_arr = np.array(n3_arr)
            
            ns1_to_distribute,ns2_to_distribute,ns3_to_distribute = spliting(n1_arr,n2_arr,n3_arr,size)
            my_ns1 = ns1_to_distribute[rank] ; my_ns2 = ns2_to_distribute[rank] ; my_ns3 = ns3_to_distribute[rank]
            
            if Par['verbose'] and rank == 0: print('start aliasing the theoretical power spectrum' + (size != 1)* ' in MPI',flush=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logk_1D       = np.log(density_field['Pk_1D_dd_filtered'][0])
                logPk_theo_1D = np.log(density_field['Pk_1D_dd_filtered'][1])

            Pk_3d_part    = np.zeros((Ns,Ns,Ns))    
            logk_alias    = np.zeros_like(Pk_3d_part)
            
            start_time  = time()
            total_loops = (2*aliasOrd+1)**3 / size

            i=1
            for n in range(len(my_ns1)):
                n1 = my_ns1[n] ; n2 = my_ns2[n] ; n3 = my_ns3[n]
                logk_alias[:,:,:] = log_kalias(n1,n2,n3,k_N,aliasOrd,kx,ky,kz)
                logk_alias[:,:,:] = np.interp(logk_alias,logk_1D,logPk_theo_1D)
                Pk_3d_part[:,:,:] = exppara(Pk_3d_part,logk_alias)

                time_since_start  = (time() - start_time)
                rmn = (time_since_start * total_loops/ i - time_since_start)/60
                percent = 100*i/total_loops
                if Par['verbose'] and rank == 0: 
                    stdout.write("\restimated remaining time: %.1f minutes, %.0f %%" %(rmn,percent)) ; stdout.flush()
                i+=1
            
            if Par['verbose'] and rank == 0: print('\n')
            del kx,ky,kz,logk_alias
            
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