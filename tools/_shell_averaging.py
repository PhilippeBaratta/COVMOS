import numpy as np
from glob import glob
from os import makedirs

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from tools._networking import *
from tools._numba_functions import *

intracomm = from_globalcomm_to_intranode_comm()
intrarank = intracomm.rank


def shellaveraging_trick_function(Par,k_3D): 
    '''
    smart way of storing position of each shell. It allows to perform a faster shell-averaging procedure
    '''
    Ns = Par['N_sample'] ; L  = Par['L'] ; k_F = 2*np.pi/L ; SA_trick = []
    
    shell_avg_file = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'shellaveraging_trick_arrays_Pk_L%s_Ns%i'%(L,Ns)
    number_of_files = len(glob(shell_avg_file+'/*')) 
        
    if number_of_files == 0:
        if Par['verbose'] and rank == 0: print('computing the shell-averaging trick arrays ' + (size != 1)* 'in MPI',flush=True)
        
        k_to_share = np.arange(1,int(Ns/2)) #for each shell up to k_N-1, without the DC mode
        k_ = spliting_one_array(k_to_share,size)[rank]            
        
        if rank == 0: makedirs(shell_avg_file,exist_ok=True)
        
        comm.Barrier()
        for k in k_:    
            lower_bound = k*k_F-k_F/2.
            upper_bound = k*k_F+k_F/2.
            index = np.where(((k_3D > lower_bound) * (k_3D < upper_bound)))
            np.save(shell_avg_file +'/'+ str(k) , index)
        
        comm.Barrier()
        if Par['verbose'] and rank == 0 : print('shell-averaging trick arrays has been saved in ' + shell_avg_file ,flush=True)
        
    if rank == 0 :
        SA_trick = []
        k_to_share = np.arange(1,int(Ns/2))
        
        for k in k_to_share:
            file = np.load(shell_avg_file +'/'+ str(k)+'.npy')
            SA_trick += [file]            
        
    return SA_trick

def shell_averaging(SA_trick,table3d):
    '''
    performs the shell-average of a given 3D table in Fourier space
    '''
    table_1D = np.zeros(int(table3d.shape[0]/2-1))
    for k in range(int(table3d.shape[0]/2-1)):
        table_1D[k] = np.average(table3d[SA_trick[k][0],SA_trick[k][1],SA_trick[k][2]])
    return table_1D

def fast_shell_averaging(Par,Pk3D):
    '''
    performs the shell-average of a given 3D table in Fourier space
    '''
    SA_trick = shellaveraging_trick_function(Par,None)
    pk1d  = shell_averaging(SA_trick,Pk3D)
    return pk1d

def shellaveraging_trick_funtion_2pcf(Par,width_shell):
    Ns = Par['N_sample'] ; L  = Par['L']
    
    shell_avg_file_2pcf = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'shellaveraging_trick_arrays_2PCF_L%s_Ns%i'%(L,Ns)
    if Par['verbose'] and rank == 0: print('computing shell averaging files for 2pcf prediction' + (size != 1)* ' in MPI',flush=True)
            
    normpos = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float64')
    unique  = sharing_array_throw_MPI((100,)    ,intracomm,'float64')
        
    if intrarank == 0 or rank == 0:
        zzz    = np.zeros((Ns,Ns,Ns))
        zzz[:] = np.linspace(0,L,Ns)
        normpos[:,:,:] = fast_norm(np.transpose(zzz,(2,1,0)),np.transpose(zzz,(1,2,0)),zzz)    ; del zzz
        unique[:]      = np.unique(normpos)[:100]
            
    #working on small scales____________________________________________
    r_to_share = np.arange(100)
    r_ = spliting_one_array(r_to_share,size)[rank]  
            
    if rank == 0: makedirs(shell_avg_file_2pcf,exist_ok=True)
            
    comm.Barrier()
    #finding and stocking the array indices corresponding to the first 100 values in normpos (only for small scales)
    for r in r_:
        index = np.where(normpos == unique[r])
        np.save(shell_avg_file_2pcf+'/smallscales'+ str(r) , index)
            
    #working on large scales____________________________________________
    #finding and stocking the array indices corresponding to shells of width ... (for larger scales)
    centre_shells_tot  = np.arange(0,int(L))+width_shell/2
    centre_shells_part = spliting_one_array(centre_shells_tot,size)[rank]

    for centre in centre_shells_part:
        index = np.where(((centre-width_shell/2)<normpos) * ((centre+width_shell/2)>normpos))
        np.save(shell_avg_file_2pcf+'/largescales'+ str(centre),index)
            
    if Par['verbose'] and rank == 0 : print('shell-averaging trick arrays for 2pcf have been saved in ' + shell_avg_file_2pcf,flush=True)
            
    comm.Barrier()
    return normpos
    

def twoPCF_Shell_Averaging(SA_SmallScales,SA_LargeScales,normpos,twoPCF_3D,Par):
    '''
    computes the shell-averaging of a 3D 2pcf
    '''
    #shell averaging the 2pcf using the two methods (small scales and large scales)
    width_shell = 2 # in Mpc/h
    LS_len = len(np.arange(0,int(Par['L']))+width_shell/2)
    
    SAdist   = np.zeros(100)
    SAdist2  = np.zeros(LS_len)
    SA2pcfP  = np.zeros(100)
    SA2pcf2P = np.zeros(LS_len)
    
    for i in range(100):
        SAdist[i]  = np.mean(normpos[SA_SmallScales[i][0],SA_SmallScales[i][1],SA_SmallScales[i][2]])
        SA2pcfP[i] = np.mean(twoPCF_3D[SA_SmallScales[i][0],SA_SmallScales[i][1],SA_SmallScales[i][2]])
    for i in range(LS_len):
        SAdist2[i] = np.mean(normpos[SA_LargeScales[i][0],SA_LargeScales[i][1],SA_LargeScales[i][2]])
        SA2pcf2P[i]= np.mean(twoPCF_3D[SA_LargeScales[i][0],SA_LargeScales[i][1],SA_LargeScales[i][2]])

    #For small R, one keeps the exact values, then at larger scales the averaged 2pcf in shells     
    R    = np.append(SAdist ,SAdist2 [SAdist2>SAdist[-1]])
    corr = np.append(SA2pcfP,SA2pcf2P[SAdist2>SAdist[-1]])
    return R,corr