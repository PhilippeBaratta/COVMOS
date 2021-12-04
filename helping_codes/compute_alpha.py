'''
This code helps the user to estimate on his own data the alpha parameter, required if velocity = True (see setting_example.ini). 
First the user has to write the function load_user_data_catalogue() (see below) used by this code to load positions and velocities of user data. Then the code assigns to cubical cells these particles. The variance of the peculiar velocity field is computed within each cell, as well as the local density. Finally approximating the obtained relation as sigma**2 = beta (delta +1)**alpha, a linear regression is performed in loglog scale in order to obtain alpha. 
Note that this code can be run in MPI if several input catalogues are provided by the user. A better estimate of the alpha parameter will be performed in this case.

inputs : - a function written by the user to load his data
         - L, the size of the cubical box
         - N_sample, the sampling parameter (L/N_sample (the grid precision) must be similar to the one set in the setting.ini file)

output : - alpha (printed)
'''

#####################################################################################################################
############################################# USER INITIALISATION ###################################################
#####################################################################################################################
from mpi4py import MPI

comm = MPI.COMM_WORLD ; size = comm.Get_size() ; rank = comm.Get_rank()

def load_user_data_catalogue(rank):
    '''
    This function must be written by the user in order to load his catalogue(s). If the code is run in mpi, several catalogues (only defined by different initial condition, but same parameters) can be treated in parallel. In this case, each MPI rank must return one catalogue.
    x,y,z,vx,vy,vz must all be provided in Mpc/h and x,y,z must all belong to [0,L]
    '''
    
    return x,y,z,vx,vy,vz

x,y,z,vx,vy,vz = load_user_data_catalogue(rank)

L         =  #in Mpc/h. Once again L/N_sample (the grid precision) must be similar to the one set in the setting.ini file
N_sample  = 

#####################################################################################################################
############################################# FUNCTIONS DEFINITION ##################################################
#####################################################################################################################
import numpy as np
from numba import njit, prange

@njit(parallel=True,cache=True)
def TOPHAT_MAS(x,y,z,L,N_sample):
    '''
    assigning particles on the 3D grid (TOPHAT WINDOW)
    '''
    for i in prange(len(x)):
        x[i] = (x[i]/L)*N_sample
        y[i] = (y[i]/L)*N_sample
        z[i] = (z[i]/L)*N_sample
    return x,y,z

@njit(parallel=True,cache=True)
def counting(x,y,z,N_sample):
    '''
    counting the number of particles falling in each 3D grid cell
    '''
    mycube_number = np.zeros((N_sample,N_sample,N_sample),dtype=np.int32)
    for i in prange(len(x)):
        mycube_number[x[i],y[i],z[i]] += 1
    return mycube_number

@njit(parallel=True,cache=True)
def stacking_for_variance(x,y,z,vi,N_sample):
    '''
    storing sum(vi) and sum(vi^2)
    '''
    mycube_sum_vi        = np.zeros((N_sample,N_sample,N_sample))
    mycube_sumsquared_vi = np.zeros((N_sample,N_sample,N_sample))
    for i in prange(len(x)):
        mycube_sum_vi[x[i],y[i],z[i]]        += vi[i]
        mycube_sumsquared_vi[x[i],y[i],z[i]] += vi[i]**2
    return mycube_sum_vi,mycube_sumsquared_vi

@njit(parallel=True,cache=True)
def unbiased_variance(N_sample,mycube_number,mycube_sumsquared_vi,mycube_sum_vi):
    '''
    preparing the computation of the variance (then computed by ns2ms2/n_nm1)
    '''
    n_nm1  = np.zeros((N_sample,N_sample,N_sample))
    ns2ms2 = np.zeros((N_sample,N_sample,N_sample))
    for i in prange(N_sample):
        for j in prange(N_sample):
            for k in prange(N_sample):
                n_nm1[i,j,k] = np.sqrt(mycube_number[i,j,k]*(mycube_number[i,j,k]-1))
                ns2ms2[i,j,k]= np.sqrt(mycube_number[i,j,k]*mycube_sumsquared_vi[i,j,k] - mycube_sum_vi[i,j,k]**2)
    return n_nm1,ns2ms2
                
@njit(parallel=True,cache=True)
def mean_of_var(var_vxyz,uniques,mycube_number):
    '''
    averaging the estimated variances for equal density
    '''
    number_for_weight = np.zeros(len(uniques),dtype=np.int32)
    meanvar           = np.zeros(len(uniques))
    for iii in prange(len(uniques)):
        the_indices            = np.where(mycube_number == uniques[iii])
        number_for_weight[iii] = len(the_indices[0])
        meanvar          [iii] = np.mean(var_vxyz[the_indices])
    return number_for_weight,meanvar

#####################################################################################################################
############################################### MAIN PIPELINE #######################################################
#####################################################################################################################
import os
import glob
import sys
import scipy.stats as SS
from shutil import rmtree
import warnings
warnings.filterwarnings("ignore")

#storing particles in cubical cells
x,y,z = TOPHAT_MAS(x,y,z,L,N_sample)
x = x.astype(np.int32)
y = y.astype(np.int32)
z = z.astype(np.int32)

nbr_part = len(x)

mycube_number  = (counting(x,y,z,N_sample)).astype(np.int32)

#The following is run to compute an unbiased estimate of the velocity variance in each cell
mycube_sum_vx, mycube_sumsquared_vx = stacking_for_variance(x,y,z,vx,N_sample)
n_nm1,ns2ms2                        = unbiased_variance(N_sample,mycube_number,mycube_sumsquared_vx,mycube_sum_vx)
var_vx                              = np.nan_to_num(ns2ms2/n_nm1)
del n_nm1,ns2ms2,mycube_sum_vx,mycube_sumsquared_vx

mycube_sum_vy, mycube_sumsquared_vy = stacking_for_variance(x,y,z,vy,N_sample)
n_nm1,ns2ms2                        = unbiased_variance(N_sample,mycube_number,mycube_sumsquared_vy,mycube_sum_vy)
var_vy                              = np.nan_to_num(ns2ms2/n_nm1)
del n_nm1,ns2ms2,mycube_sum_vy,mycube_sumsquared_vy

mycube_sum_vz, mycube_sumsquared_vz = stacking_for_variance(x,y,z,vz,N_sample)
n_nm1,ns2ms2                        = unbiased_variance(N_sample,mycube_number,mycube_sumsquared_vz,mycube_sum_vz)
var_vz                              = np.nan_to_num(ns2ms2/n_nm1)
del n_nm1,ns2ms2,mycube_sum_vz,mycube_sumsquared_vz

#sorting the number of particles falling in each cell
mycube_number = np.reshape(mycube_number,N_sample**3)
uniques       = np.unique(mycube_number)

#for cells defined by the same number of particles (same local density), the velocity variance is averaged
#saving each data potentialy coming from different catalogues
if rank == 0 :
    if not os.path.exists('./temporary'): os.makedirs('./temporary')

comm.Barrier()
np.random.seed()

number_for_weight,meanvar = mean_of_var(np.reshape(var_vx,N_sample**3),uniques,mycube_number)
np.savetxt('./temporary/t'+str(np.random.random()),np.transpose(np.vstack((uniques,meanvar,number_for_weight))))
number_for_weight,meanvar = mean_of_var(np.reshape(var_vy,N_sample**3),uniques,mycube_number)
np.savetxt('./temporary/t'+str(np.random.random()),np.transpose(np.vstack((uniques,meanvar,number_for_weight))))
number_for_weight,meanvar = mean_of_var(np.reshape(var_vz,N_sample**3),uniques,mycube_number)
np.savetxt('./temporary/t'+str(np.random.random()),np.transpose(np.vstack((uniques,meanvar,number_for_weight))))

# rank = 0 will join and average all data
comm.Barrier()
if rank == 0:
    filename_ = glob.glob(os.path.join('./temporary/t*'))
    uniques           = np.array([])
    meanvar           = np.array([])
    number_for_weight = np.array([])
    
    for results in filename_:
        uniques_,meanvar_,number_for_weight_ = np.loadtxt(results,unpack=1)
        os.remove(results)
        uniques           = np.concatenate((uniques,uniques_))
        meanvar           = np.concatenate((meanvar,meanvar_))
        number_for_weight = np.concatenate((number_for_weight,number_for_weight_))
    uniques_all = np.unique(uniques)
    meanvar_all = np.zeros(len(uniques_all))
    i = 0
    
    for N_value in uniques_all:
        the_indices    = np.where(uniques == N_value)
        meanvar_all[i] = np.average(meanvar[the_indices],weights=number_for_weight[the_indices])
        i += 1
        
    deltap1 = uniques_all/(nbr_part/N_sample**3)
    sigma2  = meanvar_all**2
        
    rmtree('./temporary')
    
    #apply linear regression in loglog space for the relation velocity variance as a function of local density for 50 < delta +1 < 1000 
    #this is because small delta are biased while large delta may significantly present deviation from the power law approximation
    condition = (deltap1>50)*(deltap1<1000)
    alpha,_,_,_,_ = SS.linregress(np.log(deltap1[condition]),np.log(sigma2[condition]))
    
    print('The alpha parameter to put in your .ini file is', alpha) ; sys.stdout.flush()