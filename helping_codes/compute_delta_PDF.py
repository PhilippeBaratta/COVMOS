'''
This code helps the user to estimate on his own data the 1D probability distribution function of the contrast density field.
First the user has to write the function load_user_data_catalogue() (see below) used by this code to load positions of user data.
Then the code exploits NBodyKit to obtain the interpolated 3D grid of the data. Finally a simple histogram on the resulting delta is computed and saved. The ascii file can be directly used in the PDF_d_file variable (see setting_example.ini). Note that this code can be run in MPI if several input catalogues are provided by the user. In this case the resulting PDF is the averaged one.

inputs : - a function written by the user to load his data (particle positions)
         - number_of_cats, the number of catalogues to be loaded
         - L, the size of the cubical box provided by the user
         - N_sample, the sampling parameter (L/N_sample (the COVMOS grid precision) must be similar to the one set in the setting.ini file)
         - MAS is the type of mass assignment scheme used to assign particles on the mesh, like 'CIC', 'TSC', 'PCS'... see NBodyKit documentation for 
           other kinds of MAS.
         - output_path, the path and filename of the output PDF

output : - an ascii file storing the (averaged) PDF of delta
'''

#####################################################################################################################
############################################# USER INITIALISATION ###################################################
#####################################################################################################################

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD ; size = comm.Get_size() ; rank = comm.Get_rank()

def load_user_data_catalogue(ref_cat_i):
    '''
    This function must be written by the user in order to load his catalogue(s). If the code is run in mpi, several catalogues (only defined by different initial condition, but same parameters) can be treated in parallel. In this case, each MPI rank must return one catalogue.
    x,y,z must all be provided in Mpc/h and x,y,z must all belong to [0,L]
    '''
        
    return x,y,z


MAS            = 'PCS'
number_of_cats = 
L              =  #in Mpc/h. Once again L/N_sample (the grid precision) must be similar to the one set in the setting.ini file
N_sample       = 
ref_cat        =  #each element of this array will pass as argument of load_user_data_catalogue()
output_path    =  #path and filename of the output power spectrum

#####################################################################################################################
############################################# SPLITTING JOBS IF MPI #################################################
#####################################################################################################################

def spliting_one_array(arr1,number_at_the_end):
    redivision = np.linspace(0,len(arr1),number_at_the_end+1,dtype=int)
    arr1_ = np.split(arr1,redivision[1:-1])
    return arr1_;

ref_cat_rank = spliting_one_array(ref_cat,size)[rank]

#####################################################################################################################
############################################### NBODYKIT PART #######################################################
#####################################################################################################################

from nbodykit.source.catalog import ArrayCatalog
import os
import shutil

if rank == 0: os.makedirs(output_path,exist_ok=True)

for ref_cat_i in ref_cat_rank:
    
    x,y,z = load_user_data_catalogue(ref_cat_i)

    number_of_part = len(x)
    data = np.empty(number_of_part, dtype=[('Position', (np.float64, 3))])
    data['Position'][:,0] = x ; data['Position'][:,1] = y ; data['Position'][:,2] = z
    
    f = ArrayCatalog({'Position' : data['Position']},comm=MPI.COMM_SELF)                                  ; del data
    mesh_comov = f.to_mesh(Nmesh=N_sample,BoxSize=L,dtype=np.float64,resampler=MAS,position='Position')   ; del f
    delta = mesh_comov.paint(mode='real').preview() -1                                                    ; del mesh_comov
    np.save(output_path + '/' + str(ref_cat_i),delta)
    
#####################################################################################################################
############################################ COMPUTING HISTOGRAM ####################################################
#####################################################################################################################

comm.Barrier()

if rank==0:
    delta = np.zeros((N_sample,N_sample,N_sample,50))
    for icov in range (1,51):
        cat = np.load(output_path + '/' + str(icov)+'.npy')
        delta[:,:,:,icov-1] = cat
        os.remove(output_path + '/' + str(icov)+'.npy')
    shutil.rmtree(output_path)
    
    max_delta = np.amax(delta)
    nbr_bins  = 1000000  
    bins      = np.logspace(np.log10(1),np.log10(max_delta+2),nbr_bins)-2
    PDF       = np.histogram(delta,bins=bins,density=1)   

    xvalues   = [np.mean([PDF[1][i+1],PDF[1][i]]) for i in range(len(PDF[1])-1)]
    x_delta   = np.linspace(-1,max_delta,nbr_bins)
    PDF       = np.interp(x_delta,xvalues,PDF[0])  

    np.savetxt(output_path,np.transpose(np.vstack((x_delta,PDF))))