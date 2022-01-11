'''
This code helps the user to estimate on his own data the one-point probability distribution function of the contrast density field.
First the user has to write the function load_user_data_catalogue() (see below) used by this code to load positions of his data.
Then the code exploits NBodyKit to obtain the interpolated 3D grid of the data. Finally a simple histogram on the resulting delta is computed and saved. The output ascii file can directly be used in the PDF_d_file variable (see setting_example.ini). Note that this code can be run in MPI if several input catalogues are provided by the user. In this case the resulting PDF is the averaged one.

inputs : - a function written by the user to load his data (particle positions only)
         - number_of_cats, the number of catalogues to be loaded
         - L, the size of the cubical box provided by the user
         - N_sample, the sampling parameter (L/N_sample (the COVMOS grid precision) must be similar to the one set in the setting.ini file)
         - MAS is the type of mass assignment scheme used to assign particles on the mesh, like 'NEAREST', 'CIC', 'TSC', 'PCS'... see NBodyKit
           documentation for other kinds of MAS.
         - ref_cat is an array consisting of labels distinguishing the different catalogues, ex: [1,2,3,4,5]
         - output_path, the path and filename of the output PDF

output : - an ascii file storing the (averaged or not) PDF of delta

comments : Since COVMOS distributes particles following a tophat or a trilinear scheme (available in this version), for COVMOS to be consistent and
           produce the right PDF, you should use the 'NEAREST' or 'CIC' MAS, respectivelly. Obviously this can be done if and only if the number of
           particles in the simulation is sufficient to provide an un-noised PDF. We advise the user to plot the result of this code before going
           further
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
     
    return x.astype(np.float64),y.astype(np.float64),z.astype(np.float64)


number_of_cats = 
L              =    #in Mpc/h. Once again L/N_sample (the grid precision) must be similar to the one set in the setting.ini file
N_sample       = 
MAS            = 
ref_cat        = [] #each element of this array is associated to a catalogue and will be passed as input of load_user_data_catalogue()
output_path    =    #path and filename of the output power spectrum

#####################################################################################################################
############################################# SPLITTING JOBS IF MPI #################################################
#####################################################################################################################

def spliting_one_array(arr1,number_at_the_end):
    redivision = np.linspace(0,len(arr1),number_at_the_end+1,dtype=int)
    arr1_ = np.split(arr1,redivision[1:-1])
    return arr1_;

ref_cat_rank = spliting_one_array(ref_cat,size)[rank]

simu_ref      = np.arange(len(ref_cat))
simu_ref_rank = spliting_one_array(simu_ref,size)[rank]

#####################################################################################################################
############################################### NBODYKIT PART #######################################################
#####################################################################################################################

from nbodykit.source.catalog import ArrayCatalog
import os
import shutil

if rank == 0: os.makedirs(output_path,exist_ok=True)

for i in range(ref_cat_rank.size):
    
    x,y,z = load_user_data_catalogue(ref_cat_rank[i])

    number_of_part = len(x)
    data = np.empty(number_of_part, dtype=[('Position', (np.float64, 3))])
    data['Position'][:,0] = x ; data['Position'][:,1] = y ; data['Position'][:,2] = z
    
    f = ArrayCatalog({'Position' : data['Position']},comm=MPI.COMM_SELF)                                  ; del data
    mesh_comov = f.to_mesh(Nmesh=N_sample,BoxSize=L,dtype=np.float64,resampler=MAS,position='Position')   ; del f
    delta = mesh_comov.paint(mode='real').preview() -1                                                    ; del mesh_comov
    np.save(output_path + '/' + str(simu_ref_rank[i]),delta)
    
#####################################################################################################################
############################################ COMPUTING HISTOGRAM ####################################################
#####################################################################################################################

comm.Barrier()

if rank==0:
    delta = np.zeros((N_sample,N_sample,N_sample,number_of_cats))
    for icov in simu_ref:
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