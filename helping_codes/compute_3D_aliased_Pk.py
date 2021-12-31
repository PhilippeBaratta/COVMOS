'''
This code helps the user to estimate on his own data the 3D power spectrum, naturally aliased thanks to FFT. Note that this method is better than artificially aliasing a 1D power spectrum since all alias contribution are naturraly added to the true signal. First the user has to write the function load_user_data_catalogue() (see below) used by this code to load positions of user data. Then the code assign particles on the mesh using the NbodyKit module. The obtained delta grid is transformed in Fourier space using FFT and the 3D power spectrum is computed.
Note that this code can be run in MPI if several input catalogues are provided by the user. In this case the resulting 3D power spectrum is the averaged one.

inputs : - a function written by the user to load his data
         - number_of_cats, the number of catalogues to be loaded
         - L, the size of the cubical box
         - N_sample, the sampling parameter (L/N_sample (the grid precision) must be similar to the one set in the setting.ini file)
         - MAS is the type of mass assignment scheme used to assign particles on the mesh, like 'CIC', 'TSC', 'PCS'... see NBodyKit documentation for 
           other kinds of MAS. Note that the MAS parameter must be similar to the one given in compute_delta_PDF.py
         - output_path is the path and filename of the output power spectrum
output : - the (averaged) 3D aliased power spectrum in a .npy file
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
    from scipy.io.idl import readsav
    
    N_DEMNUni = 1024**3
    
    icovstr = '0'*(ref_cat_i<10) + str(ref_cat_i)
    
    x=np.zeros(N_DEMNUni)
    y=np.zeros(N_DEMNUni)
    z=np.zeros(N_DEMNUni)
    
    ref_dep = 0
    for slices in range(512):
        slice_xyz = readsav('/datadec/cpt/jbel/IDL_CATALOGUES/demnuniicov%s_0z_%s.data'%(icovstr,slices),verbose=0)
        ref_fin   = len(slice_xyz.xyz[:,0]) + ref_dep
        x[ref_dep:ref_fin]  = slice_xyz.xyz[:,0] 
        y[ref_dep:ref_fin]  = slice_xyz.xyz[:,1] 
        z[ref_dep:ref_fin]  = slice_xyz.xyz[:,2] 
        ref_dep = ref_fin*1
        
    return x,y,z


MAS            = 'PCS'
number_of_cats = 50
L              = 1000. #in Mpc/h. Once again L/N_sample (the grid precision) must be similar to the one set in the setting.ini file
N_sample       = 1024
ref_cat        = np.arange(50)+1 #each element of this array will pass as argument of load_user_data_catalogue()
output_path    = '/renoir/baratta/COVMOS_public/COVMOS/target_stat_example/Pk_z0_lcdm_cic_1024' #path and filename of the output power spectrum


#####################################################################################################################
############################################# SPLITTING JOBS IF MPI #################################################
#####################################################################################################################

def spliting_one_array(arr1,number_at_the_end):
    redivision = np.linspace(0,len(arr1),number_at_the_end+1,dtype=int)
    arr1_ = np.split(arr1,redivision[1:-1])
    return arr1_;

ref_cat_rank = spliting_one_array(ref_cat,size)[rank]

#####################################################################################################################
############################################### MAIN PIPELINE #######################################################
#####################################################################################################################

from nbodykit.source.catalog import ArrayCatalog

Pk = np.zeros((N_sample,N_sample,N_sample))

for ref_cat_i in ref_cat_rank:
    
    x,y,z = load_user_data_catalogue(ref_cat_i)

    number_of_part = len(x)
    data = np.empty(number_of_part, dtype=[('Position', (np.float64, 3))])
    data['Position'][:,0] = x ; data['Position'][:,1] = y ; data['Position'][:,2] = z
    
    f = ArrayCatalog({'Position' : data['Position']},comm=MPI.COMM_SELF)                                  ; del data
    mesh_comov = f.to_mesh(Nmesh=N_sample,BoxSize=L,dtype=np.float64,resampler=MAS,position='Position')   ; del f
    delta = mesh_comov.paint(mode='real').preview() -1                                                    ; del mesh_comov
    
    delta = np.fft.fftn(delta)
    Pk += np.real(delta*np.conj(delta))
    
    
if rank == 0:  totals = np.zeros_like(Pk)
else: totals = None

comm.Barrier()
comm.Reduce( [Pk, MPI.DOUBLE], [totals, MPI.DOUBLE], op = MPI.SUM,root = 0)

if rank == 0: 
    k_F     = 2*np.pi/L  
    totals /=  (50*N_sample**6 *k_F**3)   
    np.save(output_path,totals)