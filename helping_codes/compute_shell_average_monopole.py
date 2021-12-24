'''
This code uses the NBodyKit module (https://github.com/bccp/nbodykit) to compute the monopole of the power spectrum (here in comoving space) on the user own data. The output ascii file can be set as an input of the Pk_dd_file variable in the .ini file (see setting_example.ini).
First the user has to write the function load_user_data_catalogue() (see below) used by this code to load positions of user data.
Note that the output power spectrum is deconvolved from the PCS grid assignment (compensated = True) and unaliased (interlaced = True).
This code can be run in mpi

inputs : - a function written by the user to load his data
         - L, the size of the cubical box
         - output_path, the path and filename of the output power spectrum

output : - an ascii file storing two columns: the wavemodes in h/Mpc and density power spectrum in [Mpc/h]^3 and in Fourier normalisation delta_k = (2pi)^-3 \int d^3x delta(x) e^(-ik.x) .
'''
#####################################################################################################################
############################################# USER INITIALISATION ###################################################
#####################################################################################################################

def load_user_data_catalogue():
    '''
    This function must be written by the user in order to load his catalogue(s).
    x,y,z must all be provided in float64, in Mpc/h unit and belong to [0,L]
    '''
    
    return x,y,z

x,y,z = load_user_data_catalogue()

L           =    #in Mpc/h
output_path = '' #path and filename of the output power spectrum

#####################################################################################################################
############################################### NBODYKIT PART #######################################################
#####################################################################################################################
import numpy as np
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import *
from nbodykit import use_mpi, CurrentMPIComm
comm = CurrentMPIComm.get()
use_mpi(comm)
   
    
number_of_part = len(x)
data = np.empty(number_of_part, dtype=[('Position', (np.float64, 3))])
data['Position'][:,0] = x ; data['Position'][:,1] = y ; data['Position'][:,2] = z
f = ArrayCatalog({'Position' : data['Position']},comm=comm) ; del data
mesh_comov = f.to_mesh(Nmesh=1024,BoxSize=L,dtype=np.float64,interlaced=True,compensated=True,resampler='pcs',position='Position') ; del f

del x,y,z
        
Pknbk = FFTPower(mesh_comov, mode='2d', poles = (0,), dk=0, kmin=2*np.pi/L) ; del mesh_comov

poles   = Pknbk.poles
nk      = int(1024/2 - 1)
k_new   = np.zeros(nk-1)
num_new = np.zeros(nk-1)
Pk_new  = np.zeros((nk-1)*len((0,)))
kF      = (2*np.pi)/L

for i in range(1,poles['k'].size) :
    norm = poles['k'][i]
    if norm <= ((nk-1)*kF + kF/2.) :
        k_new[int(np.floor((norm-kF/2.)/kF))] += norm*poles['modes'][i]/2.
        num_new[int(np.floor((norm-kF/2.)/kF))] += poles['modes'][i]/2.
        for p in range(len((0,))) :
            Pk_new[int(np.floor((norm-kF/2.)/kF)) + (nk-1)*p] += poles['power_'+str((0,)[p])][i].real*poles['modes'][i]/2.

k_new /= num_new
for p in range(len((0,))): Pk_new[p*k_new.size : k_new.size*(p+1)] /= num_new

Pk_new[0 : k_new.size] -= poles.attrs['shotnoise']

if comm.rank == 0: 
    np.savetxt(output_path,np.transpose(np.vstack((k_new,Pk_new[0 : k_new.size]/(2*np.pi)**3))))
    print('The monopole of the power spectrum has been saved in',output_path)
