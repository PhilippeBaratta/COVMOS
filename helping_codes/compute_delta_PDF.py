'''
This code helps the user to estimate on his own data the 1D probability distribution function of the contrast density field.
First the user has to write the function load_user_data_catalogue() (see below) used by this code to load positions of user data.
Then the code exploits NBodyKit to obtain the interpolated 3D grid of the data. Finally a simple histogram on the resulting delta is computed and saved. The ascii file can be directly used in the PDF_d_file variable (see setting_example.ini)

inputs : - a function written by the user to load his data (particle positions)
         - L, the size of the cubical box provided by the user
         - N_sample, the sampling parameter (L/N_sample (the COVMOS grid precision) must be similar to the one set in the setting.ini file)
         - output_path, the path and filename of the output PDF

output : - an ascii file storing the PDF of delta
'''

#####################################################################################################################
############################################# USER INITIALISATION ###################################################
#####################################################################################################################

def load_user_data_catalogue():
    '''
    This function must be written by the user in order to load his catalogue.
    x,y,z must all be provided in float64, in Mpc/h unit and belong to [0,L]
    '''
    
    return x,y,z

x,y,z = load_user_data_catalogue()

L           = 1000.
N_sample    = 1024
res         = 'PCS'#'NEAREST', 'LINEAR', 'NNB', 'CIC', 'TSC', 'PCS', 'QUADRATIC', 'CUBIC', 'LANCZOS2', 'LANCZOS3', 'LANCZOS4', 'LANCZOS5', 'ACG4', 'ACG5', 'ACG6', 'DB6', 'DB12', 'DB20', 'SYM6', 'SYM12', 'SYM20'
output_path = '/datadec/cppm/baratta/COVMOS/simulation_catalogues/cat_rho1.0_Ns1024_L1000.0_Pordtrilinear_lcdm_z0.0_RSDTrue_j2/PDF_PCS'


#####################################################################################################################
############################################### NBODYKIT PART #######################################################
#####################################################################################################################
import numpy as np
from nbodykit.source.catalog import ArrayCatalog

number_of_part = len(x)
data = np.empty(number_of_part, dtype=[('Position', (np.float64, 3))])
data['Position'][:,0] = x ; data['Position'][:,1] = y ; data['Position'][:,2] = z

f = ArrayCatalog({'Position' : data['Position']})                                                     ; del data
mesh_comov = f.to_mesh(Nmesh=N_sample,BoxSize=L,dtype=np.float64,resampler=res,position='Position') ; del f
delta = mesh_comov.paint(mode='real').preview() -1                                                    ; del mesh_comov

#####################################################################################################################
############################################ COMPUTING HISTOGRAM ####################################################
#####################################################################################################################

max_delta = np.amax(delta)
nbr_bins  = 1000000  
bins      = np.logspace(np.log10(1),np.log10(max_delta+2),nbr_bins)-2
PDF       = np.histogram(delta,bins=bins,density=1)   

xvalues   = [np.mean([PDF[1][i+1],PDF[1][i]]) for i in range(len(PDF[1])-1)]
x_delta   = np.linspace(-1,max_delta,nbr_bins)
PDF       = np.interp(x_delta,xvalues,PDF[0])  

np.savetxt(output_path,np.transpose(np.vstack((x_delta,PDF))))