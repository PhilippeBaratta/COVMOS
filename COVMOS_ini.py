'''
This code allows the production of the files necessary to run the code COVMOS_sim.py. It must be associated to a .ini file (see setting_example.ini).
You can run it in mpi (or not) using for instance 'mpiexec -f machinefile -n 10 python /renoir/baratta/COVMOS_public/COVMOS/COVMOS_ini.py setting_example'
'''
import sys
from initialization_funcs import *

Par = read_parameters(str(sys.argv[1]))
generate_output_repertories(Par)
density_field,velocity_field = loading_ini_files(Par)

k_3D,k_1D = Fouriermodes(Par,mode=1)

PDF_map = Mehler(Par,density_field)

density_field = matching_Pk_to_PDF(density_field,PDF_map,Par)
density_field = aliasing(density_field,Par,k_1D,k_3D)
density_field = matching_Pk_3D_to_PDF(density_field,PDF_map,Par,k_3D)
density_field = Pk_nu_compute(density_field,PDF_map,Par,k_3D)

velocity_field = get_Pk_tt_3d(velocity_field,k_3D,Par)

save_ini_files(density_field,velocity_field,Par,PDF_map)

compute_2PS_predictions(Par,density_field,PDF_map,k_1D)