'''
This file is the main code of COVMOS. It must be associated to a .ini file (see setting_example.ini) and a running mode : ini, sim, or both.
You can run it using mpi (or not) calling for instance 'mpiexec -f machinefile -n 10 python COVMOS.py both path/to/setting.ini'
'''

from tools._initialisation_setting import *

from sys import argv

try: 
    COVMOS_type = str(argv[1])
    ini_file    = str(argv[2])
except: 
    raise Exception("please run in the following way: 'python COVMOS.py mode inifile' with mode in ['ini','sim','both'] and inifile being the .ini file associated to your project")
    
if COVMOS_type in ['ini','both']:
    
    from tools._Pknu_pipeline import *
    from tools._aliasing import *
    from tools._2PStat_prediction import *

    Par = read_parameters(ini_file,mode='ini')
    generate_output_repertories(Par,mode='ini')
    density_field,velocity_field = loading_ini_files(Par,mode='ini')
    k_3D,k_1D = Fouriermodes(Par)

    PDF_map = Mehler(Par,density_field)

    density_field = matching_Pk_to_PDF(density_field,PDF_map,Par)
    density_field = aliasing(density_field,Par,k_1D,k_3D)
    density_field = matching_Pk_3D_to_PDF(density_field,PDF_map,Par,k_3D)
    density_field = Pk_nu_compute(density_field,PDF_map,Par,k_3D,k_1D)

    velocity_field = get_Pk_tt_3d(velocity_field,k_3D,Par)

    save_ini_files(density_field,velocity_field,Par,PDF_map,k_3D)

    compute_2PS_predictions(Par,density_field,PDF_map,k_1D)
    
    del Par,density_field,velocity_field,k_3D,k_1D,PDF_map
                  
if COVMOS_type in ['sim','both']:
    
    from tools._catalogue_loop import *

    Par = read_parameters(ini_file,mode='sim')
    generate_output_repertories(Par,mode='sim')
    Ary = loading_ini_files(Par,mode='sim')

    generate_analyse_catalogues(Par,Ary)
                  
if not COVMOS_type in ['ini','sim','both']: 
    raise Exception('the first argument %s is not recognised. It must be either ini, sim, or both. The second argument is the path to the .ini file'%COVMOS_type)