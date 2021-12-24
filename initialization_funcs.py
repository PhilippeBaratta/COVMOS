import numpy as np
import os
import sys
import warnings
from numba_functions import *

#############################################################################################################################
############################################# NETWORK AND PARALLEL FUNCTIONS ################################################
#############################################################################################################################
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def from_globalcomm_to_intranode_comm():
    '''
    generate a mpi communicator between processes sharing the same RAM, usefull when sharing common numpy arrays
    '''
    processor_name     = MPI.Get_processor_name()
    processor_name_all = comm.gather(processor_name, root=0)
    
    if rank == 0 :
        unique = np.unique(processor_name_all)
    else : unique = None
    
    unique = comm.scatter([unique for i in range(size)], root=0)
    
    billy_color = np.arange(unique.size)
    billy_index = np.where(unique==processor_name)[0][0]
    local_billy = billy_color[billy_index]
        
    intracomm = MPI.Comm.Split(comm,local_billy)
    return intracomm

intracomm = from_globalcomm_to_intranode_comm()
intrarank = intracomm.rank

def sharing_array_throw_MPI(shape,comm1,dtype_str):
    '''
    define a buffer in order to share arrays on processes sharing the same RAM
    '''
    rank1 = comm1.Get_rank()
    size = np.prod(shape)
    if dtype_str == 'float64': itemsize = MPI.DOUBLE.Get_size() 
    if dtype_str == 'float32': itemsize = MPI.FLOAT.Get_size()
    if dtype_str == 'int16':   itemsize = MPI.SHORT.Get_size() 
    if dtype_str == 'int32':   itemsize = MPI.INT.Get_size()
    if rank1 == 0: nbytes = size * itemsize 
    else: nbytes = 0
    win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm1) 
    buf, itemsize = win.Shared_query(0) 
    if dtype_str == 'float64': 
        assert itemsize == MPI.DOUBLE.Get_size() 
        ary = np.ndarray(buffer=buf, dtype=np.float64, shape=shape)
    if dtype_str == 'float32': 
        assert itemsize == MPI.FLOAT.Get_size() 
        ary = np.ndarray(buffer=buf, dtype=np.float32, shape=shape)
    if dtype_str == 'int16':   
        assert itemsize == MPI.SHORT.Get_size() 
        ary = np.ndarray(buffer=buf, dtype=np.int16, shape=shape)
    if dtype_str == 'int32':   
        assert itemsize == MPI.INT.Get_size() 
        ary = np.ndarray(buffer=buf, dtype=np.int32, shape=shape)
    return ary

def spliting_one_array(arr1,number_at_the_end):
    '''
    splits the elements of an array in number_at_the_end sub arrays, usefull for parallel computing
    '''
    redivision = np.linspace(0,len(arr1),number_at_the_end+1,dtype=int)
    arr1_ = np.split(arr1,redivision[1:-1])
    return arr1_;

def spliting(arr1,arr2,arr3,number_at_the_end):
    '''
    splits, for 3 differents arrays, the elements in number_at_the_end sub arrays, usefull for parallel computing
    '''
    redivision = np.linspace(0,len(arr1),number_at_the_end+1,dtype=int)
    arr1_ = np.split(arr1,redivision[1:-1])
    arr2_ = np.split(arr2,redivision[1:-1])
    arr3_ = np.split(arr3,redivision[1:-1])
    return arr1_,arr2_,arr3_;

#############################################################################################################################
############################################## INITIALISATION FUNCTIONS #####################################################
#############################################################################################################################
def read_parameters(inifile):
    '''
    read, test and complete all the parameters of the .ini file provided by the user
    It returns a dictionnary of the COVMOS parameters
    '''
    from configparser import ConfigParser
    import ast
    
    def filtering_i1_i2(redshift):
        '''
        i1 represents the filtering to apply on the input power spectrum in order for it to be compatible with the PDF
        i2 is the filtering to apply on the clipped power spectrum for the Gaussian field
        i1 and i2 are empirically chosen to maximize the range of well simulated modes
        '''
        z_ref  = [0.,0.48551,1.05352,1.45825,2.05053]
        i1_ref = [3,3,4,4,5]
        i1     = np.interp(redshift,z_ref,i1_ref)
        i2     = 12
        return i1,i2
    
    config = ConfigParser()
    config.read(inifile)
    Par = {}
    
    if config.getboolean('OUTPUTS', 'verbose') and rank == 0:
        print('\n________________________ READING/COMPUTING STATISTICAL INPUTS ________________________')
        sys.stdout.flush()
        
    ##############################################  CATALOGUE_SETTINGS  #######################################################
    var_type = 'CATALOGUE_SETTINGS'
    
    Par['project_name'] = config.get(var_type, 'project_name')
    
    try: Par['total_number_of_cat'] = config.getint(var_type, 'total_number_of_cat')
    except: raise Exception('the total_number_of_cat parameter must be an integer, please correct the ini file')
    if Par['total_number_of_cat'] <= 0 : raise Exception('the total_number_of_cat parameter must be > 0, please correct the ini file')
        
    try: Par['rho_0'] = config.getfloat(var_type, 'rho_0')
    except: raise Exception('the rho_0 parameter must be an integer or a float, please correct the ini file')
    if Par['rho_0'] <= 0 : raise Exception('the rho_0 parameter must be > 0, please correct the ini file, please correct the ini file')
        
    Par['assign_scheme'] = config.get(var_type, 'assign_scheme')
    if Par['assign_scheme'] != 'tophat' and Par['assign_scheme'] != 'trilinear':
        raise Exception('you must choose the assign_scheme parameter between tophat or trilinear, please correct the ini file')
    
    Par['fixed_Rdm_seed'] = config.getboolean(var_type, 'fixed_Rdm_seed')
    
    try: Par['L'] = config.getfloat(var_type, 'L')
    except: raise Exception('the L parameter must be an integer or a float, please correct the ini file')
    if Par['L'] <= 0 : raise Exception('the L parameter must be > 0, please correct the ini file')
        
    try: Par['N_sample'] = config.getint(var_type, 'N_sample')
    except: raise Exception('the N_sample parameter must be an integer (and power of 2), please correct the ini file')
    if not np.log(Par['N_sample']) % np.log(2) < 1e-10 :
        raise Exception('the N_sample parameter must be a power of 2 (ex. 256, 512, 1024 ...), please correct the ini file')
    if Par['N_sample'] <= 0 :
        raise Exception('the N_sample parameter must be > 0 and a power of 2 (ex. 256, 512, 1024 ...), please correct the ini file')
        
    try: Par['redshift'] = config.getfloat(var_type, 'redshift')
    except: raise Exception('the redshift parameter must be an integer or a float, please correct the ini file')
    if Par['redshift'] < 0: raise Exception('the redshift parameter must be >= 0, please correct the ini file')
                            
    try: Par['Omega_m'] = config.getfloat(var_type, 'Omega_m')
    except: raise Exception('the Omega_m parameter must be an integer or a float, please correct the ini file')
    if Par['Omega_m'] <= 0: raise Exception('the Omega_m parameter must be > 0, please correct the ini file')
        
    try: 
        Par['aliasing_order'] = config.getint(var_type, 'aliasing_order')
        if Par['aliasing_order'] < 0: raise Exception('the aliasing_order must be an integer >= 0, please correct the ini file')
    except: 
        Par['aliasing_order'] = config.get(var_type, 'aliasing_order')
        if not Par['aliasing_order'] == 'Default':
            raise Exception('If you want to chose aliasing_order = Default, please write it correctly')
    
    ###############################################  TARGET_STATISTICS  #######################################################
    var_type = 'TARGET_STATISTICS'
    
    Par['Pk_dd_file'] = config.get(var_type, 'Pk_dd_file')
    if Par['Pk_dd_file'] == '' or Par['Pk_dd_file'] == "''": Par['Pk_dd_file'] = ''
    if not os.path.exists(Par['Pk_dd_file']) and not Par['Pk_dd_file'] == '':
        raise Exception('Pk_dd_file not found:',Par['Pk_dd_file'])
    if os.path.exists(Par['Pk_dd_file']):
        if rank == 0:
            if (not Par['Pk_dd_file'][-4:] == '.npy'):
                try: 
                    kk,_ = np.loadtxt(Par['Pk_dd_file'],unpack=1)
                    if kk[0] > 2*np.pi/Par['L'] : raise Exception('the kmin in',Par['Pk_dd_file'],'is too high (> 2pi/L), you should extrapolate it down to lower modes')
                except: raise Exception('the ascii file associated to Pk_dd_file is not composed of two columns (k in h/Mpc and Pk in [Mpc/h]^3)')
            else:
                npy_file = np.load(Par['Pk_dd_file'])
                if not npy_file.shape == (Par['N_sample'],Par['N_sample'],Par['N_sample']):
                    raise Exception('the provided .npy file for Pk_dd_file is associated to a numpy array of shape different of (%i,%i,%i), please provide a different file, or change the N_sample parameter to %i'%(Par['N_sample'],Par['N_sample'],Par['N_sample'],npy_file.shape[0]))
    
    Par['PDF_d_file'] = config.get(var_type, 'PDF_d_file')
    if not os.path.exists(Par['PDF_d_file']) and not Par['PDF_d_file'] == 'gaussian' :
        raise Exception('PDF_d_file not found:',Par['PDF_d_file'], 'if you want a gaussian PDF, type gaussian')
    if os.path.exists(Par['PDF_d_file']):
        if rank == 0:
            try: _,_ = np.loadtxt(Par['PDF_d_file'],unpack=1)
            except: raise Exception('the ascii file associated to PDF_d_file is not composed of two columns (delta and PDF(delta)')
    
    try: 
        Par['filtering_parameter'] = config.getfloat(var_type, 'filtering_parameter')
        if Par['filtering_parameter'] <= 0:
            raise Exception('the filtering_parameter must be > 0, please correct the ini file')
    except: 
        Par['filtering_parameter'] = config.get(var_type, 'filtering_parameter')
        if not Par['filtering_parameter'] == 'Default':
            raise Exception('If you want to chose filtering_parameter = Default, please write it correctly')
    if Par['filtering_parameter'] == 'Default': Par['i1'],Par['i2'] = filtering_i1_i2( Par['redshift'] )
    else:
        Par['i1'] = Par['filtering_parameter']
        _,Par['i2'] = filtering_i1_i2( Par['redshift'] )
    
    ###########################################################
    Par['velocity'] = config.getboolean('OUTPUTS', 'velocity')# need it now
    ###########################################################
    
    if Par['velocity']:
        Par['Pk_tt_file'] = config.get(var_type, 'Pk_tt_file')
        if Par['Pk_tt_file'] == '' or Par['Pk_tt_file'] == "''": Par['Pk_tt_file'] = ''
        if not os.path.exists(Par['Pk_tt_file']) and not Par['Pk_tt_file'] == '':
            raise Exception('Pk_tt_file not found:',Par['Pk_tt_file'])
        if os.path.exists(Par['Pk_tt_file']):
            if rank == 0:
                try: 
                    kk,_ = np.loadtxt(Par['Pk_tt_file'],unpack=1)
                    if kk[0] > 2*np.pi/Par['L'] : raise Exception('the kmin in',Par['Pk_tt_file'],'is too high (> 2pi/L), you should extrapolate it down to lower modes')
                except: raise Exception('the ascii file associated to Pk_tt_file is not composed of two columns (k in h/Mpc and Pk in [Mpc/h]^3)')
    
        
    if Par['Pk_dd_file'] == '' or Par['velocity'] * (Par['Pk_tt_file'] == '') :
        try:
            class_dic_param = ast.literal_eval(config.get(var_type, 'classy_dict'))
            class_dic_param['P_k_max_h/Mpc'] = 50
            class_dic_param['z_pk'] = Par['redshift']
            Par['classy_dict'] = class_dic_param
        except: raise Exception('a classy dictionnary (classy_dict) with cosmological parameters must be provided if Pk_dd_file is empty or if Pk_tt_file is empty (while velocity = True)')
    
    if Par['velocity']:
        try: Par['targeted_rms'] = config.getfloat(var_type , 'targeted_rms')
        except: raise Exception('the targeted_rms parameter must be an integer or a float, please correct the ini file')
        if Par['targeted_rms'] <= 0 : raise Exception('the targeted_rms parameter must be > 0, please correct the ini file')
            
        try: Par['alpha'] = config.getfloat(var_type , 'alpha')
        except: raise Exception('the alpha parameter must be an integer or a float, please correct the ini file')
        if Par['alpha'] <= 0 : raise Exception('the alpha parameter must be > 0, please correct the ini file')
            
    #####################################################  OUTPUTS  ###########################################################
    var_type = 'OUTPUTS'
    
    Par['output_dir'] = config.get(var_type, 'output_dir')
    Par['compute_Pk_prediction'] = config.getboolean(var_type, 'compute_Pk_prediction')
    Par['compute_2pcf_prediction'] = config.getboolean(var_type, 'compute_2pcf_prediction')
    
    Par['estimate_Pk_multipoles']  = config.get(var_type, 'estimate_Pk_multipoles')
    if Par['estimate_Pk_multipoles'] == 'False': Par['estimate_Pk_multipoles'] = False
    if not (Par['estimate_Pk_multipoles'] == 'stopandrun' or Par['estimate_Pk_multipoles'] == 'detached' or Par['estimate_Pk_multipoles'] == False):
        raise Exception('the estimate_Pk_multipoles parameter must be set to stopandrun or detached or False')
    
    Par['save_catalogue'] = config.getboolean(var_type, 'save_catalogue')
    Par['verbose']        = config.getboolean(var_type, 'verbose')
    Par['debug']          = config.getboolean(var_type, 'debug')
    
    if Par['aliasing_order'] == 'Default' and not Par['Pk_dd_file'][-4:] == '.npy': Par['aliasing_order'] = 2
    
    if Par['project_name'] == '' or Par['project_name'] == "''":
        Par['project_name'] = 'COVMOS_' + 'Ns' + str(Par['N_sample']) + '_L' + str(Par['L']) + '_z' + str(Par['redshift']) + '_rho' + str(Par['rho_0']) + '_Om' + str(Par['Omega_m']) + '_' + Par['assign_scheme'] + '_scheme' + (not Par['Pk_dd_file'][-4:] == '.npy') * '_aliasOrd' + str(Par['aliasing_order']) + Par['velocity'] * ('_alpha' + str(Par['alpha']) + '_Vrms' + str(Par['targeted_rms'])) + Par['fixed_Rdm_seed'] * '_fixed_Rdm_seed'
    
    Par['output_dir_project'] = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + Par['project_name']
    
    if Par['verbose'] and rank == 0 :
        print('the ini file has been correctly read') ; sys.stdout.flush()
    
    comm.Barrier()
    return Par

def generate_output_repertories(Par):
    '''
    generate the diverse output repertories and logfile from COVMOS_ini.py
    '''
    if rank == 0:
        if Par['verbose']: print('generating the various output repertories in', Par['output_dir_project']) ; sys.stdout.flush()
        if not os.path.exists(Par['output_dir_project']): os.makedirs(Par['output_dir_project'])
        if not os.path.exists(Par['output_dir_project'] + '/ini_files'): os.makedirs(Par['output_dir_project'] + '/ini_files')
        if Par['compute_Pk_prediction'] or Par['compute_2pcf_prediction']: 
            if not os.path.exists(Par['output_dir_project'] + '/TwoPointStat_predictions'):os.makedirs(Par['output_dir_project'] + '/TwoPointStat_predictions')
        if Par['debug']: 
            if not os.path.exists(Par['output_dir_project'] + '/debug_files'): os.makedirs(Par['output_dir_project'] + '/debug_files')
            
        if Par['verbose']: print('saving all the input parameters in', Par['output_dir_project'] + '/setting.log') ; sys.stdout.flush()
        out = Par['output_dir_project'] + '/setting.log'
        fo  = open(out, "w")
        for k, v in Par.items(): fo.write(str(k) + '=' + str(v) + '\n')
        fo.close()
    comm.Barrier()
    return

def loading_ini_files(Par):
    '''
    loading or computing the density and velocities files, referred as inputs by the user
    '''
    def extrap_k_loglin(k_max,k,Pk,firstp):
        '''
        performs an extrapolation of the power spectrum, necessary for the aliasing procedure
        '''
        def extrap(x_extrap,x,y,firstp): #for this power law extrapolation I want a linear extrapolation in log scale, firstp is set usually a 2
            xlog  = np.log(x)
            ylog  = np.log(y)
            slope = (ylog[-1]-ylog[-firstp])/(xlog[-1]-xlog[-firstp])
            beta  = ylog[-1]-slope*xlog[-1] # y=slope*x+beta
            return np.exp(slope*np.log(x_extrap)+beta);
        Pkmax      = extrap(k_max,k,Pk,firstp)
        k_etptra    = np.linspace(np.log(np.amax(k)),np.log(k_max),100)[1:]
        Pk_Syl_etptra   = np.interp(k_etptra,np.log(np.array([k[-1],k_max])),np.log(np.array([Pk[-1],Pkmax])))
        k          = np.insert(k,len(k),np.exp(k_etptra))
        Pk         = np.insert(Pk,len(Pk),np.exp(Pk_Syl_etptra))
        return k,Pk
    
    def classy_compute_Pk(Par,prescription = 'Default'):
        '''
        compute the Pk or Pk_cb (if massive neutrinos are provided), given a classy dict in units of h, and in Fourier normalisation 
        delta_k = (2pi)^-3 \int d^3x delta(x) e^(-ik.x)
        if prescription = 'Default' then the power spectrum is computes as defined in the classy dict. If prescription = 'linear', in any case the power spectrum is computed using a linear prescription
        '''
        from classy import Class    
        cosmo = Class()
        
        Pk_classy  = sharing_array_throw_MPI((2,1000),comm,'float64')
        sigma8m    = sharing_array_throw_MPI((1,),comm,'float64')
        if rank == 0:
            if ('non linear' in Par['classy_dict']) and prescription == 'linear': del Par['classy_dict']['non linear']
            if Par['verbose']: 
                if prescription == 'Default': 
                    print('computing the class power spectrum (for the density field) following the dict provided by the user :', Par['classy_dict'])
                else: 
                    print('computing the class power spectrum following with a linear prescription for the velocity field :', Par['classy_dict'])
                sys.stdout.flush()
            cosmo.set(Par['classy_dict'])
            cosmo.compute()
            sigma8m[:]  = cosmo.sigma(R=8/cosmo.h(),z=Par['redshift'])

            kk = np.geomspace(1e-4,50,1000) * cosmo.h()
            Pk_grid  = np.zeros(len(kk))

            if 'm_ncdm' in Par['classy_dict']:
                for ik in range(len(kk)): Pk_grid[ik] = cosmo.pk_cb(kk[ik],Par['redshift']) 
            else :
                for ik in range(len(kk)): Pk_grid[ik] = cosmo.pk   (kk[ik],Par['redshift']) 

            Pk_classy[0,:] = kk / cosmo.h()
            Pk_classy[1,:] = Pk_grid * cosmo.h()**3 / (2*np.pi)**3

        comm.Barrier()
        return Pk_classy,sigma8m
    
    def Bel_et_al_fitting_functions(sigma8m,Pk_1D_dd_lin,Par):
        '''
        uses arxiv.org/abs/1906.07683 to compute the theta-theta power spectrum from the linear power spectrum using fitting functions
        '''
        def f_func(a,Om_m0): 
            return Om_m(a,Om_m0)**alpha(a,Om_m0)
        def Om_m(a,Om_m0):   
            return Om_m0/(a**3*E(a,Om_m0)**2)
        def E(a,Om_m0):   
            return np.sqrt(Om_m0/a**3 + (1-Om_m0))
        def alpha(a,Om_m0):  
            return 6/11 -15/2057 * np.log(Om_m(a,Om_m0)) + 205/540421 * np.log(Om_m(a,Om_m0))**2
        def ztoa(z):   
            return 1/(1+z)
        
        a1 = -0.817 + 3.198 * sigma8m ; a2 =  0.877 - 4.191 * sigma8m ; a3 = -1.199 + 4.629 * sigma8m
        Pk_tt_lin    = f_func(ztoa(Par['redshift']),Par['Omega_m'])**2 * Pk_1D_dd_lin[1]    
        Pk_tt_theo1d = Pk_tt_lin * np.exp(- Pk_1D_dd_lin[0] * (a1+a2*Pk_1D_dd_lin[0]+a3*Pk_1D_dd_lin[0]**2)) #non linear thetha-theta power spectrum
        return np.vstack((Pk_1D_dd_lin[0],Pk_tt_theo1d))
    
    density_field  = {} ; velocity_field = {}
    
    if Par['PDF_d_file'] == 'gaussian': density_field['PDF_delta'] = 'gaussian'
    else:                               density_field['PDF_delta'] = np.transpose(np.loadtxt(Par['PDF_d_file']))
    
    if Par['Pk_dd_file'][-4:] == '.npy' and rank == 0: density_field['Pk_3D_dd_alias'] = np.load(Par['Pk_dd_file'])
    else:
        if Par['Pk_dd_file'] == '': density_field['Pk_1D_dd'],_ = classy_compute_Pk(Par)
        else :
            k_dd_,Pk_1D_dd_ = np.loadtxt(Par['Pk_dd_file'],unpack=1)
            if k_dd_[-1]<50: k_dd_,Pk_1D_dd_ = extrap_k_loglin(50,k_dd_,Pk_1D_dd_,2)
            density_field['Pk_1D_dd'] = np.vstack((k_dd_,Pk_1D_dd_))
        if Par['debug'] and rank == 0: np.savetxt(Par['output_dir_project'] + '/debug_files/Pk_1D_dd',np.transpose(density_field['Pk_1D_dd']))
    
    if Par['velocity']:
        if Par['Pk_tt_file'] == '':
            Pk_1D_dd_lin,sigma8m       = classy_compute_Pk(Par,prescription = 'linear')
            velocity_field['Pk_1D_tt'] = Bel_et_al_fitting_functions(sigma8m,Pk_1D_dd_lin,Par)
        else: 
            k_tt_,Pk_1D_tt_ = np.loadtxt(Par['Pk_tt_file'],unpack=1)
            if k_dd_[-1]<10: k_tt_,Pk_1D_tt_ = extrap_k_loglin(10,k_tt_,Pk_1D_tt_,2)
            velocity_field['Pk_1D_tt'] = np.vstack((k_tt_,Pk_1D_tt_))
        if Par['debug'] and rank == 0: np.savetxt(Par['output_dir_project'] + '/debug_files/Pk_1D_tt',np.transpose(velocity_field['Pk_1D_tt']))
            
    if Par['verbose'] and rank == 0 :
        print('the statistical targets have been successfully read/computed') ; sys.stdout.flush()
    
    comm.Barrier()
    return density_field,velocity_field


def Fouriermodes(Par,mode=0):
    '''
    setting the Fourier modes on the grid and computing the shell-averaged modes
    it returns k_3D in float64, k_3D_2 in float32 and k_1D in float64
    mode = 0 -> all output are returned (k_3D,k_3D_2,k_1D)
    mode = 1 -> output 0 and 2 are returned (k_3D,k_1D)
    mode = 2 -> output 1 are returned (k_3D_2)
    '''
    Ns = Par['N_sample'] ; L  = Par['L'] ; k_F = 2*np.pi/L
    Fourier_file = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'Fouriermodes_L%s_Ns%i'%(L,Ns)
    
    if os.path.exists(Fourier_file + '.npz'):
        if Par['verbose'] and rank == 0: 
            print('load Fourier modes from', Fourier_file + '.npz') ; sys.stdout.flush()
        
        file = np.load(Fourier_file + '.npz')
        if mode == 0 or mode == 2:
            k_3D_2 = file['arr_1']
        if mode == 0 or mode == 1:
            k_3D = file['arr_0']
            k_1D = file['arr_2']
    else:
        if Par['verbose'] and rank == 0: 
            print('setting the Fourier modes on the grid and computing the shell-averaged modes') ; sys.stdout.flush()
        
        kz     = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float32')
        k_3D   = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float64')
        k_3D_2 = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float32')
        k_1D   = sharing_array_throw_MPI((int(Ns/2 -1),),comm,'float64')
        
        if intrarank == 0:
            ref           = np.arange(Ns)
            norm_1d       = np.concatenate((ref[ref<Ns/2] *k_F,(ref[ref >= Ns/2] - Ns)*k_F))
            kz_           = np.array([[norm_1d,]*Ns]*Ns)
            k_3D[:,:,:]   = fast_norm(kz_.transpose(2,1,0),kz_.transpose(0,2,1),kz_) #sqrt(kx^2 + ky^2 + kz^2)
            kz[:,:,:]     = kz_.astype(np.float32) ; del kz_
            k_3D_2[:,:,:] = (array_times_array(k_3D,k_3D)).astype(np.float32)
            
        SA_trick = shellaveraging_trick_function(Par,k_3D) # only rank 0 receives
        
        if rank == 0:
            k_1D[:] = shell_averaging(SA_trick,k_3D)
            np.savez(Fourier_file,k_3D,k_3D_2,k_1D)
            if Par['verbose']: print('Fourier modes files has been saved in' + Fourier_file + '.npz') ; sys.stdout.flush()
        
        comm.Barrier()
    if mode == 0: return k_3D,k_3D_2,k_1D
    if mode == 1: return k_3D,k_1D
    if mode == 2: return k_3D_2


def shellaveraging_trick_function(Par,k_3D): 
    '''
    smart way of storing position of each shell. It allows to perform a faster shell-averaging procedure
    '''
    Ns = Par['N_sample'] ; L  = Par['L'] ; k_F = 2*np.pi/L ; SA_trick = []
    
    shell_avg_file = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'shellaveraging_trick_arrays_Pk_L%s_N_sample%i'%(L,Ns)
    
    if os.path.exists(shell_avg_file + '.npz'):
        if Par['verbose']: 
            print('load shell-averaging trick arrays from', shell_avg_file + '.npz') ; sys.stdout.flush()
        if rank == 0:
            file = np.load(shell_avg_file + '.npz',allow_pickle=True)
            SA_trick = file['arr_0']

    else:    
        if Par['verbose'] and rank == 0:
            print('computing the shell-averaging trick arrays ' + (size != 1)* 'in MPI') ; sys.stdout.flush()
        
        k_to_share = np.arange(1,int(Ns/2)) #for each shell up to k_N-1, without the DC mode
        k_ = spliting_one_array(k_to_share,size)[rank]            
        if rank == 0:
            try: os.makedirs(Par['output_dir_project'] + '/temporary')
            except: 0
        
        comm.Barrier()
        for k in k_:    
            lower_bound      = k*k_F-k_F/2.
            upper_bound      = k*k_F+k_F/2.
            index = np.where(((k_3D > lower_bound) * (k_3D < upper_bound)))
            SA_trick.append(index)

        np.savez(Par['output_dir_project'] + '/temporary/'+ str(rank) , SA_trick)
        comm.Barrier()
        
        if rank == 0 :
            SA_trick = []
            
            for r in range(size):
                file = np.load(Par['output_dir_project'] + '/temporary/'+ str(r) + '.npz',allow_pickle=True)
                [SA_trick.append(file['arr_0'][i]) for i in range(len(file['arr_0']))]
            from shutil import rmtree
            rmtree(Par['output_dir_project'] + '/temporary',ignore_errors=True)
            np.savez(shell_avg_file,SA_trick)
            if Par['verbose']: print('shell-averaging trick array has been saved in ' + shell_avg_file + '.npz') ; sys.stdout.flush()
            file = np.load(shell_avg_file + '.npz',allow_pickle=True)
            SA_trick = file['arr_0']
    return SA_trick

def shell_averaging(SA_trick,table3d):
    '''
    performs the shell-average of a given 3D table in Fourier space
    '''
    table_1D = np.zeros(int(table3d.shape[0]/2-1))
    for k in range(int(table3d.shape[0]/2-1)):
        table_1D[k] = np.average(table3d[SA_trick[k,0],SA_trick[k,1],SA_trick[k,2]])
    return table_1D

def fast_shell_averaging(Par,Pk3D):
    SA_trick = shellaveraging_trick_function(Par,None)
    pk1d  = shell_averaging(SA_trick,Pk3D)
    return pk1d

def Mehler(Par,density_field):
    '''
    compute the non-linear transformation on a Gaussian field able to target the non-Gaussian one, using the Mehler expansion to quantify how the two-point correlation functions are impacted
    '''
    if Par['verbose'] and rank == 0:
        print('\n_______________________________ WORKING ON DENSITY PDF _______________________________') ; sys.stdout.flush()
    
    from math import factorial
    def draw_x_nu_from_x_delta(x_deltap):
        '''
        since the Gaussian field in centered and reduced, nu is also defined <-1. So here we extend the delta array below -1
        '''
        x_nup = np.array([-1],dtype=float)
        while min(x_nup)>-10:
            new_value = x_nup[0]-(x_deltap[-1]-x_deltap[-2])
            x_nup = np.insert(x_nup,0,new_value)
        x_nup = np.insert(x_deltap,0,x_nup[:-1])
        return x_nup
    def variance_from_PDF(PDF):
        '''
        computes the variance directly from the PDF shape
        '''
        x = PDF[0] ; PDFx = PDF[1]
        return np.trapz(x**2*PDFx,x) - np.trapz(x*PDFx,x)**2
    def gen_PDF_G(x,sigma,mu): 
        '''
        generate a simple Gaussian distribution of rms sigma, mean mu
        '''
        return (1/(sigma*np.sqrt(2*np.pi)))  * np.exp(-((x-mu)**2)/(2*sigma**2));
    def lambda_func(xi_G,L_of_x,PDF_nu,x,maxMehl):
        '''
        computes the lambda function mapping the 2pcf of the Gaussian field to the non Gaussian one
        '''
        xi = np.zeros(len(xi_G))
        for i in range(maxMehl): xi += (1/factorial(i)) * cn_wt_fact(i,L_of_x,PDF_nu,x)**2  * (xi_G)**i
        return xi
    def cn_wt_fact(n,L_of_x,PDF_G,x):
        '''
        computes the coefficients of the Mehler expansion (without 1/n!)
        '''
        integrand  = L_of_x * Herm_polys(n,x) * PDF_G       
        return np.trapz(integrand,x);
    def cn(n,L_of_x,PDF_G,x):
        '''
        computes the coefficients of the Mehler expansion
        '''
        integrand  = L_of_x * Herm_polys(n,x) * PDF_G 
        return       (1/factorial(n)) * np.trapz(integrand,x);
    def Herm_polys(coef,X):
        '''
        return the probabilistic hermite polynomials
        '''
        coef_array = np.append(np.zeros(coef),1)
        return np.polynomial.hermite_e.hermeval(X,coef_array);
    
    PDF_map = {}
    
    if not Par['PDF_d_file'] == 'gaussian':
        PDF_map['x_nu']    = draw_x_nu_from_x_delta(density_field['PDF_delta'][0])
        PDF_map['var_PDF'] = variance_from_PDF(density_field['PDF_delta'])
        PDF_nu             = gen_PDF_G(PDF_map['x_nu'],1,0)
        C_nu               = np.cumsum(PDF_nu)
        C_nu              /= C_nu   [-1]
        C_delta            = np.cumsum(density_field['PDF_delta'][1])
        C_delta           /= C_delta[-1]
        PDF_map['NL_map']  = np.interp(C_nu,C_delta,density_field['PDF_delta'][0])
        reduce_range       = (PDF_map['x_nu']>-10) * (PDF_map['x_nu']<10)
        PDF_map['Xi_G_template']  = np.linspace(-1.,1.,1000)
        PDF_map['Xi_NG_template'] = lambda_func(PDF_map['Xi_G_template'],PDF_map['NL_map'][reduce_range],PDF_nu[reduce_range],PDF_map['x_nu'][reduce_range],80)
        
        if Par['debug'] and rank == 0: 
            Mehler_coefs = [cn(i,PDF_map['NL_map'][reduce_range],PDF_nu[reduce_range],PDF_map['x_nu'][reduce_range]) for i in range(10)] 
            np.savetxt(Par['output_dir_project'] + '/debug_files/n_cn_Mehler_coefs.txt',np.transpose(np.vstack((np.arange(10),Mehler_coefs))))
            
        if Par['verbose'] and rank == 0: 
            print('the variance defined by the provided PDF is',PDF_map['var_PDF'])
            print('given the integral precision, the Melher expansion returns a maximum value for the 2pcf transformation of', np.amax(PDF_map['Xi_NG_template']))
            print('the corresponding error is', round(100*(np.amax(PDF_map['Xi_NG_template'])/PDF_map['var_PDF']-1),3) ,'percent')
            sys.stdout.flush()
            
    elif Par['PDF_d_file'] == 'gaussian':
        PDF_map['var_PDF'] = 0.01
        if Par['verbose'] and rank == 0: 
            print('you choose a gaussian density PDF, in order for delta to be > -1, the targeted variance will be equal to',PDF_map['var_PDF'])
            sys.stdout.flush()
    comm.Barrier()
    return PDF_map

def matching_Pk_to_PDF(density_field,PDF_map,Par):
    '''
    fitering of a 1D power spectrum following exp(-(kR)^i1) in order for the corresponding variance to match var_tgt (the one obtained from the PDF)
    '''
    if Par['verbose'] and rank == 0:
        print('\n_______________________________ WORKING ON DENSITY PK ________________________________')
        sys.stdout.flush()
    def balayage_filtering_1D(k,Pk,i1,var_tgt):
        '''
        successively filtering Pk in order to target the variance
        '''
        R_ = np.linspace(0,3,10000) ; var = []
        for R in R_: var.append(4*np.pi*np.trapz(k**2 * Pk *np.exp(-(k*R)**i1),k))
        filt_radius = np.interp(0,(np.array(var)-var_tgt)[::-1],R_[::-1])
        if filt_radius == R_[0] or filt_radius == R_[-1] : raise Exception('bad filtering process')
        Pk_filt = Pk*np.exp(-(k*filt_radius)**i1)
        filtering = np.exp(-(k*filt_radius)**i1)
        return Pk_filt,filtering

    if not 'Pk_3D_dd_alias' in density_field :
        if Par['verbose'] and rank == 0: 
            print('fitering the 1D power spectrum following exp(-(kR)^i1) in order for the corresponding variance to match', PDF_map['var_PDF'])
            sys.stdout.flush()
        Pk_filtered,_ = balayage_filtering_1D(density_field['Pk_1D_dd'][0],density_field['Pk_1D_dd'][1],Par['i1'],PDF_map['var_PDF'])
        density_field['Pk_1D_dd_filtered'] = np.vstack((density_field['Pk_1D_dd'][0],Pk_filtered))
        if Par['debug'] and rank == 0: np.savetxt(Par['output_dir_project'] + '/debug_files/k_pk_filtered_with_i1',np.transpose(density_field['Pk_1D_dd_filtered']))
    
    comm.Barrier()
    return density_field


def aliasing(density_field,Par,k_1D,k_3D):
    '''
    computes the 3D, aliased version (following an aliasing_order parameter) of a 1D power spectrum
    '''
    aliasOrd = Par['aliasing_order'] ; L = Par['L'] ; Ns = Par['N_sample'] ; a = L/Ns ; k_N = np.pi/a ; k_F = 2*np.pi/L
    
    if not 'Pk_3D_dd_alias' in density_field:
        if aliasOrd == 0: 
            if Par['verbose'] and rank == 0: 
                print('Since aliasing_order = 0, the target power spectrum is simply interpolated on 3D Fourier modes')
                sys.stdout.flush()
            density_field['Pk_3D_dd_alias'] = np.exp(np.interp(np.log(k_3D),np.log(density_field['Pk_1D_dd_filtered'][0]),np.log(density_field['Pk_1D_dd_filtered'][1]),right=1e-10))
        
        else:
            from itertools import product as iterProd
            import time
            
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
            
            if Par['verbose'] and rank == 0: 
                print('start aliasing the theoretical power spectrum' + (size != 1)* ' in MPI') ; sys.stdout.flush()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                logk_1D       = np.log(density_field['Pk_1D_dd_filtered'][0])
                logPk_theo_1D = np.log(density_field['Pk_1D_dd_filtered'][1])

            Pk_3d_part    = np.zeros((Ns,Ns,Ns))    
            logk_alias    = np.zeros_like(Pk_3d_part)
            
            start_time  = time.time()
            total_loops = (2*aliasOrd+1)**3 / size

            i=1
            for n in range(len(my_ns1)):
                n1 = my_ns1[n] ; n2 = my_ns2[n] ; n3 = my_ns3[n]
                logk_alias[:,:,:] = log_kalias(n1,n2,n3,k_N,aliasOrd,kx,ky,kz)
                logk_alias[:,:,:] = np.interp(logk_alias,logk_1D,logPk_theo_1D)
                Pk_3d_part[:,:,:] = exppara(Pk_3d_part,logk_alias)

                time_since_start  = (time.time() - start_time)
                rmn = (time_since_start * total_loops/ i - time_since_start)/60
                percent = 100*i/total_loops
                if Par['verbose'] and rank == 0: 
                    sys.stdout.write("\restimated remaining time: %.1f minutes, %.0f %%" %(rmn,percent)); sys.stdout.flush()
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

def matching_Pk_3D_to_PDF(density_field,PDF_map,Par,k_3D):
    '''
    fitering of a 3D power spectrum following exp(-(grid_k R)^i1) in order for the corresponding variance to match the one obtained from the PDF
    '''
    k_F = 2*np.pi/ Par['L']
    if rank == 0:
        if np.sum(density_field['Pk_3D_dd_alias'])*k_F**3 > PDF_map['var_PDF']:
            density_field['Pk_3D_dd_alias'],_ = balayage_filtering_3D(Par['i1'],[0.2,0.5],density_field['Pk_3D_dd_alias'],PDF_map['var_PDF'],k_3D,k_F,verbose=Par['verbose'])
    return density_field

def balayage_filtering_3D(index,start_range,pktofilt,target_var,norm_3D,k_F,verbose=False):
    var_obtained = np.sum(pktofilt)*k_F**3
    if var_obtained>target_var:
        if verbose: 
            print('filtering the 3D power spectrum to fit the right variance: from',var_obtained,'down to',target_var,'using the filtering parameter',index) ; sys.stdout.flush()
        type_filt = -1
    elif var_obtained<target_var:    
        if verbose: 
            print('powering the 3D power spectrum to fit the right variance: from',var_obtained,'up to',target_var,'using the filtering parameter',index) ; sys.stdout.flush()
        type_filt = 1
    lerangeR = start_range
    ref = lerangeR[0]
    while((lerangeR[1]-lerangeR[0])>0.005):
        var=[]
        for R in lerangeR:
            var.append(np.sum(filtering_operation(pktofilt,type_filt,norm_3D,R,index))*k_F**3)
        if   var_obtained>target_var:  interpol = np.interp(0,(np.array(var)-target_var)[::-1],lerangeR[::-1])
        elif var_obtained<target_var:  interpol = np.interp(0,(np.array(var)-target_var),lerangeR)

        if interpol in lerangeR: pass
        elif (interpol not in lerangeR) and (lerangeR[1]-lerangeR[0])<0.3: ref = ref - 0.75*ref
        else: ref = ref - 0.6*ref
        lerangeR = [interpol-ref,interpol+ref]
        if lerangeR[0]<0: lerangeR[0]=0.
    final = filtering_operation(pktofilt,type_filt,norm_3D,interpol,index)
    if verbose: 
        print('new variance is',np.sum(final)*k_F**3,'and should be',target_var) ; sys.stdout.flush()
    return final,interpol

def Pk_nu_compute(density_field,PDF_map,Par,k_3D):
    Ns = Par['N_sample'] ; k_F = 2*np.pi/Par['L']
    
    if rank == 0 and not Par['PDF_d_file'] == 'gaussian':
        import pyfftw
        from multiprocessing import cpu_count
        
        pyfftw.config.NUM_THREADS = cpu_count()

        twopcf_delta = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')
        Pk_nu        = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            twopcf_delta[:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(density_field['Pk_3D_dd_alias'],axes=(0,1,2))*(Ns*k_F)**3
            twopcf_nu           = np.interp(twopcf_delta,PDF_map['Xi_NG_template'],PDF_map['Xi_G_template']) ; del twopcf_delta
            Pk_nu[:,:,:]        = pyfftw.interfaces.numpy_fft.fftn(twopcf_nu,axes=(0,1,2))/(Ns*k_F)**3       ; del twopcf_nu

        if Par['verbose']: 
            print('percentage of negatives in Pk_nu_3D:',100*(len(Pk_nu[Pk_nu<0])/Pk_nu.size),'percents','with variance',np.sum(Pk_nu)*k_F**3,'\n')
            sys.stdout.flush()

        #clipping method
        if Par['verbose']: 
            print('applying now the clipping method') ; sys.stdout.flush()
        Pk_nu[Pk_nu<0] = 0.

        if np.sum(Pk_nu)*k_F**3 > 1.: Pk_nu,_ = balayage_filtering_3D(Par['i2'],[0.5,3],Pk_nu,1.,k_3D,k_F,verbose=Par['verbose'])
        
        del density_field['Pk_3D_dd_alias']
        density_field['Pk_nu'] = Pk_nu
    return density_field
    
def get_Pk_tt_3d(velocity_field,k_3D,Par):
    if Par['verbose'] and rank == 0:
        print('\n_______________________________ WORKING ON VELOCITY PK _______________________________') ; sys.stdout.flush()
        
    k_F = 2*np.pi/Par['L']
    
    if rank == 0 and Par['velocity']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            velocity_field['Pk_3D_tt'] = np.exp(np.interp(np.log(k_3D),np.log(velocity_field['Pk_1D_tt'][0]),np.log(velocity_field['Pk_1D_tt'][1]),right=1e-10))
        #Pk_tt_3d *= np.exp(-(k_3D*filtering_radius)**index)
        if Par['verbose']: 
            print('\nthe target Pk_tt is interpolated on 3D modes. The corresponding target variance is',np.sum(velocity_field['Pk_3D_tt'])*k_F**3,'\n') ; sys.stdout.flush()
    return velocity_field

def save_ini_files(density_field,velocity_field,Par,PDF_map):
    Ns = Par['N_sample'] ; k_F = 2.*np.pi/Par['L']
    
    if rank == 0:
        output_file = Par['output_dir_project'] + '/ini_files/ini_file'
        
        scalar_density  = Ns**3/np.sum(density_field['Pk_nu'])
        scalar_velocity = Ns**3 * k_F**3
        
        if Par['velocity']:
            byprod_pk_density,byprod_pk_velocity = sqrt_of_arrays_times_scalars(density_field['Pk_nu'],scalar_density,velocity_field['Pk_3D_tt'],scalar_velocity)
        else:
            byprod_pk_density = sqrt_of_array_times_scalar(density_field['Pk_nu'],scalar_density)
            byprod_pk_velocity = np.array([0.])
        
        if Par['verbose']: 
            print('saving the initilisation files used by COVMOS_sim.py in',output_file) ; sys.stdout.flush()
        np.savez(output_file,PDF_map['x_nu'],PDF_map['NL_map'],byprod_pk_density,byprod_pk_velocity.astype(np.float32)) 
    return

def compute_2PS_predictions(Par,density_field,PDF_map,k_1D):
    Ns = Par['N_sample'] ; L = Par['L'] ; k_F = 2.*np.pi/L ; a = L/Ns
    
    if rank == 0 and (Par['compute_Pk_prediction'] or Par['compute_2pcf_prediction']):
        
        if Par['verbose']:
            print('\n__________________________ COMPUTING TWO-P STAT PREDICTIONS __________________________') ; sys.stdout.flush()
        
        import pyfftw
        import scipy.special as SS
        from multiprocessing import cpu_count
        
        pyfftw.config.NUM_THREADS = cpu_count()

        xi_nu     = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')
        Pk_target = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')
        
        if Par['verbose']: 
            print('compute the 3D grid, output power spectrum') ; sys.stdout.flush()
        if not Par['PDF_d_file'] == 'gaussian':    
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xi_nu[:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(density_field['Pk_nu'],axes=(0,1,2))*(Ns*k_F)**3
                xi_delta     = np.interp(xi_nu,PDF_map['Xi_G_template'],PDF_map['Xi_NG_template'])          ; del xi_nu
                Pk_target[:,:,:] = pyfftw.interfaces.numpy_fft.fftn(xi_delta,axes=(0,1,2))/(Ns*k_F)**3
        else: 
            Pk_target[:,:,:] = density_field['Pk_3D_dd_alias']                                        ; del xi_nu
            xi_delta         = pyfftw.interfaces.numpy_fft.ifftn(Pk_target,axes=(0,1,2))*(Ns*k_F)**3

        if not Par['compute_2pcf_prediction']: del xi_delta

        if Par['compute_Pk_prediction']: Pk_target_grid_1d = fast_shell_averaging(Par,Pk_target)
        
        ref     = np.arange(Ns)
        norm_1d = np.concatenate((ref[ref<Ns/2] *k_F,(ref[ref >= Ns/2] - Ns)*k_F))
        kz      = np.array([[norm_1d,]*Ns]*Ns)
                    
        if Par['verbose']: 
            print('computing W0, the window function related to the',Par['assign_scheme'],'assignment scheme') ; sys.stdout.flush()
            
        kx_ademi = array_times_scalar(kz.transpose(2,1,0),a/2)
        ky_ademi = array_times_scalar(kz.transpose(0,2,1),a/2)
        kz_ademi = array_times_scalar(kz,a/2)                                                        ; del kz
        
        W0 = array_times_array_times_array(SS.spherical_jn(0,kx_ademi),SS.spherical_jn(0,ky_ademi),SS.spherical_jn(0,kz_ademi))
        del kx_ademi, ky_ademi, kz_ademi

        if Par['assign_scheme'] == 'trilinear': W0 = array_times_array(W0,W0)
        
        if Par['verbose']: 
            print('applying it to the grid power spectrum to obtain the 3D catalogue power spectrum') ; sys.stdout.flush()
            
        Pkpoisson3D = array_times_arraypow2(Pk_target,abs(W0))                                       ; del W0,Pk_target
        
        if Par['compute_Pk_prediction']: 
            if Par['verbose']: 
                print('shell-average it') ; sys.stdout.flush()
            Pkpoisson1D = fast_shell_averaging(Par,Pkpoisson3D)
            if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/grid_k_Pk_prediction.txt')
            np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/grid_k_Pk_prediction.txt',np.transpose(np.vstack((k_1D,Pk_target_grid_1d))))
            if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/catalogue_k_Pk_prediction.txt')
            np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/catalogue_k_Pk_prediction.txt',np.transpose(np.vstack((k_1D,Pkpoisson1D))))

            Pk_target = np.exp(np.interp(np.log(k_1D),np.log(density_field['Pk_1D_dd'][0]),np.log(density_field['Pk_1D_dd'][1]),right=1e-10))
            if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/target_k_Pk.txt')
            np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/target_k_Pk.txt',np.transpose(np.vstack((k_1D,Pk_target))))
    
        
        
    if Par['compute_2pcf_prediction']:
        if Par['verbose'] and rank == 0: 
            print('\nnow working on two-point correlation functions') ; sys.stdout.flush()
        
        if rank == 0: 
            twopcfP        = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                twopcfP[:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(Pkpoisson3D,axes=(0,1,2))*(Ns*k_F)**3 ; del Pkpoisson3D
        
        SA_SmallScales = [] ; SA_LargeScales = []
        
        shell_avg_file_2pcf = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'shellaveraging_trick_arrays_2PCF_L%s_N_sample%i'%(L,Ns)
        
        if os.path.exists(shell_avg_file_2pcf + '.npz'):
            if Par['verbose']: 
                print('load shell-averaging trick arrays for 2PCF from', shell_avg_file_2pcf + '.npz') ; sys.stdout.flush()
            if rank == 0:
                file = np.load(shell_avg_file_2pcf + '.npz',allow_pickle=True)
                SA_SmallScales = file['arr_0'] 
                SA_LargeScales = file['arr_1']   
                
        else:
            if Par['verbose'] and rank == 0: print('computing shell averaging files for 2pcf prediction' + (size != 1)* ' in MPI') ; sys.stdout.flush()
            
            normpos   = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float64')
            unique    = sharing_array_throw_MPI((100,),intracomm,'float64')
        
            if intrarank == 0:
                zzz    = np.zeros((Ns,Ns,Ns))
                zzz[:] = np.linspace(0,L,Ns)
                normpos[:,:,:] = fast_norm(np.transpose(zzz,(2,1,0)),np.transpose(zzz,(1,2,0)),zzz)    ; del zzz
                unique[:]      = np.unique(normpos)[:100]
            
            r_to_share = np.arange(100)
            r_ = spliting_one_array(r_to_share,size)[rank]  
            
            if rank == 0:
                try: os.makedirs(Par['output_dir_project'] + '/temporary')
                except: 0
            
            comm.Barrier()
            #finding and stocking the array indices corresponding to the first 100 values in normpos (only for small scales)
            for r in r_:
                index = np.where(normpos == unique[r])
                SA_SmallScales.append(index) 
            
            np.savez(Par['output_dir_project'] + '/temporary/smallscales'+ str(rank) , SA_SmallScales)
            #finding and stocking the array indices corresponding to shells of width ... (for larger scales)
            width_shell = 2 # in Mpc/h
            centre_shells_tot  = np.arange(0,int(L))+width_shell/2
            centre_shells_part = spliting_one_array(centre_shells_tot,size)[rank]
            
            for centre in centre_shells_part:
                index = np.where(((centre-width_shell/2)<normpos) * ((centre+width_shell/2)>normpos))
                np.savez(Par['output_dir_project'] + '/temporary/largescales'+ str(centre) , index)
            
            comm.Barrier()
            
            if rank == 0 :
                SA_SmallScales = []
                SA_LargeScales = []
                
                for r in range(size):
                    file = np.load(Par['output_dir_project'] + '/temporary/smallscales'+ str(r) + '.npz',allow_pickle=True)
                    [SA_SmallScales.append(file['arr_0'][i]) for i in range(len(file['arr_0']))]
                
                for centre in centre_shells_tot:
                    file = np.load(Par['output_dir_project'] + '/temporary/largescales'+ str(centre) + '.npz')
                    SA_LargeScales.append(file['arr_0']) 
                    #[SA_LargeScales.append([file['arr_0'][i]]) for i in range(len(file['arr_0']))]
                    
                from shutil import rmtree
                rmtree(Par['output_dir_project'] + '/temporary',ignore_errors=True)
                np.savez(shell_avg_file_2pcf,SA_SmallScales,SA_LargeScales)    
                if Par['verbose']: print('shell-averaging trick arrays for 2pcf have been saved in' + shell_avg_file_2pcf + '.npz');sys.stdout.flush()
                file = np.load(shell_avg_file_2pcf + '.npz',allow_pickle=True)
                SA_SmallScales = file['arr_0'] ; SA_LargeScales = file['arr_1']
            
        if rank == 0:
            
            def twoPCF_Shell_Averaging(SA_SmallScales,SA_LargeScales,normpos,twoPCF_3D):
                #shell averaging the 2pcf using the two methods (small scales and large scales)
                SAdist   = np.zeros(len(SA_SmallScales[:,0]))
                SAdist2  = np.zeros(len(SA_LargeScales[:,0]))
                SA2pcfP  = np.zeros(len(SA_SmallScales[:,0]))
                SA2pcf2P = np.zeros(len(SA_LargeScales[:,0]))

                for i in range(len(SA_SmallScales[:,0])):
                    SAdist[i]  = np.mean(normpos[SA_SmallScales[i,0],SA_SmallScales[i,1],SA_SmallScales[i,2]])
                    SA2pcfP[i] = np.mean(twoPCF_3D[SA_SmallScales[i,0],SA_SmallScales[i,1],SA_SmallScales[i,2]])
                for i in range(len(SA_LargeScales[:,0])):
                    SAdist2[i] = np.mean(normpos[SA_LargeScales[i,0],SA_LargeScales[i,1],SA_LargeScales[i,2]])
                    SA2pcf2P[i]= np.mean(twoPCF_3D[SA_LargeScales[i,0],SA_LargeScales[i,1],SA_LargeScales[i,2]])

                #For small R, one keeps the exact values, then at larger scales the averaged 2pcf in shells     
                R    = np.append(SAdist ,SAdist2 [SAdist2>SAdist[-1]])
                corr = np.append(SA2pcfP,SA2pcf2P[SAdist2>SAdist[-1]])
                return R,corr

            R_P,corr_P = twoPCF_Shell_Averaging(SA_SmallScales,SA_LargeScales,normpos,twopcfP)
            R_G,corr_G = twoPCF_Shell_Averaging(SA_SmallScales,SA_LargeScales,normpos,xi_delta)

            if Par['compute_2pcf_prediction']:
                if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/grid_r_Xi_prediction.txt')
                np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/grid_r_Xi_prediction.txt',np.transpose(np.vstack((R_G,corr_G))))
                if Par['verbose']: print('saving',Par['output_dir_project'] + '/TwoPointStat_predictions/catalogue_r_Xi_prediction.txt')
                np.savetxt(Par['output_dir_project'] + '/TwoPointStat_predictions/catalogue_r_Xi_prediction.txt',np.transpose(np.vstack((R_P,corr_P))))

    return