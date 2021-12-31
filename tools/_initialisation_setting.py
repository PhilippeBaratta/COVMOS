import numpy as np
from ast import literal_eval
from configparser import ConfigParser
from os import path,makedirs

from tools._numba_functions import *
from tools._shell_averaging import *
from tools._networking import *

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

intracomm = from_globalcomm_to_intranode_comm()
intrarank = intracomm.rank
    
def read_parameters(inifile):
    '''
    read, test and complete all the parameters of the .ini file provided by the user
    It returns a dictionnary of the COVMOS parameters
    '''
    
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
        print('\n________________________ READING/COMPUTING STATISTICAL INPUTS ________________________\n',flush=True)
        
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
    if not path.exists(Par['Pk_dd_file']) and not Par['Pk_dd_file'] == '':
        raise Exception('Pk_dd_file not found:',Par['Pk_dd_file'])
    if path.exists(Par['Pk_dd_file']):
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
    if not path.exists(Par['PDF_d_file']) and not Par['PDF_d_file'] == 'gaussian' :
        raise Exception('PDF_d_file not found:',Par['PDF_d_file'], 'if you want a gaussian PDF, type gaussian')
    if path.exists(Par['PDF_d_file']):
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
    if Par['filtering_parameter'] == 'Default': 
        if Par['PDF_d_file'] == 'gaussian': 
            Par['i1'] = 2 ; Par['i2'] = None
        else:
            Par['i1'],Par['i2'] = filtering_i1_i2( Par['redshift'] )
    else:
        Par['i1'] = Par['filtering_parameter']
        _,Par['i2'] = filtering_i1_i2( Par['redshift'] )
    
    ###########################################################
    Par['velocity'] = config.getboolean('OUTPUTS', 'velocity')# need it now
    ###########################################################
    
    if Par['velocity']:
        Par['Pk_tt_file'] = config.get(var_type, 'Pk_tt_file')
        if Par['Pk_tt_file'] == '' or Par['Pk_tt_file'] == "''": Par['Pk_tt_file'] = ''
        if not path.exists(Par['Pk_tt_file']) and not Par['Pk_tt_file'] == '':
            raise Exception('Pk_tt_file not found:',Par['Pk_tt_file'])
        if path.exists(Par['Pk_tt_file']):
            if rank == 0:
                try: 
                    kk,_ = np.loadtxt(Par['Pk_tt_file'],unpack=1)
                    if kk[0] > 2*np.pi/Par['L'] : raise Exception('the kmin in',Par['Pk_tt_file'],'is too high (> 2pi/L), you should extrapolate it down to lower modes')
                except: raise Exception('the ascii file associated to Pk_tt_file is not composed of two columns (k in h/Mpc and Pk in [Mpc/h]^3)')
    
        
    if Par['Pk_dd_file'] == '' or Par['velocity'] * (Par['Pk_tt_file'] == '') :
        try:
            class_dic_param = literal_eval(config.get(var_type, 'classy_dict'))
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
        Par['project_name'] = 'COVMOS_' + 'Ns' + str(Par['N_sample']) + '_L' + str(Par['L']) + '_z' + str(Par['redshift']) + '_rho' + str(Par['rho_0']) + '_Om' + str(Par['Omega_m']) + '_' + Par['assign_scheme'] + '_scheme' + (not Par['Pk_dd_file'][-4:] == '.npy') * ('_aliasOrd' + str(Par['aliasing_order'])) + Par['velocity'] * ('_alpha' + str(Par['alpha']) + '_Vrms' + str(Par['targeted_rms'])) + Par['fixed_Rdm_seed'] * '_fixed_Rdm_seed'
    
    Par['output_dir_project'] = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + Par['project_name']
    
    if Par['verbose'] and rank == 0 : print('the ini file has been correctly read',flush=True)
    
    comm.Barrier()
    return Par

def generate_output_repertories(Par):
    '''
    generate the diverse output repertories and logfile from COVMOS_ini.py
    '''
    if rank == 0:
        if Par['verbose']: print('generating the various output repertories in', Par['output_dir_project'],flush=True)
        makedirs(Par['output_dir_project'],exist_ok=True)
        makedirs(Par['output_dir_project'] + '/ini_files',exist_ok=True)
        if Par['compute_Pk_prediction'] or Par['compute_2pcf_prediction']: makedirs(Par['output_dir_project'] + '/TwoPointStat_predictions',exist_ok=True)
        if Par['compute_Pk_prediction'] or Par['debug']:
            shell_avg_file = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'shellaveraging_trick_arrays_Pk_L%s_Ns%i'%(L,Ns)
            makedirs(shell_avg_file,exist_ok=True)
        if Par['compute_2pcf_prediction'] or Par['debug']:
            shell_avg_file_2pcf = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'shellaveraging_trick_arrays_2PCF_L%s_Ns%i'%(L,Ns)
            makedirs(shell_avg_file_2pcf,exist_ok=True)
        if Par['debug']: makedirs(Par['output_dir_project'] + '/debug_files',exist_ok=True)
            
        if Par['verbose']: print('saving all the input parameters in', Par['output_dir_project'] + '/setting.log',flush=True)
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
                if prescription == 'Default': print('computing the class power spectrum (for the density field) following the dict provided by the user :', Par['classy_dict'],flush=True)
                else: print('computing the class power spectrum following with a linear prescription for the velocity field :', Par['classy_dict'],flush=True)
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
    
    if Par['Pk_dd_file'][-4:] == '.npy':
        if rank == 0: density_field['Pk_3D_dd_alias'] = np.load(Par['Pk_dd_file'])
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
            
    if Par['verbose'] and rank == 0 : print('the statistical targets have been successfully read/computed\n',flush=True)
    
    comm.Barrier()
    return density_field,velocity_field


def Fouriermodes(Par,mode=0):
    '''
    setting the Fourier modes on the grid and computing the shell-averaged modes (if compute_Pk_prediction == True or debug == True)
    it returns k_3D in float64, k_3D_2 in float32 and k_1D in float64
    mode = 0 -> all output are returned (k_3D,k_3D_2,k_1D)
    mode = 1 -> output 0 and 2 are returned (k_3D,k_1D)
    mode = 2 -> output 1 are returned (k_3D_2)
    '''
    Ns = Par['N_sample'] ; L  = Par['L'] ; k_F = 2*np.pi/L
    Fourier_file = Par['output_dir'] + (not Par['output_dir'][-1]=='/')*'/' + 'Fouriermodes_L%s_Ns%i'%(L,Ns)
    force_compute_k_1D = False
    k_3D = 0 ; k_3D_2 = 0 ; k_1D = 0
    
    if path.exists(Fourier_file + '.npz'):
        if Par['verbose'] and rank == 0: print('load Fourier modes from', Fourier_file + '.npz',flush=True)
            
        file = np.load(Fourier_file + '.npz')
        if mode == 0 or mode == 2:
            k_3D_2 = file['arr_1']
        if mode == 0 or mode == 1:
            if rank == 0: k_3D = file['arr_0']
            k_1D = file['arr_2']
            
            if len(k_1D) == 1 and (Par['compute_Pk_prediction'] or Par['debug']):
                if Par['verbose'] and rank == 0: print('the existing file ', Fourier_file, '.npz has no shell-averaged modes (k_1D) due to the chosen parameters in a previous project, lets compute it now',flush=True)
                force_compute_k_1D = True
                
    if not path.exists(Fourier_file + '.npz') or force_compute_k_1D:
        if Par['verbose'] and rank == 0:  print('setting the Fourier modes on the grid',flush=True)
        
        k_3D = sharing_array_throw_MPI((Ns,Ns,Ns),intracomm,'float64')
        
        if intrarank == 0 or rank == 0 :
            ref         = np.arange(Ns)
            norm_1d     = np.concatenate((ref[ref<Ns/2] *k_F,(ref[ref >= Ns/2] - Ns)*k_F))
            kz          = np.array([[norm_1d,]*Ns]*Ns)
            k_3D[:,:,:] = fast_norm(kz.transpose(2,1,0),kz.transpose(0,2,1),kz) #sqrt(kx^2 + ky^2 + kz^2)
            kz          = kz.astype(np.float32)
            k_3D_2      = (array_times_array(k_3D,k_3D)).astype(np.float32)
        
        if Par['compute_Pk_prediction'] or Par['debug']:
            SA_trick = shellaveraging_trick_function(Par,k_3D) # only rank 0 receives
            
            if rank == 0:
                k_1D = shell_averaging(SA_trick,k_3D)
                np.savez(Fourier_file,k_3D,k_3D_2,k_1D)
        else:
            if rank == 0:
                k_1D = [0]
                np.savez(Fourier_file,k_3D,k_3D_2,k_1D)
                
        if Par['verbose'] and rank == 0 : print('Fourier modes files has been saved in ' + Fourier_file + '.npz',flush=True)
        
        comm.Barrier()
        
    if mode == 0: return k_3D,k_3D_2,k_1D
    if mode == 1: return k_3D,k_1D
    if mode == 2: return k_3D_2
    

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
        
        if Par['verbose']: print('saving the initilisation files used by COVMOS_sim.py in',output_file,flush=True)
            
        np.savez(output_file,PDF_map['x_nu'],PDF_map['NL_map'],byprod_pk_density,byprod_pk_velocity.astype(np.float32)) 
    return