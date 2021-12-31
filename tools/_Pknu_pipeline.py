import numpy as np
import os
import sys
import warnings
import pyfftw
from multiprocessing import cpu_count

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from tools._numba_functions import *
from tools._networking import *
from tools._shell_averaging import *

intracomm = from_globalcomm_to_intranode_comm()
intrarank = intracomm.rank

def Mehler(Par,density_field):
    '''
    compute the non-linear transformation on a Gaussian field able to target the non-Gaussian one, using the Mehler expansion to quantify how the two-point correlation functions are impacted
    '''
    if Par['verbose'] and rank == 0: print('\n_______________________________ WORKING ON DENSITY PDF _______________________________\n',flush=True)
    
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
            print('the corresponding error is', round(100*(np.amax(PDF_map['Xi_NG_template'])/PDF_map['var_PDF']-1),3) ,'percent',flush=True)
            
    elif Par['PDF_d_file'] == 'gaussian':
        PDF_map['var_PDF'] = 0.01
        PDF_map['NL_map']  = [None]
        PDF_map['x_nu']    = [None]
        
        if Par['verbose'] and rank == 0: print('you choose a gaussian density PDF, in order for delta to be > -1, the targeted variance will be equal to',PDF_map['var_PDF'],flush=True)
        
    comm.Barrier()
    return PDF_map

def matching_Pk_to_PDF(density_field,PDF_map,Par):
    '''
    fitering of a 1D power spectrum following exp(-(kR)^i1) in order for the corresponding variance to match var_tgt (the one obtained from the PDF)
    '''
    if Par['verbose'] and rank == 0: print('\n_______________________________ WORKING ON DENSITY PK ________________________________\n',flush=True)
        
    def balayage_filtering_1D(k,Pk,i1,var_tgt):
        '''
        successively filtering Pk in order to target the variance
        '''
        R_ = np.linspace(0,100,100000) ; var = []
        for R in R_: var.append(4*np.pi*np.trapz(k**2 * Pk *np.exp(-(k*R)**i1),k))
        filt_radius = np.interp(0,(np.array(var)-var_tgt)[::-1],R_[::-1])
        if filt_radius == R_[0] or filt_radius == R_[-1] : raise Exception('bad filtering process')
        Pk_filt = Pk*np.exp(-(k*filt_radius)**i1)
        filtering = np.exp(-(k*filt_radius)**i1)
        return Pk_filt,filtering

    if (not Par['Pk_dd_file'][-4:] == '.npy'):
        if Par['verbose'] and rank == 0: print('fitering the 1D power spectrum following exp(-(kR)^i1) in order for the corresponding variance to match', PDF_map['var_PDF'],flush=True)
        Pk_filtered,_ = balayage_filtering_1D(density_field['Pk_1D_dd'][0],density_field['Pk_1D_dd'][1],Par['i1'],PDF_map['var_PDF'])
        density_field['Pk_1D_dd_filtered'] = np.vstack((density_field['Pk_1D_dd'][0],Pk_filtered))
        if Par['debug'] and rank == 0: np.savetxt(Par['output_dir_project'] + '/debug_files/k_pk_filtered_with_i1',np.transpose(density_field['Pk_1D_dd_filtered']))
    
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
            print('filtering the 3D power spectrum to fit the right variance: from',var_obtained,'down to',target_var,'using the filtering parameter',index,flush=True)
        type_filt = -1
    elif var_obtained<target_var:    
        if verbose: 
            print('powering the 3D power spectrum to fit the right variance: from',var_obtained,'up to',target_var,'using the filtering parameter',index,flush=True)
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
        print('new variance is',np.sum(final)*k_F**3,'and should be',target_var,flush=True)
    return final,interpol

def Pk_nu_compute(density_field,PDF_map,Par,k_3D,k_1D):
    Ns = Par['N_sample'] ; k_F = 2*np.pi/Par['L']
    
    if rank == 0:
        if Par['PDF_d_file'] == 'gaussian':
            density_field['Pk_nu'] = PDF_map['var_PDF'] * density_field['Pk_3D_dd_alias']/(k_F**3*np.sum(density_field['Pk_3D_dd_alias']))
            
        else:
            pyfftw.config.NUM_THREADS = cpu_count()

            twopcf_delta = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')
            density_field['Pk_nu'] = pyfftw.empty_aligned((Ns,Ns,Ns), dtype='float64')

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                twopcf_delta[:,:,:] = pyfftw.interfaces.numpy_fft.ifftn(density_field['Pk_3D_dd_alias'],axes=(0,1,2))*(Ns*k_F)**3
                twopcf_nu           = np.interp(twopcf_delta,PDF_map['Xi_NG_template'],PDF_map['Xi_G_template']) ; del twopcf_delta
                density_field['Pk_nu'][:,:,:] = pyfftw.interfaces.numpy_fft.fftn(twopcf_nu,axes=(0,1,2))/(Ns*k_F)**3       ; del twopcf_nu

            if Par['verbose']: 
                print('percentage of negatives in Pk_nu_3D:',100*(len(density_field['Pk_nu'][density_field['Pk_nu']<0])/density_field['Pk_nu'].size),'percents','with variance',np.sum(density_field['Pk_nu'])*k_F**3,'\n',flush=True)
                print('applying now the clipping method',flush=True)
                
            density_field['Pk_nu'][density_field['Pk_nu']<0] = 0.

            if np.sum(density_field['Pk_nu'])*k_F**3 > 1.: density_field['Pk_nu'],_ = balayage_filtering_3D(Par['i2'],[0.5,3],density_field['Pk_nu'],1.,k_3D,k_F,verbose=Par['verbose'])

        if Par['compute_Pk_prediction'] and Par['Pk_dd_file'][-4:] == '.npy':
                density_field['Pk_1D_dd']    = np.zeros((2,len(k_1D)))
                density_field['Pk_1D_dd'][1] = fast_shell_averaging(Par,density_field['Pk_3D_dd_alias'])
                density_field['Pk_1D_dd'][0] = k_1D
        del density_field['Pk_3D_dd_alias']

    return density_field
    
def get_Pk_tt_3d(velocity_field,k_3D,Par):
    if Par['verbose'] and Par['velocity'] and rank == 0: print('\n_______________________________ WORKING ON VELOCITY PK _______________________________\n',flush=True)
        
    k_F = 2*np.pi/Par['L']
    
    if rank == 0 and Par['velocity']:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            velocity_field['Pk_3D_tt'] = np.exp(np.interp(np.log(k_3D),np.log(velocity_field['Pk_1D_tt'][0]),np.log(velocity_field['Pk_1D_tt'][1]),right=1e-10))
        #Pk_tt_3d *= np.exp(-(k_3D*filtering_radius)**index)
        if Par['verbose']: print('the target Pk_tt is interpolated on 3D modes. The corresponding target variance is',np.sum(velocity_field['Pk_3D_tt'])*k_F**3,'\n',flush=True)
    return velocity_field