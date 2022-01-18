from glob import glob
from os.path import join
import numpy as np
    
    
def compute_COVMOS_covariance(Par):
    def get_unbiased_multipoles(rep,density_part,L,RSD):
        
        if not RSD:
            data = np.load(rep+'COVMOS_all_Pk.npy')
        if RSD:
            data = np.load(rep+'COVMOS_all_Pk_RSD.npy')

        Pk_UB   = np.zeros((data.shape[0],254))  
        quad_UB = np.zeros((data.shape[0],254))
        hexa_UB = np.zeros((data.shape[0],254))

        for i in range(data.shape[0]):
            Pk_UB[i,:]   = (density_part*L**3 /data[i,1,0])**3  * data[i,2,:]
            quad_UB[i,:] = (density_part*L**3 /data[i,1,0])**3  * data[i,3,:]
            hexa_UB[i,:] = (density_part*L**3 /data[i,1,0])**3  * data[i,4,:]
            
        return Pk_UB,quad_UB,hexa_UB
    
    if Par['compute_covariance']:
        if Par['verbose']: print('computing the unbiased covariances in',Par['folder_cov'],flush=True)

        Pk_UB_comov,quad_UB_comov,hexa_UB_comov = get_unbiased_multipoles(Par['folder_Pk'],Par['rho_0'],Par['L'],False)
        multipoles_UB_comov = np.concatenate((Pk_UB_comov,quad_UB_comov,hexa_UB_comov),axis=1)

        cov_Pk_comov    = np.cov(np.transpose(Pk_UB_comov)  ,bias=False)
        cov_quad_comov  = np.cov(np.transpose(quad_UB_comov),bias=False)
        cov_hexa_comov  = np.cov(np.transpose(hexa_UB_comov),bias=False)
        cov_multi_comov = np.cov(np.transpose(multipoles_UB_comov),bias=False)

        np.save(Par['folder_cov']+'COVMOS_cov_monopole_noRSD',cov_Pk_comov)
        np.save(Par['folder_cov']+'COVMOS_cov_quadrupole_noRSD',cov_quad_comov)
        np.save(Par['folder_cov']+'COVMOS_cov_hexadecapole_noRSD',cov_hexa_comov)
        np.save(Par['folder_cov']+'COVMOS_cov_multipoles_noRSD',cov_multi_comov)

        if Par['velocity']:
            Pk_UB_red,quad_UB_red,hexa_UB_red = get_unbiased_multipoles(Par['folder_Pk_RSD'],Par['rho_0'],Par['L'],True)
            multipoles_UB_red = np.concatenate((Pk_UB_red,quad_UB_red,hexa_UB_red),axis=1)

            cov_Pk_red    = np.cov(np.transpose(Pk_UB_red)  ,bias=False)
            cov_quad_red  = np.cov(np.transpose(quad_UB_red),bias=False)
            cov_hexa_red  = np.cov(np.transpose(hexa_UB_red),bias=False)
            cov_multi_red = np.cov(np.transpose(multipoles_UB_red),bias=False)

            np.save(Par['folder_cov']+'COVMOS_cov_monopole_RSD',cov_Pk_red)
            np.save(Par['folder_cov']+'COVMOS_cov_quadrupole_RSD',cov_quad_red)
            np.save(Par['folder_cov']+'COVMOS_cov_hexadecapole_RSD',cov_hexa_red)
            np.save(Par['folder_cov']+'COVMOS_cov_multipoles_RSD',cov_multi_red)
        
    return