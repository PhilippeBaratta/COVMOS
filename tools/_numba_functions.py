from numba import njit, prange
import numpy as np

@njit(parallel=True,cache=True)
def array_times_scalar(array,scalar):
    '''
    performs the product of a numpy array to a scalar using numba
    '''
    array_ = np.zeros_like(array)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                    array_[i,j,k] = array[i,j,k]*scalar
    return array_
    
@njit(parallel=True,cache=True)
def array_times_array(array1,array2):
    '''
    computes the product of two numpy arrays using numba
    '''
    array_ = np.zeros_like(array1)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                array_[i,j,k] = array1[i,j,k]*array2[i,j,k]
    return array_
    
@njit(parallel=True,cache=True)
def array_times_arraypow2(array1,array2):
    '''
    computes the product array1*array2^2 using numba
    '''
    array_ = np.zeros_like(array1)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                array_[i,j,k] = array1[i,j,k]*array2[i,j,k]**2
    return array_
    
@njit(parallel=True,cache=True)
def array_times_array_times_array(array1,array2,array3):
    '''
    computes the product of three numpy arrays using numba
    '''
    array_ = np.zeros_like(array1)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                array_[i,j,k] = array1[i,j,k]*array2[i,j,k]*array3[i,j,k]
    return array_
    
@njit(parallel=True,cache=True)
def fast_norm(k1,k2,k3):
    '''
    computes the grid norm of 3 numpy arrays using numba
    '''
    array_ = np.zeros_like(k1)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                array_[i,j,k] = np.sqrt(k1[i,j,k]**2 + k2[i,j,k]**2 + k3[i,j,k]**2)
    return array_

@njit(parallel=True,cache=True)
def fast_sqrtnorm(k1,k2,k3):
    '''
    computes the grid norm of 3 numpy arrays using numba
    '''
    array_ = np.zeros_like(k1)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                array_[i,j,k] = k1[i,j,k]**2 + k2[i,j,k]**2 + k3[i,j,k]**2
    return array_
    
@njit(parallel=True,cache=True)
def sqrt_of_array_times_scalar(array,scalar):
    '''
    computes sqrt(arr*scalar) using numba
    '''
    array_ = np.zeros_like(array)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                    array_[i,j,k] = np.sqrt(array[i,j,k]*scalar)
    return array_

@njit(parallel=True,cache=True)
def sqrt_of_arrays_times_scalars(array1,scalar1,array2,scalar2,k_3D):
    '''
    computes sqrt(arr*scalar) for two pairs arrays-scalar using numba
    '''
    array1_ = np.zeros_like(array1)
    array2_ = np.zeros_like(array2)
    for i in prange(array1_.shape[0]):
        for j in prange(array1_.shape[1]):
            for k in prange(array1_.shape[2]):
                if k_3D_2[i,j,k] != 0:
                    array1_[i,j,k] = np.sqrt(array1[i,j,k]*scalar1)
                    array2_[i,j,k] = np.sqrt(array2[i,j,k]*scalar2)/k_3D[i,j,k]**2
    return array1_,array2_
    
@njit(parallel=True,cache=True)
def log_kalias(n1,n2,n3,k_N,j_contrib,kx,ky,kz):
    '''
    computes the logarithm of the 3D mode on which the aliasing procedure is interpolating using numba
    '''
    logk_alias = np.zeros_like(kx)
    for i in prange(logk_alias.shape[0]):
        for j in prange(logk_alias.shape[0]):
            for k in prange(logk_alias.shape[0]):
                logk_alias[i,j,k] = np.log(np.sqrt((kx[i,j,k]-2.*(n1-j_contrib)*k_N)**2 + (ky[i,j,k]-2.*(n2-j_contrib)*k_N)**2 + (kz[i,j,k]-2.*(n3-j_contrib)*k_N)**2))
    return logk_alias

@njit(parallel=True,cache=True)
def exppara(array1,array2):
    '''
    computes arr1 + exp(arr2) using numba
    '''
    exp = np.zeros_like(array1)
    for i in prange(exp.shape[0]):
        for j in prange(exp.shape[0]):
            for k in prange(exp.shape[0]):
                exp[i,j,k] = array1[i,j,k]+np.exp(array2[i,j,k])
    return exp

@njit(parallel=True,cache=True)
def filtering_operation(pktofilt_,type_filt_,norm_3D_,R_,index_):
    '''
    computes Pk exp(+-(k_3D*R)^i) using numba
    '''
    array_ = np.zeros_like(pktofilt_)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                array_[i,j,k] = pktofilt_[i,j,k] * np.exp(type_filt_*(norm_3D_[i,j,k]*R_)**index_)
    return array_

@njit(parallel = True, cache=True)
def inverse_div_theta(complex1,byprod_pk_velocity):
    array = np.zeros_like(complex1)
    sampling = array.shape[0]
    for i in prange(sampling):
        for j in prange(sampling):
            for k in prange(sampling):
                if k_3D_2[i,j,k] != 0:
                    array[i,j,k] = -1j*(complex1[i,j,k]*byprod_pk_velocity[i,j,k])
    return array  

@njit(parallel=True,cache=True)
def from_delta_to_rho_times_a3(delta,rho_0a3):
    array_ = np.zeros_like(delta)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                    array_[i,j,k] = (delta[i,j,k]+1)*rho_0a3
    return array_

@njit(parallel=True, cache=True)
def mean_delta_times_a3(delta,a3,rho_0):
    rho_       = np.zeros_like(delta)
    mean_grid_ = np.zeros_like(delta)
    sampling = mean_grid_.shape[0]
    
    for i in prange(sampling):
        for j in prange(sampling):
            for k in prange(sampling):
                rho_[i,j,k] = (delta[i,j,k]+1)*rho_0
                
    for i in prange(sampling):
        for j in prange(sampling):
            for k in prange(sampling):
                ip1 = (i != (sampling-1)) * (i+1)
                jp1 = (j != (sampling-1)) * (j+1)
                kp1 = (k != (sampling-1)) * (k+1)
                
                mean_grid_[i,j,k] = a3 *(rho_[i,j,k] + rho_[ip1,j,k] + rho_[i,jp1,k] + rho_[i,j,kp1] + rho_[ip1,jp1,k] + rho_[ip1,j,kp1] + rho_[i,jp1,kp1] + rho_[ip1,jp1,kp1]) / 8
    return rho_,mean_grid_

@njit(parallel=True,cache=True)
def expo(array1):
    array_ = np.zeros_like(array1)
    for i in prange(array_.shape[0]):
        for j in prange(array_.shape[1]):
            for k in prange(array_.shape[2]):
                    array_[i,j,k] = np.exp(array1[i,j,k])
    return array_
