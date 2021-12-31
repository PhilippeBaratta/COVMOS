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
def sqrt_of_arrays_times_scalars(array1,scalar1,array2,scalar2):
    '''
    computes sqrt(arr*scalar) for two pairs arrays-scalar using numba
    '''
    array1_ = np.zeros_like(array1)
    array2_ = np.zeros_like(array2)
    for i in prange(array1_.shape[0]):
        for j in prange(array1_.shape[1]):
            for k in prange(array1_.shape[2]):
                    array1_[i,j,k] = np.sqrt(array1[i,j,k]*scalar1)
                    array2_[i,j,k] = np.sqrt(array2[i,j,k]*scalar2)
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