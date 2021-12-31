import numpy as np

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