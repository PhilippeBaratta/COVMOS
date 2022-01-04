import numpy as np
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import *
from nbodykit import use_mpi, CurrentMPIComm
comm = CurrentMPIComm.get()
use_mpi(comm)
import sys
import os
import time

from _velocity_model import *

filename = str (sys.argv[1])
RSD      = str(sys.argv[2])
z_snap   = float(sys.argv[3])
L        = float(sys.argv[4])
savecat  = str(sys.argv[5])
Omega_m  = float(sys.argv[6])
number_ref  = str(sys.argv[7])
Pk_file  = str(sys.argv[8])
Pk_RSD_file  = str(sys.argv[9])

if savecat == 'False' : savecat = False 
if savecat == 'True'  : savecat = True
    
if RSD == 'False' : RSD = False 
if RSD == 'True'  : RSD = True

def save_Pk_in_right_way(Pknbk,RSD_here,filename,L,number_ref,Pk_file,Pk_RSD_file):
    poles = Pknbk.poles
    nk      = int(512/2 - 1)
    k_new   = np.zeros(nk-1)
    num_new = np.zeros(nk-1)
    Pk_new  = np.zeros((nk-1)*len((0,2,4)))
    kF      = (2*np.pi)/L
    for i in range(1,poles['k'].size) :
        norm = poles['k'][i]
        if norm <= ((nk-1)*kF + kF/2.) :
            k_new[int(np.floor((norm-kF/2.)/kF))] += norm*poles['modes'][i]/2.
            num_new[int(np.floor((norm-kF/2.)/kF))] += poles['modes'][i]/2.
            for p in range(len((0,2,4))) :
                Pk_new[int(np.floor((norm-kF/2.)/kF)) + (nk-1)*p] += poles['power_'+str((0,2,4)[p])][i].real*poles['modes'][i]/2.
    k_new /= num_new
    for p in range(len((0,2,4))): Pk_new[p*k_new.size : k_new.size*(p+1)] /= num_new
    Pk_new[0 : k_new.size] -= poles.attrs['shotnoise']
    if comm.rank == 0:
        if RSD_here : basename = Pk_RSD_file
        else        : basename = Pk_file
        np.savetxt(basename+number_ref,np.transpose(np.vstack((k_new,np.ones(len(k_new))*number,Pk_new[0 : k_new.size]/(2*np.pi)**3,Pk_new[k_new.size : k_new.size*2]/(2*np.pi)**3,Pk_new[k_new.size*2 : k_new.size*3]/(2*np.pi)**3))))
    return

def loadcatalogue(filename,RSD=False):
    myarray = np.fromfile(filename, dtype=np.float32)
    myarray = np.delete(myarray,0)
    if   RSD == False : nbrarray = 3
    elif RSD == True  : nbrarray = 6
    catalog = np.reshape(myarray,(nbrarray,int(len(myarray)/nbrarray)))
    x = catalog[0] ; y = catalog[1] ; z = catalog[2]
    if RSD == True:
        vx = catalog[3] ; vy = catalog[4] ; vz = catalog[5]
    if RSD == False:
        return x,y,z
    elif RSD == True:
        return x,y,z,vx,vy,vz

time.sleep(5)

if RSD: x,y,z,vx,vy,vz = loadcatalogue(filename,RSD=RSD)
else  : x,y,z          = loadcatalogue(filename,RSD=RSD)
number = len(x)
    
#comoving space
data = np.empty(number, dtype=[('Position', (np.float32, 3))])
data['Position'][:,0] = x ; data['Position'][:,1] = y ; data['Position'][:,2] = z
f = ArrayCatalog({'Position' : data['Position']},comm=comm) ; del data
mesh_comov = f.to_mesh(Nmesh=512,BoxSize=L,dtype=np.float32,interlaced=True,compensated=True,resampler='pcs',position='Position') ; del f

#redshift space
if RSD:
    x,y,z = apply_RSD(x,y,z,vx,vy,vz,z_snap,Omega_m,L) ; del vx,vy,vz
    
    data = np.empty(number, dtype=[('Position', (np.float32, 3))])
    data['Position'][:,0] = x ; data['Position'][:,1] = y ; data['Position'][:,2] = z
    f = ArrayCatalog({'Position' : data['Position']},comm=comm) ; del data
    mesh_redshift = f.to_mesh(Nmesh=512,BoxSize=L,dtype=np.float32,interlaced=True,compensated=True,resampler='pcs',position='Position') ; del f
    
del x,y,z
if not savecat: os.remove(filename)
        
#comobile PK        
Pknbk_comob = FFTPower(mesh_comov, mode='2d', poles = (0,2,4), dk=0, kmin=2*np.pi/L) ; del mesh_comov
save_Pk_in_right_way(Pknbk_comob,False,filename,L,number_ref,Pk_file,Pk_RSD_file)

#redshift PK  
if RSD:
    Pknbk_red = FFTPower(mesh_redshift, mode='2d', poles = (0,2,4), dk=0, kmin=2*np.pi/L) ; del mesh_redshift
    save_Pk_in_right_way(Pknbk_red,True,filename,L,number_ref,Pk_file,Pk_RSD_file)