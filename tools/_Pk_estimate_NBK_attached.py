import numpy as np
from nbodykit.source.catalog import ArrayCatalog
from nbodykit.lab import *

from tools._velocity_model import *

def Pk_poles_estimate(sim_ref,Par,tot_obj,cat,v_cat):

    singlecomm = MPI.COMM_SELF # compute on a single thread without MPI
    def save_Pk_in_right_way(Pknbk,RSD_here,filename,L,Par,sim_ref):
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
        if RSD_here : basename = Par['file_Pk_RSD']
        else        : basename = Par['file_Pk']
        np.savetxt(basename+sim_ref['number'],np.transpose(np.vstack((k_new,np.ones(len(k_new))*tot_obj,Pk_new[0 : k_new.size]/(2*np.pi)**3,Pk_new[k_new.size : k_new.size*2]/(2*np.pi)**3,Pk_new[k_new.size*2 : k_new.size*3]/(2*np.pi)**3))))
        return

    #comoving space
    data = np.empty(tot_obj, dtype=[('Position', (np.float32, 3))])
    data['Position'][:,0] = cat[0,:] ; data['Position'][:,1] = cat[1,:] ; data['Position'][:,2] = cat[2,:]
    f = ArrayCatalog({'Position' : data['Position']},comm=singlecomm) ; del data
    mesh_comov = f.to_mesh(Nmesh=512,BoxSize=Par['L'],dtype=np.float32,interlaced=True,compensated=True,resampler='pcs',position='Position') ; del f

    #redshift space
    if Par['velocity']:
        GADGET = 100*np.sqrt(1+Par['redshift'])
        cat[0,:],cat[1,:],cat[2,:] = apply_RSD(cat[0,:],cat[1,:],cat[2,:],v_cat[0,:]*GADGET,v_cat[1,:]*GADGET,v_cat[2,:]*GADGET,Par['redshift'],Par['Omega_m'],Par['L'])

        data = np.empty(tot_obj, dtype=[('Position', (np.float32, 3))])
        data['Position'][:,0] = cat[0,:] ; data['Position'][:,1] = cat[1,:] ; data['Position'][:,2] = cat[2,:]
        f = ArrayCatalog({'Position' : data['Position']},comm=singlecomm) ; del data
        mesh_redshift = f.to_mesh(Nmesh=512,BoxSize=Par['L'],dtype=np.float32,interlaced=True,compensated=True,resampler='pcs',position='Position') ; del f

    #comobile PK        
    Pknbk_comob = FFTPower(mesh_comov, mode='2d', poles = (0,2,4), dk=0, kmin=2*np.pi/Par['L']) ; del mesh_comov
    save_Pk_in_right_way(Pknbk_comob,False,sim_ref['sim_name']+'.data',Par['L'],Par,sim_ref)

    #redshift PK  
    if Par['velocity']:
        Pknbk_red = FFTPower(mesh_redshift, mode='2d', poles = (0,2,4), dk=0, kmin=2*np.pi/Par['L']) ; del mesh_redshift
        save_Pk_in_right_way(Pknbk_red,True,sim_ref['sim_name']+'.data',Par['L'],Par,sim_ref)
    return

