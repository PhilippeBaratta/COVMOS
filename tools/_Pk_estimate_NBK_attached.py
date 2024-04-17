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


def Pk_poles_estimate_various_SN(sim_ref,Par,tot_obj,cat,v_cat):

    singlecomm = MPI.COMM_SELF # compute on a single thread without MPI
    def save_Pk_in_right_way(Pknbk,RSD_here,filename,L,Par,sim_ref,SN):
        poles = Pknbk.poles
        nk      = int(256/2 - 1)
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
        if SN == 0.2:
            np.savetxt(basename+sim_ref['number'],np.transpose(np.vstack((k_new,np.ones(len(k_new))*tot_obj,Pk_new[0 : k_new.size]/(2*np.pi)**3,Pk_new[k_new.size : k_new.size*2]/(2*np.pi)**3,Pk_new[k_new.size*2 : k_new.size*3]/(2*np.pi)**3))))
        else:
            np.savetxt(basename+sim_ref['number']+'_'+str(SN),np.transpose(np.vstack((k_new,np.ones(len(k_new))*tot_obj,Pk_new[0 : k_new.size]/(2*np.pi)**3,Pk_new[k_new.size : k_new.size*2]/(2*np.pi)**3,Pk_new[k_new.size*2 : k_new.size*3]/(2*np.pi)**3))))
        return
    
    for SN in [0.2,0.1,0.05,0.01,0.001]:
        
        indices = np.arange(tot_obj)
        number_to_select = int(SN * Par['L']**3)
        
        if number_to_select < tot_obj:
            to_be_selected = np.random.choice(indices,number_to_select,replace=False)

            cat0 = cat[0,:][to_be_selected]
            cat1 = cat[1,:][to_be_selected]
            cat2 = cat[2,:][to_be_selected]
        else : 
            number_to_select = tot_obj
            cat0 = cat[0,:]
            cat1 = cat[1,:]
            cat2 = cat[2,:]
        
        #comoving space
        data = np.empty(number_to_select, dtype=[('Position', (np.float32, 3))])
        data['Position'][:,0] = cat0 ; data['Position'][:,1] = cat1 ; data['Position'][:,2] = cat2
        f = ArrayCatalog({'Position' : data['Position']},comm=singlecomm) ; del data
        mesh_comov = f.to_mesh(Nmesh=256,BoxSize=Par['L'],dtype=np.float32,interlaced=True,compensated=True,resampler='pcs',position='Position') ; del f

        #comobile PK        
        Pknbk_comob = FFTPower(mesh_comov, mode='2d', poles = (0,2,4), dk=0, kmin=2*np.pi/Par['L']) ; del mesh_comov
        save_Pk_in_right_way(Pknbk_comob,False,sim_ref['sim_name']+'.data',Par['L'],Par,sim_ref,SN)

    return


def Pk_poles_estimate_edge(sim_ref,Par,tot_obj,cat,v_cat,Ary):

    singlecomm = MPI.COMM_SELF # compute on a single thread without MPI
    def save_Pk_in_right_way_edge(Pknbk,RSD_here,filename,L,Par,sim_ref,Ary):
        
        if RSD_here : basename = Par['file_Pk_RSD']
        else        : basename = Par['file_Pk']
        file = np.load(Par['folder_saving']+'indices_edge.npz')
        ind_32   = file['arr_1']
        ind_48   = file['arr_3']
        ind_64   = file['arr_5']
        ind_80   = file['arr_7']
        
        Pk32 = Pknbk[ind_32[0],ind_32[1],ind_32[2]]/(2*np.pi)**3 
        Pk48 = Pknbk[ind_48[0],ind_48[1],ind_48[2]]/(2*np.pi)**3 
        Pk64 = Pknbk[ind_64[0],ind_64[1],ind_64[2]]/(2*np.pi)**3 
        Pk80 = Pknbk[ind_80[0],ind_80[1],ind_80[2]]/(2*np.pi)**3 
        
        np.savetxt(basename+sim_ref['number']+'1000k3d_4shells',np.transpose(np.vstack((Pk32,Pk48,Pk64,Pk80))))
        return
    
    def save_Pk_in_right_way_edge2(Pknbk,RSD_here,filename,L,Par,sim_ref,Ary):
        poles = Pknbk.poles
        nk      = int(512/2 - 1)
        k_new   = np.zeros(nk-1)
        num_new = np.zeros(nk-1)
        Pk_new  = np.zeros((nk-1)*len((0,)))
        kF      = (2*np.pi)/L
        for i in range(1,poles['k'].size) :
            norm = poles['k'][i]
            if norm <= ((nk-1)*kF + kF/2.) :
                k_new[int(np.floor((norm-kF/2.)/kF))] += norm*poles['modes'][i]/2.
                num_new[int(np.floor((norm-kF/2.)/kF))] += poles['modes'][i]/2.
                for p in range(len((0,))) :
                    Pk_new[int(np.floor((norm-kF/2.)/kF)) + (nk-1)*p] += poles['power_'+str((0,)[p])][i].real*poles['modes'][i]/2.
        k_new /= num_new
        for p in range(len((0,))): Pk_new[p*k_new.size : k_new.size*(p+1)] /= num_new
        Pk_new[0 : k_new.size] -= poles.attrs['shotnoise']
        if RSD_here : basename = Par['file_Pk_RSD']
        else        : basename = Par['file_Pk']
        
        np.savetxt(basename+sim_ref['number']+'shells_avg',np.transpose(np.vstack((k_new,num_new,Pk_new[0:k_new.size]/(2*np.pi)**3))))
        return
    

    #comoving space
    data = np.empty(tot_obj, dtype=[('Position', (np.float32, 3))])
    data['Position'][:,0] = cat[0,:] ; data['Position'][:,1] = cat[1,:] ; data['Position'][:,2] = cat[2,:]
    f = ArrayCatalog({'Position' : data['Position']},comm=singlecomm) ; del data
    mesh_comov = f.to_mesh(Nmesh=512,BoxSize=Par['L'],dtype=np.float32,interlaced=False,compensated=False,resampler='pcs',position='Position') ; del f

    #redshift space
    if Par['velocity']:
        GADGET = 100*np.sqrt(1+Par['redshift'])
        cat[0,:],cat[1,:],cat[2,:] = apply_RSD(cat[0,:],cat[1,:],cat[2,:],v_cat[0,:]*GADGET,v_cat[1,:]*GADGET,v_cat[2,:]*GADGET,Par['redshift'],Par['Omega_m'],Par['L'])

        data = np.empty(tot_obj, dtype=[('Position', (np.float32, 3))])
        data['Position'][:,0] = cat[0,:] ; data['Position'][:,1] = cat[1,:] ; data['Position'][:,2] = cat[2,:]
        f = ArrayCatalog({'Position' : data['Position']},comm=singlecomm) ; del data
        mesh_redshift = f.to_mesh(Nmesh=512,BoxSize=Par['L'],dtype=np.float32,interlaced=False,compensated=False,resampler='pcs',position='Position') ; del f
    
    #comobile PK
    delta = mesh_comov.paint(mode='real').preview() -1 
    delta = np.fft.fftn(delta)
    Pk    = np.real(delta*np.conj(delta))
    save_Pk_in_right_way_edge(Pk,False,sim_ref['sim_name']+'.data',Par['L'],Par,sim_ref,Ary)
    
    Pknbk_comob = FFTPower(mesh_comov, mode='2d', poles = (0,), dk=0, kmin=2*np.pi/Par['L']) ; del mesh_comov
    #save_Pk_in_right_way_edge2(Pknbk_comob,False,sim_ref['sim_name']+'.data',Par['L'],Par,sim_ref,Ary)
    
    #redshift PK  
    if Par['velocity']:
        delta = mesh_redshift.paint(mode='real').preview() -1 
        delta = np.fft.fftn(delta)
        Pk    = np.real(delta*np.conj(delta))
        save_Pk_in_right_way_edge(Pk,True,sim_ref['sim_name']+'.data',Par['L'],Par,sim_ref,Ary)
        
        Pknbk_redshift = FFTPower(mesh_redshift, mode='2d', poles = (0,), dk=0, kmin=2*np.pi/Par['L']) ; del mesh_redshift
        #save_Pk_in_right_way_edge2(Pknbk_redshift,True,sim_ref['sim_name']+'.data',Par['L'],Par,sim_ref,Ary)
        
    return
























def Pk_poles_estimate_mask(sim_ref,Par,tot_obj,cat,v_cat,Ary):

    singlecomm = MPI.COMM_SELF # compute on a single thread without MPI
    
    from numba import njit, prange
    import nbodykit.source.mesh.catalog
    import math as mt

    @njit(parallel=True)
    def displace_observer(x, y, z, dx, dy, dz, L):
        x_dis, y_dis, z_dis = np.zeros(x.size), np.zeros(x.size), np.zeros(x.size)
        for i in prange(x.size):
            x_dis[i] = x[i] - dx
            y_dis[i] = y[i] - dy
            z_dis[i] = z[i] - dz
        return x_dis, y_dis, z_dis
    @njit(parallel=True)
    def spherical_coord(x, y, z):
        r, theta, phi = np.zeros(x.size), np.zeros(x.size), np.zeros(x.size)
        for i in prange(r.size):
            r[i] = np.sqrt(x[i]*x[i] + y[i]*y[i] + z[i]*z[i])
            theta[i] = np.arccos(z[i]/r[i])
            phi[i] = np.arctan2(y[i],x[i])
        return r, theta, phi

    @njit(parallel=True)
    def cartesian_coord(r, theta, phi):
        x, y, z = np.zeros(r.size), np.zeros(r.size), np.zeros(r.size)
        for i in prange(r.size):
            x[i] = r[i]*np.sin(theta[i])*np.cos(phi[i])
            y[i] = r[i]*np.sin(theta[i])*np.sin(phi[i])
            z[i] = r[i]*np.cos(theta[i])
        return x, y, z


    #---------------
    # Select particles in a cone with r=[rmin, rmax], theta = [theta_min, theta_max], phi = [phi_min, phi_max]
    # Input : catalog of points in a box in cartesian coordinates
    # Output : catalog of points in a cone in cartesian coordinates
    #---------------
    def select_part(x, y, z, rmax, rmin, theta_max, theta_min, phi_max=np.pi, phi_min=-np.pi):
        #-- Cartesian to spherical (the selection is easier in spherical coordinates)
        r, theta, phi = spherical_coord(x, y, z)

        #-- Select particles with with r in [rmin, rmax], theta in [theta_min, theta_max], phi in [phi_min, phi_max]
        r_s = r[(r<rmax) & (r>rmin) & (theta<theta_max) & (theta>theta_min) & (phi<phi_max) & (phi>phi_min)]
        theta_s = theta[(r<rmax) & (r>rmin) & (theta<theta_max) & (theta>theta_min) & (phi<phi_max) & (phi>phi_min)]
        phi_s = phi[(r<rmax) & (r>rmin) & (theta<theta_max) & (theta>theta_min) & (phi<phi_max) & (phi>phi_min)]

        #-- Selected particles : Spherical to cartesian coordinates
        x_s, y_s, z_s = cartesian_coord(r_s, theta_s, phi_s)
        return x_s, y_s, z_s
    
    #---------------
    # Compute |delta_k|^2 from delta_k (extracted from Nbody-kit)
    #---------------
    def compute_3d_power(deltak, L):
        #-- Compute the |delta_k|^2
        p3d = deltak.copy()
        for (s0, s1, s2) in zip(p3d.slabs, deltak.slabs, deltak.slabs):
            s0[...] = s1 * s2.conj()

        #-- remove the zero mode
        for i, s0 in zip(p3d.slabs.i, p3d.slabs):
            # clear the zero mode.
            mask = True
            for i1 in i:
                mask = mask & (i1 == 0)
            s0[mask] = 0    

        #-- multiply by L^3
        p3d[...] *= L**3
        return p3d

    def shell_average(p3d):
        #-- k binning
        kmax=np.pi*p3d.Nmesh.min()/p3d.BoxSize.max() + np.pi / p3d.BoxSize.max()
        kedges, kcoords = nbodykit.algorithms.fftpower._find_unique_edges(p3d.x, 2 * np.pi / p3d.BoxSize, kmax, p3d.pm.comm)
        muedges = np.linspace(-1, 1, 2, endpoint=True) # just an array [-1,1], i.e. theta in [0,2pi], i.e no binning in mu
        edges = [kedges, muedges]
        coords = [kcoords, None]
        #-- shell average
        result, pole_result = nbodykit.algorithms.fftpower.project_to_basis(p3d, edges, poles=[0], los=[0, 0, 1])
        #-- structure the array
        cols = ['k', 'power', 'modes']
        icols = [0, 2, 3]
        edges = edges[0:1]
        coords = coords[0:1]

        dtype = np.dtype([(name, result[icol].dtype.str) for icol,name in zip(icols,cols)])
        power = np.squeeze(np.empty(result[0].shape, dtype=dtype))
        for icol, col in zip(icols, cols):
            power[col][:] = np.squeeze(result[icol])
        return power

    def adapted_shells(nmesh, L, poles):
        kF = 2*np.pi/L
        nk = int(nmesh/2 - 1)
        k_new = np.zeros(nk-1)
        num_new = np.zeros(nk-1)
        Pk_new = np.zeros((nk-1))
        for i in range(1,poles['k'].size) :
            norm = poles['k'][i]
            if norm <= ((nk-1)*kF + kF/2.) :
                k_new[mt.floor((norm-kF/2.)/kF)] += norm*poles['modes'][i]/2.
                num_new[mt.floor((norm-kF/2.)/kF)] += poles['modes'][i]/2.
                Pk_new[mt.floor((norm-kF/2.)/kF)] += poles['power'][i].real*poles['modes'][i]/2.
        k_new /= num_new
        Pk_new /= num_new
        return k_new, Pk_new

    def compute_pk_from_deltak(deltak, nmesh, L):
        #-- Compute the |delta_k|^2
        p3d = compute_3d_power(deltak, L)
        #-- Shell average
        power = shell_average(p3d)
        #-- Adapt the shells
        k, Pk = adapted_shells(p3d.Nmesh.min(), p3d.BoxSize.max(), power)
        return k, Pk/(2*np.pi)**3
    
    def get_delta_k(x,y,z):
        data = np.empty(x.size, dtype=[('Position', (np.float32, 3))])
        data['Position'][:,0] = x; data['Position'][:,1] = y; data['Position'][:,2] = z
        caty = ArrayCatalog({'Position' : data['Position']},comm=singlecomm)
        mesh = caty.to_mesh(Nmesh=256, BoxSize=1000.,dtype=np.float32, position='Position', interlaced=True, compensated=True,resampler='pcs')
        deltak = mesh.compute(mode='complex', Nmesh=256)
        return deltak
    
    def volume_cone(theta_a, rmin, rmax):
        return 2*np.pi*(1-np.cos(theta_a))*(rmax**3/3. - rmin**3/3.)
    
    '''Wk = np.load('/pathto/randomcat/deltak_mean60_np2000000000_rmin800.0_rmax1000.0_tmax0.349065.npy')
    dx, dy, dz = 0,0,-1000 #Mpc/h

    delta_k_full_real = get_delta_k(cat[0,:], cat[1,:], cat[2,:])
    k,Pk_full_real    = compute_pk_from_deltak(delta_k_full_real, 512, 1000.)                            ; del delta_k_full_real
    
    x_real, y_real, z_real = displace_observer(cat[0,:], cat[1,:], cat[2,:], dx, dy, dz, 1000.)
    x_real, y_real, z_real = select_part(x_real, y_real, z_real, 1000., 800., np.pi/9, 0.)
    
    delta_k_masked_real  = get_delta_k(x_real, y_real, z_real)                                           ; del x_real, y_real, z_real
    delta_k_masked_real -= Wk
    k,Pk_masked_real     = compute_pk_from_deltak(delta_k_masked_real, 512, 1000.)                       ; del delta_k_masked_real
    Pk_masked_real      *= volume_cone(np.pi/9, 800, 1000)/1000.**3
    
    GADGET = 100*np.sqrt(1+Par['redshift'])
    cat[0,:],cat[1,:],cat[2,:] = apply_RSD(cat[0,:],cat[1,:],cat[2,:],v_cat[0,:]*GADGET,v_cat[1,:]*GADGET,v_cat[2,:]*GADGET,Par['redshift'],Par['Omega_m'],Par['L'])
    
    delta_k_full_rsd = get_delta_k(cat[0,:], cat[1,:], cat[2,:])
    k,Pk_full_rsd    = compute_pk_from_deltak(delta_k_full_rsd, 512, 1000.)                              ; del delta_k_full_rsd
    
    x_rsd, y_rsd, z_rsd = displace_observer(cat[0,:], cat[1,:], cat[2,:], dx, dy, dz, 1000.)
    x_rsd, y_rsd, z_rsd = select_part(x_rsd, y_rsd, z_rsd, 1000., 800., np.pi/9, 0.)
    
    delta_k_masked_rsd  = get_delta_k(x_rsd, y_rsd, z_rsd)                                               ; del x_rsd, y_rsd, z_rsd
    delta_k_masked_rsd -= Wk
    k,Pk_masked_rsd     = compute_pk_from_deltak(delta_k_masked_rsd, 512, 1000.)                         ; del delta_k_masked_rsd
    Pk_masked_rsd      *= volume_cone(np.pi/9, 800, 1000)/1000.**3
    
    np.savetxt(Par['file_Pk']+sim_ref['number'],np.transpose(np.vstack((k,np.ones(len(k))*tot_obj,Pk_full_real/(2*np.pi)**3,Pk_masked_real/(2*np.pi)**3))))
    np.savetxt(Par['file_Pk_RSD']+sim_ref['number'],np.transpose(np.vstack((k,np.ones(len(k))*tot_obj,Pk_full_rsd/(2*np.pi)**3,Pk_masked_rsd/(2*np.pi)**3))))'''
    
    
    def pk_in_sperical_mask(x,y,z,rmax,Wk_file):
        #adding the Sylvain's lines of code to analyticaly compute Wk for the spherical mask
        @njit(parallel=True)
        def bessel1(x):
            return np.sin(x)/x**2 - np.cos(x)/x

        @njit(parallel=True)
        def window_numba(k, R):
            return np.complex((4*np.pi*R**3)*((bessel1(k*R))/(k*R)))
        
        @njit(parallel=True)
        def compute_grid_Wk(kx, ky, kz, R):
            Wk = np.zeros((kx.size, ky.size, kz.size), dtype=np.complex128)
            for i in prange(kx.size):
                for j in prange(ky.size):
                    for k in prange(kz.size):
                        if i!=0 or j!=0 or k!=0:
                            Wk[i,j,k] = window_numba(np.sqrt(kx[i]**2+ky[j]**2+kz[k]**2), R)
                            #print(Wk[i,j,k])
            return Wk
        
        def volume_sphere(R):
            return 4./3.*(np.pi*R**3)
        
        
        r,theta,phi = spherical_coord(x,y,z)
        condition = r<rmax
        r     = r[condition]
        theta = theta[condition]
        phi   = phi[condition]                                                                               ; del condition
        x, y, z = cartesian_coord(r, theta, phi)                                                             ; del r, theta, phi
        
        new_number= x.size
        
        delta_k_masked  = get_delta_k(x, y, z)
        
        #Wk = np.load(Wk_file)
        #-- Get W(k) on the grid
        kx, ky, kz = delta_k_masked.x[0][:,0,0], delta_k_masked.x[1][0,:,0], delta_k_masked.x[2][0,0,:] # Get the right shape for the Fourier grid based on the nbodykit array 
        Wk = compute_grid_Wk(kx, ky, kz, rmax) # Compute W(k) on the grid
        Wk/= volume_sphere(rmax)
        
        delta_k_masked -= Wk
        k,Pk_masked     = compute_pk_from_deltak(delta_k_masked, 256, 1000.)                                 ; del delta_k_masked
        Pk_masked      *= (4/3*np.pi*rmax**3)/1000.**3
        return k,Pk_masked,np.ones(len(k))*new_number
    
    
    
    
    
    delta_k_full = get_delta_k(cat[0,:], cat[1,:], cat[2,:])
    k,Pk_full    = compute_pk_from_deltak(delta_k_full, 256, 1000.)                             ; del delta_k_full
    
    k,pk_500,n_500 = pk_in_sperical_mask(cat[0,:], cat[1,:], cat[2,:],500,'')
    k,pk_200,n_200 = pk_in_sperical_mask(cat[0,:], cat[1,:], cat[2,:],200,'')
    
    #GADGET = 100*np.sqrt(1+Par['redshift'])
    #cat[0,:],cat[1,:],cat[2,:] = apply_RSD(cat[0,:],cat[1,:],cat[2,:],v_cat[0,:]*GADGET,v_cat[1,:]*GADGET,v_cat[2,:]*GADGET,Par['redshift'],Par['Omega_m'],Par['L'])
    
    #delta_k_rsd = get_delta_k(cat[0,:], cat[1,:], cat[2,:])
    #k,Pk_rsd    = compute_pk_from_deltak(delta_k_rsd, 512, 1000.)                               ; del delta_k_rsd
    
    #k,pk_500_rsd,n_500_rsd = pk_in_sperical_mask(cat[0,:], cat[1,:], cat[2,:],500,'/pathtorep/Wk_sphere_r500.0.npy')
    #k,pk_450_rsd,n_450_rsd = pk_in_sperical_mask(cat[0,:], cat[1,:], cat[2,:],450,'/pathtorep/Wk_sphere_r450.0.npy')
    #k,pk_400_rsd,n_400_rsd = pk_in_sperical_mask(cat[0,:], cat[1,:], cat[2,:],400,'/pathtorep/Wk_sphere_r400.0.npy')
    #k,pk_350_rsd,n_350_rsd = pk_in_sperical_mask(cat[0,:], cat[1,:], cat[2,:],350,'/pathtorep/Wk_sphere_r350.0.npy')
    #k,pk_300_rsd,n_300_rsd = pk_in_sperical_mask(cat[0,:], cat[1,:], cat[2,:],300,'/pathtorep/Wk_sphere_r300.0.npy')
    #k,pk_200_rsd,n_200_rsd = pk_in_sperical_mask(cat[0,:], cat[1,:], cat[2,:],200,'')

    np.savetxt(Par['file_Pk']    +sim_ref['number'],np.transpose(np.vstack((k,np.ones(len(k))*tot_obj,n_500,n_200,Pk_full,pk_500,pk_200))))
    #np.savetxt(Par['file_Pk_RSD']+sim_ref['number'],np.transpose(np.vstack((k,np.ones(len(k))*tot_obj,n_500_rsd,n_350_rsd,n_200_rsd,Pk_rsd,pk_500_rsd,pk_350_rsd,pk_200_rsd))))
    return

    
    
def Pk_poles_estimate_instru(sim_ref,Par,tot_obj,cat,v_cat,Ary):

    singlecomm = MPI.COMM_SELF # compute on a single thread without MPI
    
    from numba import njit, prange

    
    
    import nbodykit.source.mesh.catalog
    import math as mt
    ## //////////////////////////////////////////// ###
    #                NBODYKIT UTILS
    ## //////////////////////////////////////////// ###

    #---------------
    # Compute |delta_k|^2 from delta_k (extracted from Nbody-kit)
    #---------------
    def compute_3d_power(deltak, L):
        #-- Compute the |delta_k|^2
        p3d = deltak.copy()
        for (s0, s1, s2) in zip(p3d.slabs, deltak.slabs, deltak.slabs):
            s0[...] = s1 * s2.conj()

        #-- remove the zero mode
        for i, s0 in zip(p3d.slabs.i, p3d.slabs):
            # clear the zero mode.
            mask = True
            for i1 in i:
                mask = mask & (i1 == 0)
            s0[mask] = 0    

        #-- multiply by L^3
        p3d[...] *= L**3
        return p3d

    #---------------
    # Perform the shell averaging (extracted from Nbody-kit)
    #---------------
    def shell_average(p3d):
        #-- k binning
        kmax=np.pi*p3d.Nmesh.min()/p3d.BoxSize.max() + np.pi / p3d.BoxSize.max()
        kedges, kcoords = nbodykit.algorithms.fftpower._find_unique_edges(p3d.x, 2 * np.pi / p3d.BoxSize, kmax, p3d.pm.comm)
        muedges = np.linspace(-1, 1, 2, endpoint=True) # just an array [-1,1], i.e. theta in [0,2pi], i.e no binning in mu
        edges = [kedges, muedges]
        coords = [kcoords, None]
        #-- shell average
        result, pole_result = nbodykit.algorithms.fftpower.project_to_basis(p3d, edges, poles=[0], los=[0, 0, 1])
        #-- structure the array
        cols = ['k', 'power', 'modes']
        icols = [0, 2, 3]
        edges = edges[0:1]
        coords = coords[0:1]

        dtype = np.dtype([(name, result[icol].dtype.str) for icol,name in zip(icols,cols)])
        power = np.squeeze(np.empty(result[0].shape, dtype=dtype))
        for icol, col in zip(icols, cols):
            power[col][:] = np.squeeze(result[icol])
        return power

    #---------------
    # Adapte the shell averaging (extracted from Nbody-kit)
    #---------------
    def adapted_shells(nmesh, L, poles):
        kF = 2*np.pi/L
        nk = int(nmesh/2 - 1)
        k_new = np.zeros(nk-1)
        num_new = np.zeros(nk-1)
        Pk_new = np.zeros((nk-1))
        for i in range(1,poles['k'].size) :
            norm = poles['k'][i]
            if norm <= ((nk-1)*kF + kF/2.) :
                k_new[mt.floor((norm-kF/2.)/kF)] += norm*poles['modes'][i]/2.
                num_new[mt.floor((norm-kF/2.)/kF)] += poles['modes'][i]/2.
                Pk_new[mt.floor((norm-kF/2.)/kF)] += poles['power'][i].real*poles['modes'][i]/2.
        k_new /= num_new
        Pk_new /= num_new
        return k_new, Pk_new

    #---------------
    # Full Nbodykit pipeline to compute the P(k) from delta_k
    #---------------
    def compute_pk_from_deltak(deltak, nmesh, L):
        #-- Compute the |delta_k|^2
        p3d = compute_3d_power(deltak, L)
        #-- Shell average
        power = shell_average(p3d)
        #-- Adapt the shells
        k, Pk = adapted_shells(p3d.Nmesh.min(), p3d.BoxSize.max(), power)
        return k, Pk
    
    def get_delta_k(x,y,z):
        data = np.empty(x.size, dtype=[('Position', (np.float32, 3))])
        data['Position'][:,0] = x; data['Position'][:,1] = y; data['Position'][:,2] = z
        caty = ArrayCatalog({'Position' : data['Position']},comm=singlecomm)
        mesh = caty.to_mesh(Nmesh=512, BoxSize=1000., position='Position', interlaced=True, compensated=True)
        deltak = mesh.compute(mode='complex', Nmesh=512)
        return deltak
    
    
    def pk_in_sperical_mask(x,y,z,rmax,Wk_file):
        r,theta,phi = spherical_coord(x,y,z)
        condition = r<rmax
        r     = r[condition]
        theta = theta[condition]
        phi   = phi[condition]                                                                               ; del condition
        x, y, z = cartesian_coord(r, theta, phi)                                                             ; del r, theta, phi
        
        delta_k_masked  = get_delta_k(x, y, z)
        Wk = np.load(Wk_file)
        delta_k_masked -= Wk
        k,Pk_masked     = compute_pk_from_deltak(delta_k_masked, 512, 1000.)                                 ; del delta_k_masked
        Pk_masked      *= (4/3*np.pi*rmax**3)/1000.**3
        return k,Pk_masked/(2*np.pi)**3 
    
    
    def compute_pk_with_error(x,y,z,error):
        from scipy.integrate import quad
        def redshifttodistance(z,c,H0,Omega_m,Omega_l):
            def integrand(z):
                return (c/H0)/(np.sqrt(Omega_m*(1.+z)**3 + Omega_l));#gives value in h^-1
            return quad(integrand, 0, z)[0];
        redshifttodistance = np.vectorize(redshifttodistance)

        zs  = np.linspace(0,0.25,1000)
        MPC = redshifttodistance(zs,299792.458,67,0.32,1-0.32)

        zs_cat = np.interp(z,MPC,zs)
        zs_cat = np.random.normal(loc=zs_cat,scale=error)
        new_zcartesian  = redshifttodistance(zs_cat,299792.458,67,0.32,1-0.32)

        condition = (new_zcartesian<1000) * (new_zcartesian>0)

        delta_k = get_delta_k(x[condition], y[condition], new_zcartesian[condition])
        k,Pk    = compute_pk_from_deltak(delta_k, 512, 1000.)
        return k,Pk
        
    cat+=500.
    delta_k_full = get_delta_k(cat[0,:], cat[1,:], cat[2,:])
    k,Pk_full    = compute_pk_from_deltak(delta_k_full, 512, 1000.)                             ; del delta_k_full
    
    k_,pk_01 = compute_pk_with_error(cat[0,:],cat[1,:],cat[2,:],0.001)
    k_,pk_05 = compute_pk_with_error(cat[0,:],cat[1,:],cat[2,:],0.005)
    k_,pk_1  = compute_pk_with_error(cat[0,:],cat[1,:],cat[2,:],0.01)
    
    #GADGET = 100*np.sqrt(1+Par['redshift'])
    #cat[0,:],cat[1,:],cat[2,:] = apply_RSD(cat[0,:],cat[1,:],cat[2,:],v_cat[0,:]*GADGET,v_cat[1,:]*GADGET,v_cat[2,:]*GADGET,Par['redshift'],Par['Omega_m'],Par['L'])
    
    #delta_k_rsd = get_delta_k(cat[0,:], cat[1,:], cat[2,:])
    #k,Pk_rsd    = compute_pk_from_deltak(delta_k_rsd, 512, 1000.)                             ; del delta_k_rsd
    
    #k_,pk_01_rsd = compute_pk_with_error(cat[0,:],cat[1,:],cat[2,:],0.001)
    #k_,pk_05_rsd = compute_pk_with_error(cat[0,:],cat[1,:],cat[2,:],0.005)
    #k_,pk_1_rsd  = compute_pk_with_error(cat[0,:],cat[1,:],cat[2,:],0.01)
    
    np.savetxt(Par['file_Pk']    +sim_ref['number'],np.transpose(np.vstack((k,np.ones(len(k))*tot_obj,Pk_full,pk_01,pk_05,pk_1))))
    #np.savetxt(Par['file_Pk_RSD']+sim_ref['number'],np.transpose(np.vstack((k,np.ones(len(k))*tot_obj,Pk_rsd,pk_01_rsd,pk_05_rsd,pk_1_rsd))))
    return
