import numpy as np
from numba import njit, prange

def discrete_assignment(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,rho,v_grid,Par,Ary):
    a = Par['a']
    velocity = Par['velocity']
    N_sample = Par['N_sample']
    unit = Par['unit']
    
    if Par['assign_scheme'] == 'tophat':
        cat,rho_itp,v_itp = discrete_assignment_tophat(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,Ary['grid_pos'],a)
    if Par['assign_scheme'] == 'trilinear':
        if velocity:
            cat,rho_itp,v_itp = discrete_assignment_trilinear_velocity(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,rho,v_grid,Ary['grid_pos'],Ary['vertex'],N_sample,a,unit)
        else:
            cat,rho_itp,v_itp = discrete_assignment_trilinear(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,rho,v_grid,Ary['grid_pos'],Ary['vertex'],N_sample,a,unit)
    return cat,rho_itp,v_itp


@njit(parallel=True,cache=True)
def discrete_assignment_tophat(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,grid_pos,a):
    for i in prange(len(non_0_)):
        non_0         = non_0_[i]
        N_cell        = Nbr [non_0]
        cumu_non_0    = cumu[non_0]
        cumu_non_0_p1 = cumu[non_0+1]
        les_randoms_  = les_randoms[:,cumu_non_0:cumu_non_0_p1]
        grille        = grid_pos[:,non_0]
        cat[0,cumu_non_0:cumu_non_0_p1] = les_randoms_[0]*a + grille[0]
        cat[1,cumu_non_0:cumu_non_0_p1] = les_randoms_[1]*a + grille[1]
        cat[2,cumu_non_0:cumu_non_0_p1] = les_randoms_[2]*a + grille[2]
    return cat,rho_itp,v_itp
            
@njit(parallel=True,cache=True)
def discrete_assignment_trilinear(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,rho,v_grid,grid_pos,vertex,N_sample,a,unit):
    for i in prange(len(non_0_)):
        non_0         = non_0_[i]
        N_cell        = Nbr [non_0]
        cumu_non_0    = cumu[non_0]
        cumu_non_0_p1 = cumu[non_0+1]
        les_randoms_  = les_randoms[:,cumu_non_0:cumu_non_0_p1]
        grille        = grid_pos[:,non_0]
        
        vtx_0 = vertex[0,non_0]                     ; vtx_1 = vertex[1,non_0]                     ; vtx_2 = vertex[2,non_0]
        vtx_3 = (vtx_0 != (N_sample-1)) * (vtx_0+1) ; vtx_4 = (vtx_1 != (N_sample-1)) * (vtx_1+1) ; vtx_5 = (vtx_2 != (N_sample-1)) * (vtx_2+1)
        
        rho_000 = rho[vtx_0,vtx_1,vtx_2] ; rho_a0a = rho[vtx_3,vtx_1,vtx_5] ; rho_0a0 = rho[vtx_0,vtx_4,vtx_2] ; rho_aa0 = rho[vtx_3,vtx_4,vtx_2]
        rho_a00 = rho[vtx_3,vtx_1,vtx_2] ; rho_0aa = rho[vtx_0,vtx_4,vtx_5] ; rho_00a = rho[vtx_0,vtx_1,vtx_5] ; rho_aaa = rho[vtx_3,vtx_4,vtx_5]
        
        C1 = rho_000+rho_0a0+rho_00a+rho_0aa
        C2 = -rho_000+rho_a00-rho_0a0-rho_00a+rho_a0a-rho_0aa+rho_aa0+rho_aaa
        C3 = rho_000+rho_a00+rho_0a0+rho_00a+rho_a0a+rho_0aa+rho_aa0+rho_aaa
        x  = (a*(-C1+np.sqrt(C1**2+C2*C3*les_randoms_[0])))/C2
        
        C1 = a*(-rho_000-rho_00a+rho_0a0+rho_0aa)+x*(rho_000+rho_00a-rho_0a0-rho_0aa-rho_a00-rho_a0a+rho_aa0+rho_aaa)
        C2 = 2*a**2 *(rho_000+rho_00a)-2*a*x*(rho_000+rho_00a-rho_a00-rho_a0a)
        C3 = a**2*(a*(rho_000+rho_00a+rho_0a0+rho_0aa)+x*(-rho_000-rho_00a-rho_0a0-rho_0aa+rho_a00+rho_a0a+rho_aa0+rho_aaa))
        y  = -(C2-np.sign(C1)*np.sqrt(C2**2+4*C1*C3*les_randoms_[1])*np.sign(C1))/(2*C1)
        
        C1 = a*(a*(-rho_000+rho_00a)+x*(rho_000-rho_00a-rho_a00+rho_a0a)+y*(rho_000-rho_00a-rho_0a0+rho_0aa))+x*y*(-rho_000+rho_00a+rho_0a0-rho_0aa+rho_a00-rho_a0a-rho_aa0+rho_aaa)
        C2 = 2*a*(a*(a*rho_000+x*(-rho_000+rho_a00)+y*(-rho_000+rho_0a0))+x*y*(rho_000-rho_0a0-rho_a00+rho_aa0))
        C3 = a**2*(a**2*(rho_000+rho_00a)+a*(y*(-rho_000-rho_00a+rho_0a0+rho_0aa)+x*(-rho_000-rho_00a+rho_a00+rho_a0a))+x*y*(rho_000+rho_00a-rho_0a0-rho_0aa-rho_a00-rho_a0a+rho_aa0+rho_aaa))
        z  = -(C2-np.sign(C1)*np.sqrt(C2**2+4*C1*C3*les_randoms_[2])*np.sign(C1))/(2*C1)
        
        cat[0,cumu_non_0:cumu_non_0_p1] = x + grille[0]
        cat[1,cumu_non_0:cumu_non_0_p1] = y + grille[1]
        cat[2,cumu_non_0:cumu_non_0_p1] = z + grille[2]
        
    return cat,rho_itp,v_itp


@njit(parallel=True,cache=True)
def discrete_assignment_trilinear_velocity(cat,rho_itp,v_itp,non_0_,Nbr,cumu,les_randoms,rho,v_grid,grid_pos,vertex,N_sample,a,unit):
    '''
    the velocity keywork can not be recognized here if passed, don't know why
    '''
    for i in prange(len(non_0_)):
        non_0         = non_0_[i]
        N_cell        = Nbr [non_0]
        cumu_non_0    = cumu[non_0]
        cumu_non_0_p1 = cumu[non_0+1]
        les_randoms_  = les_randoms[:,cumu_non_0:cumu_non_0_p1]
        grille        = grid_pos[:,non_0]
        
        vtx_0 = vertex[0,non_0]                     ; vtx_1 = vertex[1,non_0]                     ; vtx_2 = vertex[2,non_0]
        vtx_3 = (vtx_0 != (N_sample-1)) * (vtx_0+1) ; vtx_4 = (vtx_1 != (N_sample-1)) * (vtx_1+1) ; vtx_5 = (vtx_2 != (N_sample-1)) * (vtx_2+1)
        
        rho_000 = rho[vtx_0,vtx_1,vtx_2] ; rho_a0a = rho[vtx_3,vtx_1,vtx_5] ; rho_0a0 = rho[vtx_0,vtx_4,vtx_2] ; rho_aa0 = rho[vtx_3,vtx_4,vtx_2]
        rho_a00 = rho[vtx_3,vtx_1,vtx_2] ; rho_0aa = rho[vtx_0,vtx_4,vtx_5] ; rho_00a = rho[vtx_0,vtx_1,vtx_5] ; rho_aaa = rho[vtx_3,vtx_4,vtx_5]
        
        C1 = rho_000+rho_0a0+rho_00a+rho_0aa
        C2 = -rho_000+rho_a00-rho_0a0-rho_00a+rho_a0a-rho_0aa+rho_aa0+rho_aaa
        C3 = rho_000+rho_a00+rho_0a0+rho_00a+rho_a0a+rho_0aa+rho_aa0+rho_aaa
        x  = (a*(-C1+np.sqrt(C1**2+C2*C3*les_randoms_[0])))/C2
        
        C1 = a*(-rho_000-rho_00a+rho_0a0+rho_0aa)+x*(rho_000+rho_00a-rho_0a0-rho_0aa-rho_a00-rho_a0a+rho_aa0+rho_aaa)
        C2 = 2*a**2 *(rho_000+rho_00a)-2*a*x*(rho_000+rho_00a-rho_a00-rho_a0a)
        C3 = a**2*(a*(rho_000+rho_00a+rho_0a0+rho_0aa)+x*(-rho_000-rho_00a-rho_0a0-rho_0aa+rho_a00+rho_a0a+rho_aa0+rho_aaa))
        y  = -(C2-np.sign(C1)*np.sqrt(C2**2+4*C1*C3*les_randoms_[1])*np.sign(C1))/(2*C1)
        
        C1 = a*(a*(-rho_000+rho_00a)+x*(rho_000-rho_00a-rho_a00+rho_a0a)+y*(rho_000-rho_00a-rho_0a0+rho_0aa))+x*y*(-rho_000+rho_00a+rho_0a0-rho_0aa+rho_a00-rho_a0a-rho_aa0+rho_aaa)
        C2 = 2*a*(a*(a*rho_000+x*(-rho_000+rho_a00)+y*(-rho_000+rho_0a0))+x*y*(rho_000-rho_0a0-rho_a00+rho_aa0))
        C3 = a**2*(a**2*(rho_000+rho_00a)+a*(y*(-rho_000-rho_00a+rho_0a0+rho_0aa)+x*(-rho_000-rho_00a+rho_a00+rho_a0a))+x*y*(rho_000+rho_00a-rho_0a0-rho_0aa-rho_a00-rho_a0a+rho_aa0+rho_aaa))
        z  = -(C2-np.sign(C1)*np.sqrt(C2**2+4*C1*C3*les_randoms_[2])*np.sign(C1))/(2*C1)
        
        cat[0,cumu_non_0:cumu_non_0_p1] = x + grille[0]
        cat[1,cumu_non_0:cumu_non_0_p1] = y + grille[1]
        cat[2,cumu_non_0:cumu_non_0_p1] = z + grille[2]
        
        vx_000 = v_grid[0,vtx_0,vtx_1,vtx_2] ; vy_000 = v_grid[1,vtx_0,vtx_1,vtx_2] ; vz_000 = v_grid[2,vtx_0,vtx_1,vtx_2] 
        vx_a00 = v_grid[0,vtx_3,vtx_1,vtx_2] ; vy_a00 = v_grid[1,vtx_3,vtx_1,vtx_2] ; vz_a00 = v_grid[2,vtx_3,vtx_1,vtx_2] 
        vx_0a0 = v_grid[0,vtx_0,vtx_4,vtx_2] ; vy_0a0 = v_grid[1,vtx_0,vtx_4,vtx_2] ; vz_0a0 = v_grid[2,vtx_0,vtx_4,vtx_2] 
        vx_00a = v_grid[0,vtx_0,vtx_1,vtx_5] ; vy_00a = v_grid[1,vtx_0,vtx_1,vtx_5] ; vz_00a = v_grid[2,vtx_0,vtx_1,vtx_5] 
        vx_a0a = v_grid[0,vtx_3,vtx_1,vtx_5] ; vy_a0a = v_grid[1,vtx_3,vtx_1,vtx_5] ; vz_a0a = v_grid[2,vtx_3,vtx_1,vtx_5] 
        vx_0aa = v_grid[0,vtx_0,vtx_4,vtx_5] ; vy_0aa = v_grid[1,vtx_0,vtx_4,vtx_5] ; vz_0aa = v_grid[2,vtx_0,vtx_4,vtx_5] 
        vx_aa0 = v_grid[0,vtx_3,vtx_4,vtx_2] ; vy_aa0 = v_grid[1,vtx_3,vtx_4,vtx_2] ; vz_aa0 = v_grid[2,vtx_3,vtx_4,vtx_2] 
        vx_aaa = v_grid[0,vtx_3,vtx_4,vtx_5] ; vy_aaa = v_grid[1,vtx_3,vtx_4,vtx_5] ; vz_aaa = v_grid[2,vtx_3,vtx_4,vtx_5] 
        
        amx = a-x ; amy = a-y ; amz = a-z ; a3 = a**3
        
        v_of_part_x = vx_000*amx*amy*amz+vx_a00*x*amy*amz+vx_0a0*amx*y*amz+vx_00a*amx*amy*z+vx_a0a*x*amy*z+vx_0aa*amx*y*z+vx_aa0*x*y*amz+vx_aaa*x*y*z
        v_of_part_y = vy_000*amx*amy*amz+vy_a00*x*amy*amz+vy_0a0*amx*y*amz+vy_00a*amx*amy*z+vy_a0a*x*amy*z+vy_0aa*amx*y*z+vy_aa0*x*y*amz+vy_aaa*x*y*z
        v_of_part_z = vz_000*amx*amy*amz+vz_a00*x*amy*amz+vz_0a0*amx*y*amz+vz_00a*amx*amy*z+vz_a0a*x*amy*z+vz_0aa*amx*y*z+vz_aa0*x*y*amz+vz_aaa*x*y*z
        rho_trip    = rho_000*amx*amy*amz+rho_a00*x*amy*amz+rho_0a0*amx*y*amz+rho_00a*amx*amy*z+rho_a0a*x*amy*z+rho_0aa*amx*y*z+rho_aa0*x*y*amz+rho_aaa*x*y*z
        
        v_itp[0,cumu_non_0:cumu_non_0_p1] = v_of_part_x/a3 * unit
        v_itp[1,cumu_non_0:cumu_non_0_p1] = v_of_part_y/a3 * unit
        v_itp[2,cumu_non_0:cumu_non_0_p1] = v_of_part_z/a3 * unit
        rho_itp[cumu_non_0:cumu_non_0_p1] = rho_trip   /a3
    
    return cat,rho_itp,v_itp