import numpy as np

def loadcatalogue(filename,velocity=False):
    '''
    load the positions in Mpc/h and velocities in km/s/(1+z) of a COVMOS catalogue  
    '''
    myarray = np.fromfile(filename, dtype=np.float32)
    myarray = np.delete(myarray,0)
    if   velocity == False : nbrarray = 3
    elif velocity == True  : nbrarray = 6
    catalog = np.reshape(myarray,(nbrarray,int(len(myarray)/nbrarray)))
    x = catalog[0] ; y = catalog[1] ; z = catalog[2]
    if velocity == True:
        vx = catalog[3] ; vy = catalog[4] ; vz = catalog[5]
    if velocity == False:
        return x,y,z
    elif velocity == True:
        return x,y,z,vx,vy,vz