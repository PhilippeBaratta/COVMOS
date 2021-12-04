'''
This simple code helps the user to compute the rms value on his own data to put in the targeted_rms variable (see setting_example.ini)
'''

import numpy as np

def load_velocities_from_user_data():
    '''
    Please complete this function in order to load 3 velocity (numpy) arrays of your data. Unit must be provided in Mpc/h
    '''
    
    return vx,vy,vz

vx,vy,vz = load_velocities_from_user_data()

var_vx = np.var(vx)
var_vy = np.var(vy)
var_vz = np.var(vz)

mean_var_vi = np.mean([var_vx,var_vy,var_vz])

rms = np.sqrt(mean_var_vi)

print('The rms value to put in the targeted_rms variable (see setting_example.ini) is', rms/100)