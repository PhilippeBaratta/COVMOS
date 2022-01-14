import numpy as np
from os import rename
from os.path import dirname,realpath,exists
import subprocess
import socket
from time import sleep

from tools._velocity_model import *

def save_and_or_analyse_cat(Par,sim_ref,tot_obj,cat,v_cat):
    
    if Par['save_catalogue']: save_catalogue_data(Par,tot_obj,cat,v_cat,sim_ref)
    if Par['estimate_Pk_multipoles'] == 'detached' :
        if not Par['save_catalogue'] : save_catalogue_data(Par,tot_obj,cat,v_cat,sim_ref) # the cat must be saved for detached Pk estimate
        Pk_poles_estimate_detached(Par,sim_ref)
    if Par['estimate_Pk_multipoles'] == 'stopandrun' :
        from tools._Pk_estimate_NBK_attached import Pk_poles_estimate
        Pk_poles_estimate(sim_ref,Par,tot_obj,cat,v_cat)
    return         

def save_catalogue_data(Par,tot_obj,cat,v_cat,sim_ref):
    GADGET = 100*np.sqrt(1+Par['redshift'])
    file = open(sim_ref['sim_name'], mode='wb')
    if not Par['velocity']: 
        np.concatenate(([tot_obj],cat[0,:],cat[1,:],cat[2,:])).astype(np.float32).tofile(file) #.astype(np.float32).tofile(file)
    if Par['velocity']: 
        np.concatenate(([tot_obj],cat[0,:],cat[1,:],cat[2,:],v_cat[0,:]*GADGET,v_cat[1,:]*GADGET,v_cat[2,:]*GADGET)).astype(np.float32).tofile(file)
    file.close
    while not exists(sim_ref['sim_name']):
        sleep(1)
    rename(sim_ref['sim_name'],sim_ref['sim_name']+'.data')
    
def Pk_poles_estimate_detached(Par,sim_ref):
    filename = sim_ref['sim_name']+'.data'
    RSD = Par['velocity']
    redshift = Par['redshift']
    savecat = Par['save_catalogue']
    L = Par['L']
    Omega_m = Par['Omega_m']
    number_ref = sim_ref['number']
    Pk_file     = Par['file_Pk']
    Pk_RSD_file = Par['file_Pk_RSD']
    dir_path = dirname(realpath(__file__))
    pid = subprocess.Popen(['ssh', '-o', 'StrictHostKeyChecking=no',socket.gethostname(),'python','-W','ignore',dir_path+'/_Pk_estimate_NBK_detached.py',filename,str(RSD),str(redshift),str(L),str(savecat),str(Omega_m),str(number_ref),str(Pk_file),str(Pk_RSD_file),'-c','&'],shell=False,stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return

