import numpy as np
import os
from plancklens.helpers import mpi


save_to_scratch = True

scratch = os.environ['SCRATCH']
scratch = scratch if scratch[-1] =='/' else scratch +'/'
scratch = scratch if save_to_scratch else ''


################## Definitions ##################

class probe():
    def __init__(self,name, freq, beam, depth_i=None, depth_p=None):
        self.name = name
        
        if type(freq) == list:
            assert(len(freq)==len(beam))
        else:
            freq, beam, = [freq], [beam]
        self.frequency = np.array(freq)
        self.fwhm = np.array(beam)
        
        if ((depth_i == None) and (depth_p == None)):
            raise ValueError
        elif depth_i == None:
            self.depth_p = np.array(depth_p) if type(depth_p) == list else np.array([depth_p])
            self.depth_i = self.depth_p / np.sqrt(2)
        elif depth_p == None:
            self.depth_i = np.array(depth_i) if type(depth_i) == list else np.array([depth_i])
            self.depth_p = self.depth_i * np.sqrt(2)
        else:
            self.depth_p = np.array(depth_p) if type(depth_p) == list else np.array([depth_p])
            self.depth_i = np.array(depth_i) if type(depth_i) == list else np.array([depth_i])
        assert(len(self.frequency)==len(self.depth_i)==len(self.depth_p))
            

def make_if_not_exist(foldertree):
    if not os.path.exists(foldertree):
        mpi.rprint(f"CONFIG INFO: Making {foldertree}")
        os.makedirs(foldertree)
################ QE Reconstruction ###############
lmax_ivf = 2048
lmin_ivf = 2  
lmax_qlm =4096 
nside = 2048
nlev_t = 1. 
nlev_p = 1.5 
nsims = 25

idx = 0
qe_dir = 'Data/QE/'
qe_dir = qe_dir if not save_to_scratch else scratch+qe_dir
qe_root = 'phi_phi_from_cmbs4'
qe_prefix = qe_dir+qe_root

remove_temp = True

##################### CMB-S4 ####################
cmbs4_freq = [ 20.,  27.,  39.,  93., 145., 225., 280.]
cmbs4_depth_p = [30, 30.8, 17.6, 8.0, 10.0, 22.0, 54.0]
cmbs4_fwhm = [11,7.4, 5.1, 2.9, 2.8, 9.8, 23.6]
probe_cmbs4 = probe('CMBS4',cmbs4_freq ,cmbs4_fwhm,depth_p=cmbs4_depth_p)

cmbs4_No_of_maps = 100

cmbs4_map_dir = 'Data/maps/CMBONLY_GALCUT1/'
cmbs4_map_dir = cmbs4_map_dir if not save_to_scratch else scratch+cmbs4_map_dir
cmbs4_map_root = 'cmb_'
cmbs4_map_prefix = cmbs4_map_dir+cmbs4_map_root


cmbs4_cl_dir = '../../Data/camb/CMBS4/'
cmbs4_cl_root = 's4bird'
cmbs4_cl_prefix = cmbs4_cl_dir+cmbs4_cl_root

instrument_cmbs4={
    "frequency": probe_cmbs4.frequency,
    "depth_p" : probe_cmbs4.depth_p,
    "depth_i" : probe_cmbs4.depth_i,
}

################### LiteBird ######################
probe_litebird = probe('LiteBird',140.,23.0,4.2,5.9)

litebird_No_of_maps = 1

litebird_map_dir =  'Data/maps/LiteBird/'
litebird_map_dir = litebird_map_dir if not save_to_scratch else scratch+litebird_map_dir
litebird_map_root = 's4bird'
litebird_map_prefix = litebird_map_dir+litebird_map_root

litebird_cl_dir = 'Data/camb/LiteBird/'
litebird_cl_root = 's4bird'
litebird_cl_prefix = litebird_cl_dir+litebird_cl_root

instrument_litebird={
    "frequency": probe_litebird.frequency,
    "depth_p" : probe_litebird.depth_p,
    "depth_i" : probe_litebird.depth_i,
}
############### Delensing ######################
delen_dir = 'Data/delens/'
delen_root = 'delens'
delen_prefix = delen_dir+delen_root

lmax = 500  # desired lmax of the (de)lensed field.
dlmax = 2048  # lmax of the (un)lensed fields is lmax + dlmax.  (some buffer is required for accurate lensing at lmax)
#nside = 2048 # The lensed tlm's are computed with healpy map2alm from a lensed map at resolution 'nside_lens'
facres = -1


############## Mask ###########################
mask_dir = '../../Data/masks/'
mask_root = 'cmbs4'
mask_prefix = mask_dir+mask_root
mask_percent = 60
overwrite = True

