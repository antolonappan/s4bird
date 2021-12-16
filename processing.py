import os
import healpy as hp
import numpy as np
import plancklens
from plancklens.filt import filt_simple, filt_util
from plancklens import utils
from plancklens import qest, qecl, qresp
from plancklens import nhl
from plancklens.n1 import n1
from plancklens.sims import planck2018_sims, phas, maps, utils as maps_utils
from plancklens.filt import filt_cinv
from simulation import  s4bird_sims_general
from plancklens.helpers import mpi
from delens import Delensing, Pseudo_cl, Efficency
import toml
from lenspyx.utils import camb_clfile2
from likelihood import LH_HL,LH_simple,LH_HL_mod
from covariance import SampleCov, SampleCOV
from libparam import FilteringAndQE


try:
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-delens', dest='delens', action='store_true', help='do Delensing')
    args = parser.parse_args()
    ini = args.inifile[0]
except:
    ini = toml.load('/global/u2/l/lonappan/workspace/S4bird/ini_new/libparam.ini')['file']


ini_dir = '/global/u2/l/lonappan/workspace/S4bird/ini_new'
ini_file = os.path.join(ini_dir,ini)

config = toml.load(ini_file)

# GET CONFIG SECTIONS
of_config = config['OF']
by_config = config['BY']


# FILE CONFIG
base = file_config['base_name']
workbase = file_config['base_folder']
pathbase = os.path.join(workbase,base)



path_final = os.path.join(pathbase,f"SIM_SET{sim_set}")        
map_path = os.path.join(path_final,'Maps')

# CL CONFIG
cl_folder = os.path.join(workbase,fid_config['folder'])
cl_base = fid_config['base']

############################################################


delens_path = os.path.join(path_final, delens_config['folder'])

transfer = transf if bool(delens_config['apply_transf']) else None

delens_lib = Delensing(delens_path,sims,ivfs_raw,qlms_dd,qresp_dd,nhl_dd,n_sims,lmax_qlm,cl_unl['pp'],nside,maskpaths[0],qe_key,transf=transfer,save_template=True,verbose=False)


pseudocl_path = os.path.join(path_final, pseudo_cl_config['folder'])

if pseudo_cl_config['beam'] == 'None':
    beam_pcl = None
else:
    beam_pcl = pseudo_cl_config['beam']
    
pseudocl_lib = Pseudo_cl(pseudocl_path,delens_lib,pseudo_cl_config['mask'],beam=beam_pcl)

"""
eff_path = os.path.join(path_final,eff_config['folder'])


bias_file = os.path.join(pathbase,'GS','Efficency',f'bias_{qe_key}.pkl') if bool(eff_config['bias_do']) else None

    

eff_lib = Efficency(eff_path,pseudocl_lib,n_sims,cl_len['bb'],bool(eff_config['bias_do']),bias_file)

 

if bool(eff_config['save_bias']) and bool(map_config['do_GS']):
    eff_lib.save_bias()


lh_path = os.path.join(path_final,f"{lh_config['folder']}_{qe_key}")

bias_cov_f = os.path.join(pathbase,'GS',f"{lh_config['folder']}_{qe_key}","Covariance",f"bias_cov_{lh_config['lmin']}_{lh_config['lmax']}.pkl")

if lh_config['do']:
    #cov_lib = SampleCov(os.path.join(lh_path,'Covariance'),eff_lib,512,10,
    #                    lh_config['lmin'],lh_config['lmax'],map_config['do_GS'],bias_cov_f)
    
    cov_lib = SampleCOV(os.path.join(lh_path,'Covariance'),pathbase,pseudo_cl_config['folder'],
                        qe_key,n_sims,512,10,lh_config['lmin'],lh_config['lmax'])
    lh_lib = locals()[f"LH_{lh_config['model']}"](lh_path,eff_lib,cov_lib,lh_config['nsamples'],
                                                  cl_len['bb'],nlev_p,map_config['beam'],lh_config['lmin'],
                                                  lh_config['lmax'],bool(lh_config['fit_lensed']),
                                                  base,bool(lh_config['fix_alens']),bool(lh_config['cache']))
    print(f"Likelihood:{lh_lib.name}")

"""
if __name__ == "__main__":
    jobs = np.arange(n_sims)
            
    if args.delens:
         for i in jobs[mpi.rank::mpi.size]:
            print(f"Delensing map-{i} in Processor-{mpi.rank}")
            QU = delens_lib.get_delensed_field(i)
            del QU
    
    if args.cl:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Pure B-Mode of  map-{i} in Processor-{mpi.rank}")
            cl = pseudocl_lib.get_lensed_cl(i)
            del cl
            cl = pseudocl_lib.get_delensed_cl(i)
            del cl
    
    if args.lh:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Running MCMC on map-{i} in Processor-{mpi.rank}")
            pos = lh_lib.posterior(i)
            

