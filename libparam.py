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

from likelihood import LH_HL,LH_simple
from covariance import SampleCov


try:
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-ivt', dest='ivt', action='store_true', help='do T. filtering')
    parser.add_argument('-ivp', dest='ivp', action='store_true', help='do P. filtering')
    parser.add_argument('-dd', dest='dd', action='store_true', help='perform dd qlms')
    parser.add_argument('-delens', dest='delens', action='store_true', help='do Delensing')
    parser.add_argument('-cl', dest='cl', action='store_true', help='perform psuedo cls')
    parser.add_argument('-lh', dest='lh', action='store_true', help='run mcmc')
    args = parser.parse_args()
    ini = args.inifile[0]
except:
    ini = toml.load('/global/u2/l/lonappan/workspace/S4bird/ini/libparam.ini')['file']


ini_dir = '/global/u2/l/lonappan/workspace/S4bird/ini'
ini_file = os.path.join(ini_dir,ini)

config = toml.load(ini_file)

# GET CONFIG SECTIONS
qe_config = config['QE']
file_config = config['File']
map_config = config['Map']
delens_config = config['Delens']
pseudo_cl_config = config['Pseudo_cl']
fid_config = config['Fiducial']
eff_config = config['Efficency']
lh_config = config['Likelihood']

# QE CONFIG
lmax_ivf = qe_config['lmax_ivf']
lmin_ivf = qe_config['lmin_ivf']  
lmax_qlm = qe_config['lmax_qlm']
qe_key = qe_config["key"]

# MAP CONFIG
nlev_t = map_config['nlev_t']
nlev_p = map_config['nlev_p']
nside = map_config['nside']
n_sims = map_config['nsims']
maskpaths = [map_config['mask']]

# FILE CONFIG
base = file_config['base_name']
workbase = file_config['base_folder']
pathbase = os.path.join(workbase,base)

if bool(map_config['do_GS']):
    path_final = os.path.join(pathbase,'GS')
else:
    path_final = os.path.join(pathbase,'RS')        
map_path = os.path.join(path_final,'Maps')

# CL CONFIG
cl_folder = os.path.join(workbase,fid_config['folder'])
cl_base = fid_config['base']

############################################################

TEMP =  os.path.join(path_final,qe_config['folder'])

transf = hp.gauss_beam( map_config['beam']/ 60. / 180. * np.pi, lmax=lmax_ivf)

cl_unl_fname = os.path.join(cl_folder,f"{cl_base}_lenspotentialCls.dat")
cl_len_fname = os.path.join(cl_folder,f"{cl_base}_lensedCls.dat")

cl_unl = utils.camb_clfile(cl_unl_fname)
cl_len = utils.camb_clfile(cl_len_fname)
cl_weight = utils.camb_clfile(cl_len_fname)
cl_weight['bb'] *= 0.



sims = s4bird_sims_general(nside,map_path)

##################################################################

libdir_cinvt = os.path.join(TEMP, 'cinv_t')
libdir_cinvp = os.path.join(TEMP, 'cinv_p')
libdir_ivfs  = os.path.join(TEMP, 'ivfs')
ninv_t = [np.array([3. / nlev_t ** 2])] + maskpaths
cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf,nside, cl_len, transf, ninv_t,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p = [[np.array([3. / nlev_p ** 2])] + maskpaths]
cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_p)

ivfs_raw    = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cl_len)

ftl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
fel = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
fbl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl, fel, fbl)

qlms_dd = qest.library_sepTP(os.path.join(TEMP, 'qlms_dd'), ivfs, ivfs,   cl_len['te'], nside, lmax_qlm=lmax_qlm)

nhl_dd = nhl.nhl_lib_simple(os.path.join(TEMP, 'nhl_dd'), ivfs, cl_weight, lmax_qlm)

qresp_dd = qresp.resp_lib_simple(os.path.join(TEMP, 'qresp'), lmax_ivf, cl_weight, cl_len,
                                 {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)

delens_path = os.path.join(path_final, delens_config['folder'])

transfer = transf if bool(delens_config['apply_transf']) else None

delens_lib = Delensing(delens_path,sims,ivfs_raw,qlms_dd,qresp_dd,nhl_dd,n_sims,lmax_qlm,cl_unl['pp'],nside,maskpaths[0],qe_key,transf=transfer,save_template=True,verbose=False)


pseudocl_path = os.path.join(path_final, pseudo_cl_config['folder'])

if pseudo_cl_config['beam'] == 'None':
    beam_pcl = None
else:
    beam_pcl = pseudo_cl_config['beam']
    
pseudocl_lib = Pseudo_cl(pseudocl_path,delens_lib,pseudo_cl_config['mask'],beam=beam_pcl)


eff_path = os.path.join(path_final,eff_config['folder'])


bias_file = os.path.join(pathbase,'GS','Efficency','bias.pkl') if bool(eff_config['bias_do']) else None

    

eff_lib = Efficency(eff_path,pseudocl_lib,n_sims,cl_len['bb'],bool(eff_config['bias_do']),bias_file)

 

if bool(eff_config['save_bias']) and bool(map_config['do_GS']):
    eff_lib.save_bias()


lh_path = os.path.join(path_final,f"{lh_config['folder']}_{qe_key}")
print(lh_path)
if lh_config['do']:
    cov_lib = SampleCov(os.path.join(lh_path,'Covariance'),eff_lib,512,10,
                        lh_config['lmin'],lh_config['lmax'])
    lh_lib = locals()[f"LH_{lh_config['model']}"](lh_path,eff_lib,cov_lib,lh_config['nsamples'],
                                                  cl_len['bb'],nlev_p,map_config['beam'],lh_config['lmin'],
                                                  lh_config['lmax'],bool(lh_config['fit_lensed']),
                                                  base,bool(lh_config['fix_alens']),bool(lh_config['cache']))


if __name__ == "__main__":
    jobs = np.arange(n_sims)
    if args.ivt:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Filtering temperature Map-{i} in Processor-{mpi.rank}")
            tlm = ivfs.get_sim_tlm(i)
            del tlm
    
    if args.ivp:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Filtering polarization Map-{i} in Processor-{mpi.rank}")
            elm = ivfs.get_sim_elm(i)
            del elm
    
    if args.dd:
         for i in jobs[mpi.rank::mpi.size]:
            print(f"Making QE-{i} in Processor-{mpi.rank}")
            qlm = qlms_dd.get_sim_qlm(qe_key,i)
            del qlm
            
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
            
            
        
