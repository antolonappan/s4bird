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
from likelihood import LH_HL,LH_simple,LH_HL_mod
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

file_config = config['File']
delens_config = config['Delens']
pseudo_cl_config = config['Pseudo_cl']
fid_config = config['Fiducial']
eff_config = config['Efficency']
map_config = config['Map']
lh_config = config['Likelihood']

base = file_config['base_name']
workbase = file_config['base_folder']
if bool(map_config['do_GS']):
    pathbase = os.path.join(workbase,base,'GS')
else:
    pathbase = os.path.join(workbase,base,'RS')

#combination
qe_config = config['QE']
comb_config = config['Combination']
qe_key = qe_config["key"]

#Fiducial
cl_folder = os.path.join(workbase,fid_config['folder'])
cl_base = fid_config['base']
cl_unl_fname = os.path.join(cl_folder,f"{cl_base}_lenspotentialCls.dat")
cl_len_fname = os.path.join(cl_folder,f"{cl_base}_lensedCls.dat")


#LiteBird
qe_config_LB = config['QE_LB']
map_config_LB = config['Map_LB']
file_config_LB = config['File_LB']

lmax_ivf_LB = qe_config_LB['lmax_ivf']
lmin_ivf_LB = qe_config_LB['lmin_ivf']  
lmax_qlm_LB = qe_config_LB['lmax_qlm']
qe_key_LB = qe_config_LB["key"]

nlev_t_LB = map_config_LB['nlev_t']
nlev_p_LB = map_config_LB['nlev_p']
nside_LB = map_config_LB['nside']
n_sims_LB = map_config['nsims']


base_LB = file_config_LB['base_name']
workbase_LB = file_config_LB['base_folder']

if bool(map_config['do_GS']):
    pathbase_LB = os.path.join(workbase_LB,base_LB,'GS')
else:
    pathbase_LB = os.path.join(workbase_LB,base_LB,'RS')

map_path_LB = os.path.join(pathbase_LB,'Maps')

maskpaths_LB = [map_config_LB['mask']]

TEMP_LB =  os.path.join(pathbase_LB,qe_config_LB['folder'])

transf_LB = hp.gauss_beam(map_config_LB['beam']/ 60. / 180. * np.pi, lmax=lmax_ivf_LB)


cl_unl = utils.camb_clfile(cl_unl_fname)
cl_len = utils.camb_clfile(cl_len_fname)
cl_weight = utils.camb_clfile(cl_len_fname)
cl_weight['bb'] *= 0.



sims_LB = s4bird_sims_general(nside_LB,map_path_LB)


libdir_cinvt_LB = os.path.join(TEMP_LB, 'cinv_t')
libdir_cinvp_LB = os.path.join(TEMP_LB, 'cinv_p')
libdir_ivfs_LB  = os.path.join(TEMP_LB, 'ivfs')
ninv_t_LB = [np.array([3. / nlev_t_LB ** 2])] + maskpaths_LB
cinv_t_LB = filt_cinv.cinv_t(libdir_cinvt_LB, lmax_ivf_LB,nside_LB, cl_len, transf_LB, ninv_t_LB,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p_LB = [[np.array([3. / nlev_p_LB ** 2])] + maskpaths_LB]
cinv_p_LB = filt_cinv.cinv_p(libdir_cinvp_LB, lmax_ivf_LB, nside_LB, cl_len, transf_LB, ninv_p_LB)

ivfs_raw_LB    = filt_cinv.library_cinv_sepTP(libdir_ivfs_LB, sims_LB, cinv_t_LB, cinv_p_LB, cl_len)

ftl_LB = np.ones(lmax_ivf_LB + 1, dtype=float) * (np.arange(lmax_ivf_LB + 1) >= lmin_ivf_LB)
fel_LB = np.ones(lmax_ivf_LB + 1, dtype=float) * (np.arange(lmax_ivf_LB + 1) >= lmin_ivf_LB)
fbl_LB = np.ones(lmax_ivf_LB + 1, dtype=float) * (np.arange(lmax_ivf_LB + 1) >= lmin_ivf_LB)
ivfs_LB   = filt_util.library_ftl(ivfs_raw_LB, lmax_ivf_LB, ftl_LB, fel_LB, fbl_LB)

##################################################################################

#CMB_S4

qe_config_S4 = config['QE_S4']
map_config_S4 = config['Map_S4']
file_config_S4 = config['File_S4']

base_S4 = file_config_S4['base_name']
workbase_S4 = file_config_S4['base_folder']
if bool(map_config['do_GS']):
    pathbase_S4 = os.path.join(workbase_S4,base_S4,'GS')
else:
    pathbase_S4 = os.path.join(workbase_S4,base_S4,'RS')
    
map_path_S4 = os.path.join(pathbase_S4,'Maps')
maskpaths_S4 = [map_config_S4['mask']]

if not bool(comb_config['do']):
    lmax_ivf_S4 = qe_config_S4['lmax_ivf']
    lmin_ivf_S4 = qe_config_S4['lmin_ivf']  
    lmax_qlm_S4 = qe_config_S4['lmax_qlm']
    qe_key_S4 = qe_config_S4["key"]

    nlev_t_S4 = map_config_S4['nlev_t']
    nlev_p_S4 = map_config_S4['nlev_p']
    nside_S4 = map_config_S4['nside']
    n_sims_S4= map_config['nsims']


    TEMP_S4 =  os.path.join(pathbase_S4,qe_config_S4['folder'])

    transf_S4 = hp.gauss_beam( map_config_S4['beam']/ 60. / 180. * np.pi, lmax=lmax_ivf_S4)
    sims_S4 = s4bird_sims_general(nside_S4,map_path_S4)
else:
    lmax_ivf_S4 = qe_config['lmax_ivf']
    lmin_ivf_S4 = qe_config['lmin_ivf']  
    lmax_qlm_S4 = qe_config['lmax_qlm']
    qe_key_S4 = qe_config["key"]

    nlev_t_S4 = map_config['nlev_t']
    nlev_p_S4 = map_config['nlev_p']
    nside_S4 = map_config['nside']
    n_sims_S4= map_config['nsims']
    
    TEMP_S4 =  os.path.join(pathbase,qe_config['folder'])
    sims_S4 = s4bird_sims_general(nside_S4,map_path_LB,map_path_S4,comb_config['weights'],
                                  [map_config_LB['beam'],map_config_S4['beam']],map_config['beam'])
    transf_S4 = sims_S4.fl


libdir_cinvt_S4 = os.path.join(TEMP_S4, 'cinv_t')
libdir_cinvp_S4 = os.path.join(TEMP_S4, 'cinv_p')
libdir_ivfs_S4  = os.path.join(TEMP_S4, 'ivfs')
ninv_t_S4 = [np.array([3. / nlev_t_S4 ** 2])] + maskpaths_S4
cinv_t_S4 = filt_cinv.cinv_t(libdir_cinvt_S4, lmax_ivf_S4,nside_S4, cl_len, transf_S4, ninv_t_S4,
                        marge_monopole=True, marge_dipole=True, marge_maps=[])

ninv_p_S4 = [[np.array([3. / nlev_p_S4 ** 2])] + maskpaths_S4]
cinv_p_S4 = filt_cinv.cinv_p(libdir_cinvp_S4, lmax_ivf_S4, nside_S4, cl_len, transf_S4, ninv_p_S4)

ivfs_raw_S4    = filt_cinv.library_cinv_sepTP(libdir_ivfs_S4, sims_S4, cinv_t_S4, cinv_p_S4, cl_len)

ftl_S4 = np.ones(lmax_ivf_S4 + 1, dtype=float) * (np.arange(lmax_ivf_S4 + 1) >= lmin_ivf_S4)
fel_S4 = np.ones(lmax_ivf_S4 + 1, dtype=float) * (np.arange(lmax_ivf_S4 + 1) >= lmin_ivf_S4)
fbl_S4 = np.ones(lmax_ivf_S4 + 1, dtype=float) * (np.arange(lmax_ivf_S4 + 1) >= lmin_ivf_S4)
ivfs_S4   = filt_util.library_ftl(ivfs_raw_S4, lmax_ivf_S4, ftl_S4, fel_S4, fbl_S4)


##################################################################################
qlms_dd_S4 = qest.library_sepTP(os.path.join(TEMP_S4, 'qlms_dd'), ivfs_S4, ivfs_S4,   cl_len['te'],
                                nside_S4, lmax_qlm=lmax_qlm_S4)

nhl_dd_S4 = nhl.nhl_lib_simple(os.path.join(TEMP_S4, 'nhl_dd'), ivfs_S4, cl_weight, lmax_qlm_S4)

qresp_dd_S4 = qresp.resp_lib_simple(os.path.join(TEMP_S4, 'qresp'), lmax_ivf_S4, cl_weight, cl_len,
                                 {'t': ivfs_S4.get_ftl(), 'e':ivfs_S4.get_fel(), 'b':ivfs_S4.get_fbl()},
                                    lmax_qlm_S4)
#################################################################################




delens_path = os.path.join(pathbase, delens_config['folder'])

transfer = transf_LB if bool(delens_config['apply_transf']) else None

delens_lib = Delensing(delens_path,sims_LB,ivfs_raw_LB,qlms_dd_S4,
                       qresp_dd_S4,nhl_dd_S4,n_sims_S4,lmax_qlm_S4,cl_unl['pp'],
                       nside_LB,maskpaths_S4[0],qe_key_S4,transf=transfer,
                       save_template=True,verbose=False)


pseudocl_path = os.path.join(pathbase, pseudo_cl_config['folder'])

if pseudo_cl_config['beam'] == 'None':
    beam_pcl = None
else:
    beam_pcl = pseudo_cl_config['beam']
    
pseudocl_lib = Pseudo_cl(pseudocl_path,delens_lib,pseudo_cl_config['mask'],beam=beam_pcl)


eff_path = os.path.join(pathbase,eff_config['folder'])

bias_file = os.path.join(workbase,base,'GS','Efficency','bias.pkl') if bool(eff_config['bias_do']) else None

eff_lib = Efficency(eff_path,pseudocl_lib,n_sims_S4,cl_len['bb'],bool(eff_config['bias_do']),bias_file)

if bool(eff_config['save_bias']) and bool(map_config['do_GS']):
    eff_lib.save_bias()

    


lh_path = os.path.join(pathbase,f"{lh_config['folder']}_{qe_key_LB}")
if lh_config['do']:
    cov_lib = SampleCov(os.path.join(lh_path,'Covariance'),eff_lib,512,10,
                        lh_config['lmin'],lh_config['lmax'])
    lh_lib = locals()[f"LH_{lh_config['model']}"](lh_path,eff_lib,cov_lib,lh_config['nsamples'],
                                              cl_len['bb'],nlev_p_LB,map_config_LB['beam'],lh_config['lmin'],
                                              lh_config['lmax'],bool(lh_config['fit_lensed']),
                                              base,bool(lh_config['fix_alens']),bool(lh_config['cache']))

if __name__ == "__main__":
    jobs = np.arange(n_sims_S4)
    if args.ivt:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Filtering temperature Map-{i} in Processor-{mpi.rank}")
            tlm = ivfs_S4.get_sim_tlm(i)
            del tlm
    
    if args.ivp:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Filtering polarization Map-{i} in Processor-{mpi.rank}")
            elm = ivfs_S4.get_sim_elm(i)
            del elm
    
    if args.dd:
         for i in jobs[mpi.rank::mpi.size]:
            print(f"Making QE-{i} in Processor-{mpi.rank}")
            qlm = qlms_dd_S4.get_sim_qlm(qe_key,i)
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
            r = lh_lib.posterior(i)
            