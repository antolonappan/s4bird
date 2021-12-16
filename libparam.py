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






class FilteringAndQE:
    def __init__(self,ini_file,sim_set_overule=None):
        config = toml.load(ini_file)
        # GET CONFIG SECTIONS
        qe_config = config['QE']
        file_config = config['File']
        map_config = config['Map']

        # QE CONFIG
        lmax_ivf = qe_config['lmax_ivf']
        lmin_ivf = qe_config['lmin_ivf']  
        self.lmax_qlm = qe_config['lmax_qlm']
        self.qe_key = qe_config["key"]

        # MAP CONFIG
        nlev_t = map_config['nlev_t']
        nlev_p = map_config['nlev_p']
        self.nside = map_config['nside']
        self.n_sims = map_config['nsims']
        self.maskpaths = [map_config['mask']]
        sim_set = sim_set_overule if sim_set_overule is not None else int(map_config['set'])

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

        TEMP =  os.path.join(path_final,qe_config['folder'])

        transf = hp.gauss_beam( map_config['beam']/ 60. / 180. * np.pi, lmax=lmax_ivf)

        cl_unl_fname = os.path.join(cl_folder,f"{cl_base}_scal_dls.dat")
        cl_len_fname = os.path.join(cl_folder,f"{cl_base}_lensed_dls.dat")

        cl_unl = camb_clfile2(cl_unl_fname)
        cl_len = utils.camb_clfile(cl_len_fname)
        cl_weight = utils.camb_clfile(cl_len_fname)
        cl_weight['bb'] *= 0.



        self.sims = s4bird_sims_general(nside,map_path)

        ##################################################################

        libdir_cinvt = os.path.join(TEMP, 'cinv_t')
        libdir_cinvp = os.path.join(TEMP, 'cinv_p')
        libdir_ivfs  = os.path.join(TEMP, 'ivfs')
        ninv_t = [np.array([3. / nlev_t ** 2])] + maskpaths
        cinv_t = filt_cinv.cinv_t(libdir_cinvt, lmax_ivf,nside, cl_len, transf, ninv_t,
                                marge_monopole=True, marge_dipole=True, marge_maps=[])

        ninv_p = [[np.array([3. / nlev_p ** 2])] + maskpaths]
        cinv_p = filt_cinv.cinv_p(libdir_cinvp, lmax_ivf, nside, cl_len, transf, ninv_p)

        self.ivfs_raw    = filt_cinv.library_cinv_sepTP(libdir_ivfs, sims, cinv_t, cinv_p, cl_len)

        ftl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
        fel = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
        fbl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
        self.ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl, fel, fbl)

        self.qlms_dd = qest.library_sepTP(os.path.join(TEMP, 'qlms_dd'), ivfs, ivfs,   cl_len['te'], nside, lmax_qlm=lmax_qlm)


        self.nhl_dd = nhl.nhl_lib_simple(os.path.join(TEMP, qe_config['nhl_dir']), ivfs, cl_weight, lmax_qlm)

        self.qresp_dd = qresp.resp_lib_simple(os.path.join(TEMP, qe_config['qresp_dir']), lmax_ivf, cl_weight, cl_len,
                                         {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)


if __name__ == "__main__":
    
    
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-ivt', dest='ivt', action='store_true', help='do T. filtering')
    parser.add_argument('-ivp', dest='ivp', action='store_true', help='do P. filtering')
    parser.add_argument('-dd', dest='dd', action='store_true', help='perform dd qlms')
    args = parser.parse_args()
    ini = args.inifile[0]


    ini_dir = '/global/u2/l/lonappan/workspace/S4bird/ini_new'
    ini_file = os.path.join(ini_dir,ini)

    fqe = FilteringAndQE(ini_file)
    jobs = np.arange(fqe.n_sims)
    if args.ivt:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Filtering temperature Map-{i} in Processor-{mpi.rank}")
            tlm = fqe.ivfs.get_sim_tlm(i)
            del tlm
    
    if args.ivp:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Filtering polarization Map-{i} in Processor-{mpi.rank}")
            elm = fqe.ivfs.get_sim_elm(i)
            del elm
    
    if args.dd:
         for i in jobs[mpi.rank::mpi.size]:
            print(f"Making QE-{i} in Processor-{mpi.rank}")
            qlm = fqe.qlms_dd.get_sim_qlm(qe_key,i)
            del qlm
