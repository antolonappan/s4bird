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
import toml
from lenspyx.utils import camb_clfile2
from glob import glob






class FilteringAndQE:
    def __init__(self,ini_file,sim_set_overule=None):
        config = toml.load(ini_file)
        # GET CONFIG SECTIONS
        qe_config = config['QE']
        file_config = config['File']
        map_config = config['Map']
        fid_config = config['Fiducial']
        fg_config = config['Foreground']



        # QE CONFIG
        lmax_ivf = qe_config['lmax_ivf']
        lmin_ivf = qe_config['lmin_ivf']  
        lmax_qlm = qe_config['lmax_qlm']
        qe_key = qe_config["key"]
        self.lmax_qlm = lmax_qlm
        self.qe_key  = qe_key
             

        # MAP CONFIG
        nlev_t = map_config['nlev_t']
        nlev_p = map_config['nlev_p']
        nside = map_config['nside']
        n_sims = map_config['nsims']
        maskpaths = [map_config['mask']]
        sim_set = sim_set_overule if sim_set_overule is not None else int(map_config['set'])
        self.sim_set = sim_set
        self.nside = nside
        self.n_sims = n_sims
        self.maskpaths = maskpaths

        # FILE CONFIG
        base = file_config['base_name']
        self.base = base
        workbase = file_config['base_folder']
        pathbase = os.path.join(workbase,base)

        
        do_fg = bool(fg_config['do']) #if sim_set is 1 else False
        
        self.do_fg = do_fg
        
        if do_fg:
            path_final = os.path.join(pathbase,f"SIM_SET{sim_set}_FG")  
        else:
            path_final = os.path.join(pathbase,f"SIM_SET{sim_set}")  
        map_path = os.path.join(path_final,'Maps')

        # CL CONFIG
        cl_folder = os.path.join(workbase,fid_config['folder'])
        cl_base = fid_config['base']

        ############################################################

        TEMP =  os.path.join(path_final,qe_config['folder'])
        self.QE_dir = TEMP

        transf = hp.gauss_beam( map_config['beam']/ 60. / 180. * np.pi, lmax=lmax_ivf)
        self.transf = transf

        cl_unl_fname = os.path.join(cl_folder,f"{cl_base}_lenspotential.dat")
        cl_len_fname = os.path.join(cl_folder,f"{cl_base}_lensed_dls.dat")

        cl_unl = camb_clfile2(cl_unl_fname)
        cl_len = utils.camb_clfile(cl_len_fname)
        cl_weight = utils.camb_clfile(cl_len_fname)
        cl_weight['bb'] *= 0.
        self.cl_unl = cl_unl
        self.cl_len = cl_len


        sims = s4bird_sims_general(nside,map_path)
        self.sims = sims

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
        self.ivfs_raw = ivfs_raw
        
        ftl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
        fel = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
        fbl = np.ones(lmax_ivf + 1, dtype=float) * (np.arange(lmax_ivf + 1) >= lmin_ivf)
        ivfs   = filt_util.library_ftl(ivfs_raw, lmax_ivf, ftl, fel, fbl)
        self.ivfs = ivfs
        qlms_dd = qest.library_sepTP(os.path.join(TEMP, 'qlms_dd'), ivfs, ivfs, cl_len['te'], nside, lmax_qlm=lmax_qlm)

        self.qlms_dd = qlms_dd
        nhl_dd = nhl.nhl_lib_simple(os.path.join(TEMP, qe_config['nhl_dir']), ivfs, cl_weight, lmax_qlm)
        self.nhl_dd = nhl_dd
        qresp_dd = qresp.resp_lib_simple(os.path.join(TEMP, qe_config['qresp_dir']), lmax_ivf, cl_weight, cl_len,
                                         {'t': ivfs.get_ftl(), 'e':ivfs.get_fel(), 'b':ivfs.get_fbl()}, lmax_qlm)
        self.qresp_dd = qresp_dd
        
        qcls_dd = qecl.library(os.path.join(TEMP, 'qcls_dd'), qlms_dd, qlms_dd, np.arange(400,1000))
        self.qcls_dd = qcls_dd
        
        # ss_dict = { k : v for k, v in zip( np.concatenate( [ range(i*60, (i+1)*60) for i in range(0,5) ] ),
        #             np.concatenate( [ np.roll( range(i*60, (i+1)*60), -1 ) for i in range(0,5) ] ) ) }
        # ivfs_s = filt_util.library_shuffle(ivfs, ss_dict)
        # qlms_ss = qest.library_sepTP(os.path.join(TEMP, 'qlms_ss'), ivfs,ivfs_s, cl_len['te'], nside,lmax_qlm=lmax_qlm)
        # qcls_ss = qecl.library(os.path.join(TEMP, 'qcls_ss'), qlms_ss, qlms_ss,np.array([]))
        # self.qcls_ss = qcls_ss


if __name__ == "__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-ivt', dest='ivt', action='store_true', help='do T. filtering')
    parser.add_argument('-ivp', dest='ivp', action='store_true', help='do P. filtering')
    parser.add_argument('-missing', dest='missing',action='store_true', help='only do missing')
    parser.add_argument('-set',dest='set',action='store',type=int,default=None)
    parser.add_argument('-dd', dest='dd', action='store_true', help='perform dd qlms')
    parser.add_argument('-qclss', dest='qclss', action='store_true', help='perform qcls ss')
    args = parser.parse_args()
    ini = args.inifile[0]
    
    ini_dir = '/global/u2/l/lonappan/workspace/s4bird/s4bird/ini'
    ini_file = os.path.join(ini_dir,ini)

    fqe = FilteringAndQE(ini_file,args.set)
    jobs = np.arange(fqe.n_sims)
    
    if args.missing:
        pfiles = glob(f"{fqe.QE_dir}/ivfs/*elm.fits")
        tfiles = glob(f"{fqe.QE_dir}/ivfs/*tlm.fits")
        m_p_idx = [int(pfile.split('_')[2]) for pfile in pfiles]
        m_p = [i for i in range(fqe.n_sims) if i not in m_p_idx]
        m_t_idx = [int(tfile.split('_')[2]) for tfile in tfiles]
        m_t = [i for i in range(fqe.n_sims) if i not in m_t_idx]
        print("Polarization missing idx:", m_p, f"Length:{len(m_p)}")
        print("Temperature missing idx:", m_t, f"Length:{len(m_t)}")
        
    if args.ivt:
        
        if args.missing:
            jobs = np.array(m_t)
            assert len(m_t) <= mpi.size
        
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Filtering temperature Map-{i} in Processor-{mpi.rank}")
            tlm = fqe.ivfs.get_sim_tlm(i)
            del tlm
    
    if args.ivp:
        if args.missing:
            jobs = np.array(m_p)
            assert len(m_p) <= mpi.size
            
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Filtering polarization Map-{i} in Processor-{mpi.rank}")
            elm = fqe.ivfs.get_sim_elm(i)
            del elm
    
    if args.dd:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Making QE-{i} in Processor-{mpi.rank}")
            qlm = fqe.qlms_dd.get_sim_qlm(fqe.qe_key,i)
            del qlm
            
    if args.qclss:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Making qcls-{i} in Processor-{mpi.rank}")
            qcls = fqe.qcls_ss.get_sim_qcl(fqe.qe_key,i)
            del qcls
            

            
