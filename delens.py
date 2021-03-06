import os
import numpy as np
from plancklens.helpers import mpi
from library import Delensing, Pseudo_cl, Efficency
import toml
from quest import FilteringAndQE
from glob import glob



class DelensAndCl:
    def __init__(self,ini):
        ini_dir = '/global/u2/l/lonappan/workspace/s4bird/s4bird/ini/'
        
        if type(ini) == str:
            ini_file = os.path.join(ini_dir,ini)
            config = toml.load(ini_file)
        elif type(ini) == dict:
            config = ini
            if list(config.keys()) != ['File','OF','BY','Delens','Pseudo_cl']:
                raise ValueError("Insufficent keys in the given dict, I need ['File','OF','BY','Delens','Pseudo_cl'] configurations" )




        # GET CONFIG SECTIONS
        file_config = config['File']
        of_config = config['OF']
        by_config = config['BY']
        delens_config = config['Delens']
        pseudo_cl_config = config['Pseudo_cl']

        # FILE CONFIG
        base = file_config['base_name']
        workbase = file_config['base_folder']
        pathbase = os.path.join(workbase,base)

        of_fqe = FilteringAndQE(os.path.join(ini_dir,of_config['ini']),int(of_config['set']))
        by_fqe = FilteringAndQE(os.path.join(ini_dir,by_config['ini']),int(by_config['set']))
        print(f"Delensing {of_fqe.base}-SET{of_fqe.sim_set} with {by_fqe.base}-SET{by_fqe.sim_set}")

        assert of_fqe.do_fg == by_fqe.do_fg
        self.do_fg = of_fqe.do_fg
        
        if self.do_fg:
            path_final = os.path.join(pathbase,f"SIM_SET{of_fqe.sim_set}_FG") 
        else:
            path_final = os.path.join(pathbase,f"SIM_SET{of_fqe.sim_set}")   
        

        ############################################################


        delens_path = os.path.join(path_final, delens_config['folder'])

        #assert of_fqe.maskpaths[0] == by_fqe.maskpaths[0]

        sims = of_fqe.sims

        if delens_config['template']=='of':
            ivfs = of_fqe.ivfs
            print(f"Template is constructed using {of_fqe.base}-SET{of_fqe.sim_set}")
        elif delens_config['template']=='by':
            if (of_config['ini'] != by_config['ini']) and (of_config['set'] != by_config['set']):
                by_fqe_special = FilteringAndQE(os.path.join(ini_dir,by_config['ini']),int(of_config['set']))
                ivfs = by_fqe_special.ivfs
                print(f"Template is constructed using {by_fqe.base}-SET{by_fqe_special.sim_set}")
            else:
                ivfs= by_fqe.ivfs
                print(f"Template is constructed using {by_fqe.base}-SET{by_fqe.sim_set}")
        else:
            raise ValueError

        transf_of = of_fqe.transf
        qlms_dd = by_fqe.qlms_dd
        qresp_dd = by_fqe.qresp_dd
        nhl_dd = by_fqe.nhl_dd
        n_sims = by_fqe.n_sims
        self.n_sims = n_sims
        lmax_qlm = by_fqe.lmax_qlm
        cl_unl = by_fqe.cl_unl
        nside = by_fqe.nside
        maskpaths = by_fqe.maskpaths
        qe_key = by_fqe.qe_key

        self.delens_lib = Delensing(delens_path,sims,ivfs,qlms_dd,qresp_dd,nhl_dd,n_sims,lmax_qlm,cl_unl['pp'],nside,maskpaths[0],qe_key,delens_config['template'],transf=transf_of,save_template=False,verbose=False)

        pcl_dir = pseudo_cl_config['folder']
        pcl_beam = pseudo_cl_config['beam']
        pcl_nside = int(pseudo_cl_config['nside'])
        pcl_binsize = int(pseudo_cl_config['binsize'])
        pcl_apo_scale = int(pseudo_cl_config['apo_scale'])
        pcl_apo_method = pseudo_cl_config['apo_method']
        pcl_fsky = pseudo_cl_config['fsky']
        pcl_purify_b = pseudo_cl_config['purify_b']
        pcl_maskbase = pseudo_cl_config['maskbase']

        pcl_path = os.path.join(path_final, pcl_dir)
        self.pseudocl_lib = Pseudo_cl(pcl_path,self.delens_lib,pcl_maskbase,pcl_beam,pcl_purify_b,pcl_nside,pcl_fsky,pcl_binsize,pcl_apo_scale,pcl_apo_method,delens_config['template'])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-delens', dest='delens', action='store_true', help='do Delensing')
    parser.add_argument('-cl', dest='cl', action='store_true', help='perform psuedo cls')
    parser.add_argument('-missing', dest='missing',action='store_true', help='only do missing')
    args = parser.parse_args()
    ini = args.inifile[0]
    

    
    dap = DelensAndCl(ini)
    
    jobs = np.arange(dap.n_sims)
    if args.missing:
        qfiles = glob(f"{dap.delens_lib.lib_dir}/*Q.fits")
        ufiles = glob(f"{dap.delens_lib.lib_dir}/*U.fits")
        m_p_idx = [int(qfile.split('_')[-2]) for qfile in qfiles]
        m_p = [i for i in range(dap.n_sims) if i not in m_p_idx]
        m_t_idx = [int(ufile.split('_')[-2]) for ufile in ufiles]
        m_t = [i for i in range(dap.n_sims) if i not in m_t_idx]
        print("Q missing idx:", m_p, f"Length:{len(m_p)}")
        print("U missing idx:", m_t, f"Length:{len(m_t)}")
            
    if args.delens:
        if args.missing:
            jobs = np.array(m_t)
            assert len(m_p) <= mpi.size
            
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Delensing map-{i} in Processor-{mpi.rank}")
            QU = dap.delens_lib.get_delensed_field(i)
            del QU
            
    if args.cl:
        for i in jobs[mpi.rank::mpi.size]:
            print(f"Pure B-Mode of  map-{i} in Processor-{mpi.rank}")
            cl = dap.pseudocl_lib.get_lensed_cl(i)
            del cl
            cl = dap.pseudocl_lib.get_delensed_cl(i)
            del cl
            

