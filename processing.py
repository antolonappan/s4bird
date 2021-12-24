import os
import numpy as np
from plancklens.helpers import mpi
from delens import Delensing, Pseudo_cl, Efficency
import toml
from libparam import FilteringAndQE


try:
    import argparse
    parser = argparse.ArgumentParser(description='ini')
    parser.add_argument('inifile', type=str, nargs=1)
    parser.add_argument('-delens', dest='delens', action='store_true', help='do Delensing')
    args = parser.parse_args()
    ini = args.inifile[0]
except:
    ini = toml.load('/global/u2/l/lonappan/workspace/New_s4bird/s4bird/ini_new/libparam.ini')['file']


ini_dir = '/global/u2/l/lonappan/workspace/New_s4bird/s4bird/ini_new/'
ini_file = os.path.join(ini_dir,ini)

config = toml.load(ini_file)

# GET CONFIG SECTIONS
file_config = config['File']
of_config = config['OF']
by_config = config['BY']
delens_config = config['Delens']

# FILE CONFIG
base = file_config['base_name']
workbase = file_config['base_folder']
pathbase = os.path.join(workbase,base)

of_fqe = FilteringAndQE(os.path.join(ini_dir,of_config['ini']),int(of_config['set']))
by_fqe = FilteringAndQE(os.path.join(ini_dir,by_config['ini']),int(by_config['set']))


path_final = os.path.join(pathbase,f"SIM_SET{of_fqe.sim_set}")   

############################################################


delens_path = os.path.join(path_final, delens_config['folder'])

assert of_fqe.maskpaths[0] == by_fqe.maskpaths[0]

sims = of_fqe.sims

if delens_config['template']=='of':
    ivfs_raw = of_fqe.ivfs_raw
elif delens_config['template']=='by':
    ivfs_raw = by_fqe.ivfs_raw
else:
    raise ValueError
    
qlms_dd = by_fqe.qlms_dd
qresp_dd = by_fqe.qresp_dd
nhl_dd = by_fqe.nhl_dd
n_sims = by_fqe.n_sims
lmax_qlm = by_fqe.lmax_qlm
cl_unl = by_fqe.cl_unl
nside = by_fqe.nside
maskpaths = by_fqe.maskpaths
qe_key = by_fqe.qe_key

delens_lib = Delensing(delens_path,sims,ivfs_raw,qlms_dd,qresp_dd,nhl_dd,n_sims,lmax_qlm,cl_unl['pp'],nside,maskpaths[0],qe_key,transf=True,save_template=True,verbose=False)

#pseudocl_lib = Pseudo_cl(pseudocl_path,delens_lib,pseudo_cl_config['mask'],beam=)

if __name__ == "__main__":
    jobs = np.arange(n_sims)
            
    if args.delens:
         for i in jobs[mpi.rank::mpi.size]:
            print(f"Delensing map-{i} in Processor-{mpi.rank}")
            QU = delens_lib.get_delensed_field(i)
            del QU

            

