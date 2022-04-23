from simulation import SimExperiment, CMBLensed, SimExperimentFG
import os
import toml
import numpy as np
from plancklens.helpers import mpi
from plancklens import utils
import argparse
from noise import NoiseMap_s4_LAT

ini_dir = '/global/u2/l/lonappan/workspace/s4bird/s4bird/ini/'


parser = argparse.ArgumentParser(description='ini')
parser.add_argument('inifile', type=str, nargs=1)
parser.add_argument('-map_exp', dest='map_exp',action='store_true',help='Make experiment maps')
parser.add_argument('-map_lensed', dest='map_lensed',action='store_true',help='Make Lensed CMB maps')
args = parser.parse_args()
ini = args.inifile[0]




ini_file = os.path.join(ini_dir,ini)
config = toml.load(ini_file)



map_config = config['Map']
file_config = config['File']
fid_config = config['Fiducial']
fg_config = config['Foreground']

do_fg = bool(fg_config['do'])

base = file_config['base_name']
workbase = file_config['base_folder']

nlev_t = map_config['nlev_t']
nlev_p = map_config['nlev_p']
nside = map_config['nside']
maskfile = map_config['mask']
beam = map_config['beam']
n_sims = map_config['nsims']
sim_set = int(map_config['set'])

pathbase = os.path.join(workbase,base)


raw_mappath = os.path.join(workbase,map_config['folder'])

if do_fg:
    map_path = os.path.join(pathbase,f"SIM_SET{sim_set}_FG",'Maps')
else:
    map_path = os.path.join(pathbase,f"SIM_SET{sim_set}",'Maps')

input_mappath = os.path.join(raw_mappath,f"CMB_SET{sim_set}")

noise_config = config['Noise']
noise_folder = os.path.join(pathbase,'Noise')
noise_do_red = noise_config['do_red']

cl_folder = os.path.join(workbase,fid_config['folder'])
cl_base = fid_config['base']


fg_dir = os.path.join(workbase,fg_config['folder'])
fg_str = fg_config['model']
fg_nside = fg_config['nside']
fg_table = fg_config['table']


if args.map_exp:
    if do_fg:
        exp_map = SimExperimentFG(input_mappath,map_path,fg_nside,maskfile,beam,fg_dir,fg_str,fg_table)
    else:
        exp_map = SimExperiment(input_mappath,map_path,nside,maskfile,beam,nlev_t,nlev_p,n_sims,noise_folder,bool(noise_do_red))
    
    exp_map.run_job(n_sims)
    
mpi.barrier()

if args.map_lensed:
    cl_path = os.path.join(workbase,'CAMB')
    unlen_file =  "BBSims_scal_dls.dat"
    pot_file = "BBSims_lenspotential.dat"
    len_file = "BBSims_lensed_dls.dat"
    cmb_map = CMBLensed(raw_mappath,n_sims,cl_path,unlen_file,pot_file,len_file,sim_set)
    cmb_map.run_job()
mpi.barrier()
