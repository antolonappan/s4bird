from simulation import GaussSim, Sim_experiment
import os
import toml
from plancklens.helpers import mpi
from plancklens import utils
import argparse
from noise import NoiseMap_s4_LAT

ini_dir = '/global/u2/l/lonappan/workspace/s4bird/s4bird/validations_dir/ini'

base_map_ini = toml.load(os.path.join(ini_dir,'GS_maps.ini'))
base_map_map_config = base_map_ini['Map']
base_map_fid_config = base_map_ini['Fiducial']
cl_len_base = utils.camb_clfile(base_map_fid_config['lensed'])
base_outfolder = base_map_map_config['folder']
base_nside = base_map_map_config['nside']
base_nsims = base_map_map_config['nsims']
if base_map_map_config['seed_file'] == 'None':
    base_seeds = None
else:
    base_seeds = base_map_map_config['seed_file']


gs_base = GaussSim(cl_len_base,base_outfolder,base_nside,base_nsims,base_seeds)




parser = argparse.ArgumentParser(description='ini')
parser.add_argument('inifile', type=str, nargs=1)
parser.add_argument('-gs_map_base', dest='gs_map_base',action='store_true',help='Make Gaussian base maps')
parser.add_argument('-noisemap', dest='noisemap',action='store_true',help='Make Noise maps')
parser.add_argument('-map_exp', dest='map_exp',action='store_true',help='Make experiment maps')
args = parser.parse_args()
ini = args.inifile[0]




ini_file = os.path.join(ini_dir,ini)
config = toml.load(ini_file)


map_config = config['Map']
file_config = config['File']

nlev_t = map_config['nlev_t']
nlev_p = map_config['nlev_p']
nside = map_config['nside']
maskfile = map_config['mask']
raw_mappath = map_config['folder']
beam = map_config['beam']
n_sims = map_config['nsims']
base = file_config['base_name']
workbase = file_config['base_folder']


pathbase = os.path.join(workbase,base)
map_path = os.path.join(pathbase,'maps')

noise_config = config['Noise']
noise_folder = noise_config['folder']
noise_do_red = noise_config['do_red']
    

noise_map = NoiseMap_s4_LAT(noise_folder,nside,n_sims)
exp_map = Sim_experiment(raw_mappath,map_path,nside,maskfile,beam,nlev_t,nlev_p,n_sims,bool(noise_do_red),noise_folder)


if args.gs_map_base:
    gs_base.run_job()
mpi.barrier()

if args.map_exp:
    exp_map.run_job()
mpi.barrier()

if args.noisemap:
    noise_map.run_job()
mpi.barrier()