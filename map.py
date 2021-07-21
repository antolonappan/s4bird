from simulation import GaussSim, SimExperiment, CMBLensed
import os
import toml
from plancklens.helpers import mpi
from plancklens import utils
import argparse
from noise import NoiseMap_s4_LAT

ini_dir = '/global/u2/l/lonappan/workspace/S4bird/ini'


parser = argparse.ArgumentParser(description='ini')
parser.add_argument('inifile', type=str, nargs=1)
parser.add_argument('-map_gs', dest='map_gs',action='store_true',help='Make Gaussian maps')
parser.add_argument('-map_exp', dest='map_exp',action='store_true',help='Make experiment maps')
parser.add_argument('-map_lensed', dest='map_lensed',action='store_true',help='Make Lensed CMB maps')
args = parser.parse_args()
ini = args.inifile[0]




ini_file = os.path.join(ini_dir,ini)
config = toml.load(ini_file)



map_config = config['Map']
file_config = config['File']
fid_config = config['Fiducial']


base = file_config['base_name']
workbase = file_config['base_folder']

nlev_t = map_config['nlev_t']
nlev_p = map_config['nlev_p']
nside = map_config['nside']
maskfile = map_config['mask']
beam = map_config['beam']
n_sims = map_config['nsims']    

pathbase = os.path.join(workbase,base)

if bool(map_config['do_GS']):
    raw_mappath = os.path.join(workbase,map_config['GS_folder'])
    map_path = os.path.join(pathbase,'GS','Maps')
else:
    raw_mappath = os.path.join(workbase,map_config['RS_folder'])
    map_path = os.path.join(pathbase,'RS','Maps')        

noise_config = config['Noise']
noise_folder = os.path.join(pathbase,'Noise')
noise_do_red = noise_config['do_red']

cl_folder = os.path.join(workbase,fid_config['folder'])
cl_base = fid_config['base']







if args.map_gs:
    gs_base = GaussSim(cl_folder,cl_base,raw_mappath,nside,n_sims)
    gs_base.run_job()
mpi.barrier()

if args.map_exp:
    exp_map = SimExperiment(raw_mappath,map_path,nside,maskfile,beam,nlev_t,nlev_p,n_sims,noise_folder,bool(noise_do_red))
    exp_map.run_job()
mpi.barrier()

if args.map_lensed:
    cmb_map = CMBLensed(raw_mappath,nside,cl_folder,cl_base,n_sims)
    cmb_map.run_job()
mpi.barrier()
    

