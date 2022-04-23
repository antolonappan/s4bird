import mpi
from quest import FilteringAndQE
from plancklens import utils
import argparse
import healpy as hp
import numpy as np
import os

parser = argparse.ArgumentParser(description='ini')
parser.add_argument('inifile', type=str, nargs=1)
args = parser.parse_args()
ini = args.inifile[0]

ini_dir = '/global/u2/l/lonappan/workspace/s4bird/s4bird/ini'
ini_file = os.path.join(ini_dir,ini)

fqe = FilteringAndQE(ini_file)
jobs = np.arange(fqe.n_sims)

proj_dir = '/project/projectdirs/litebird/simulations/maps/lensing_project_paper'

base_dir = os.path.join(proj_dir,'LB_MASS_FG1')
mass_dir = os.path.join(base_dir,'MASS')
nn_dir = os.path.join(base_dir,'N0_and_A')

if mpi.rank == 0:
    os.makedirs(mass_dir,exist_ok=True)
    os.makedirs(nn_dir,exist_ok=True)
mpi.barrier()

key = fqe.qe_key.split('_')[-1]


for i in jobs[mpi.rank::mpi.size]:
    fname = os.path.join(mass_dir,f"phi_{key}_sims_{i:04d}.fits")
    print(f"Transfering QE-{i} in Processor-{mpi.rank}")
    qlm = fqe.qlms_dd.get_sim_qlm(fqe.qe_key,i)
    hp.write_alm(fname,qlm)

mpi.barrier()

if mpi.rank == 0:
    qresp = fqe.qresp_dd.get_response(fqe.qe_key, 'p')
    N0 = fqe.nhl_dd.get_sim_nhl(0,  fqe.qe_key,  fqe.qe_key)
    Norm = utils.cli(qresp)
    fname_n0 = os.path.join(nn_dir,'N0.fits')
    hp.write_cl(fname_n0,N0)
    fname_norm = os.path.join(nn_dir,'Norm.fits')
    hp.write_cl(fname_norm,Norm)
