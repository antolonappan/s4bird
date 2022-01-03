#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=64
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH -J Filtering Pol
#SBATCH -o out/filt_pol1.out
#SBATCH -e out/filt_pol1.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate PC2
cd /global/u2/l/lonappan/workspace/New_s4bird/s4bird

export ini=litebird1.ini


mpirun -np $SLURM_NTASKS python libparam.py $ini -ivp