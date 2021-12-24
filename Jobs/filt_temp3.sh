#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/filt_temp3.out
#SBATCH -e out/filt_temp3.err
#SBATCH --time=01:20:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate PC2
cd /global/u2/l/lonappan/workspace/New_s4bird/s4bird

export ini=litebird3.ini


mpirun -np $SLURM_NTASKS python libparam.py $ini -ivt -missing

