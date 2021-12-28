#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=1
#SBATCH -J Filtering Temp
#SBATCH -o out/filt_temp1.out
#SBATCH -e out/filt_temp1.err
#SBATCH --time=01:20:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate PC2
cd /global/u2/l/lonappan/workspace/New_s4bird/s4bird

export ini=litebird1.ini


mpirun -np $SLURM_NTASKS python libparam.py $ini -ivt -missing

