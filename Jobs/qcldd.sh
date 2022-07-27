#!/bin/bash
# SBATCH --qos=debug
# SBATCH --constraint=haswell
# SBATCH --ntasks=1000
# SBATCH --cpus-per-task=1
# SBATCH -J qclss
# SBATCH -o out/qclss.out
# SBATCH -e out/qclss.err
# SBATCH --time=00:05:00
# SBATCH --mail-type=begin,end,fail
# SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate PC2
cd /global/u2/l/lonappan/workspace/s4bird/s4bird

export ini=litebird1.ini


mpirun -np $SLURM_NTASKS python quest.py $ini -qcldd
