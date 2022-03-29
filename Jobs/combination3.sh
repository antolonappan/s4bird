#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=64
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH -J combine3
#SBATCH -o out/combination3.out
#SBATCH -e out/combination3.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate PC2
cd /global/u2/l/lonappan/workspace/s4bird/s4bird

export ini=combination3.ini

#mpirun -np $SLURM_NTASKS python combination.py $ini -job -red
#mpirun -np $SLURM_NTASKS python combination.py $ini -job
mpirun -np $SLURM_NTASKS python combination.py $ini -comb