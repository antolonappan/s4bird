#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=64
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH -J Pseudo2
#SBATCH -o out/Cl2.out
#SBATCH -e out/Cl2.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/s4bird/s4bird

export ini=delensing2.ini

#10 min
mpirun -np $SLURM_NTASKS python delens.py $ini -cl 
