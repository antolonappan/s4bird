#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -J Delensing3
#SBATCH -o out/delens3.out
#SBATCH -e out/delens3.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/s4bird/s4bird

export ini=delensing3.ini

#15 min
mpirun -np $SLURM_NTASKS  python delens.py $ini -delens -missing