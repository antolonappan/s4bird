#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=10
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J weight
#SBATCH -o out/weights.out
#SBATCH -e out/weights.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/s4bird/s4bird

export ini=litebird1.ini

mpirun -np $SLURM_NTASKS python map.py $ini -fg_weight