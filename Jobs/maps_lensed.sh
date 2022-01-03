#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=64
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH -J Lensed Maps
#SBATCH -o out/maps_lensed.out
#SBATCH -e out/maps_lensed.err
#SBATCH --time=00:45:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird.ini

mpirun -np $SLURM_NTASKS python map.py $ini -map_lensed
