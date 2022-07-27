#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=100
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH -J ExperimentMaps1
#SBATCH -o out/maps_exp1.out
#SBATCH -e out/maps_exp1.err
#SBATCH --time=01:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/s4bird/s4bird

export ini=cmbs4_1.ini

mpirun -np $SLURM_NTASKS python map.py $ini -map_exp