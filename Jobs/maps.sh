#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=64
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/maps.out
#SBATCH -e out/maps.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird.ini

#1hr 20 mins
mpirun -np 1000 python map.py $ini -map_lensed

#mpirun -np 100 python map.py $ini -map_gs

#mpirun -np 1000 python map.py $ini -map_exp