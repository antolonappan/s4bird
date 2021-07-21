#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=100
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/s4bird.out
#SBATCH -e out/s4bird.err
#SBATCH --time=01:0:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird.ini

#1hr 20 mins
#mpirun -np 100 python map.py $ini -map_lensed

#mpirun -np 100 python map.py $ini -map_gs

mpirun -np 100 python map.py $ini -map_exp