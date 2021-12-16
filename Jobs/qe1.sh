#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --nodes=64
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/qe1.out
#SBATCH -e out/qe1.err
#SBATCH --time=01:45:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird.ini


#1hr 20 mins
#mpirun -np 1000 python libparam.py $ini -ivt 

#1hr 15 mins
mpirun -np 1000 python libparam.py $ini -ivp

#20 min
#mpirun -np 100 python libparam.py $ini -dd