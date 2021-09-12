#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=50
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/qe_cross.out
#SBATCH -e out/qe_cross.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird_w_S4pLB.ini


#1hr 20 mins
#mpirun -np 100 python libparam_cross.py $ini -ivt 

#1hr 15 mins
#mpirun -np 100 python libparam_cross.py $ini -ivp

#20 min
mpirun -np 100 python libparam_cross.py $ini -dd