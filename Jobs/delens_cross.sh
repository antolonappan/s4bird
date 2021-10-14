#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=50
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/delens_cross.out
#SBATCH -e out/delens_cross.err
#SBATCH --time=00:15:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird_w_cmbs4.ini

#15 min
#mpirun -np 100 python libparam_cross.py $ini -delens

#10 min
#mpirun -np 100 python libparam_cross.py $ini -cl 

mpirun -np 100 python libparam_cross.py $ini -lh 