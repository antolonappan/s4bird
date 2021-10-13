#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=50
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/delens.out
#SBATCH -e out/delens.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird_s4mask.ini

#15 min
#mpirun -np 100 python libparam.py $ini -delens

#10 min
#mpirun -np 100 python libparam.py $ini -cl 

mpirun -np 100 python libparam.py $ini -lh 