#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/s4bird.out
#SBATCH -e out/s4bird.err
#SBATCH --time=00:05:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird.ini

#1hr 20 mins
#mpirun -np 100 python map.py $ini -map_lensed

#mpirun -np 100 python map.py $ini -map_gs

mpirun -np 1 python map.py $ini -map_exp

#1hr 20 mins
#mpirun -np 13 python libparam.py $ini -ivt 

#1hr 15 mins
#mpirun -np 13 python libparam.py $ini -ivp

#20 min
#mpirun -np 13 python libparam.py $ini -dd

#15 min
#mpirun -np 13 python libparam.py $ini -delens

#10 min
#mpirun -np 13 python libparam.py $ini -cl 






#mpirun -np 13 python libparam_cross.py $ini  -ivt 

#mpirun -np 13 python libparam_cross.py $ini  -ivp

#mpirun -np 13 python libparam_cross.py $ini -dd

#15 min
#mpirun -np 13 python libparam_cross.py $ini -delens

#10 min
#mpirun -np 13 python libparam_cross.py $ini -cl 