#!/bin/bash
#SBATCH --qos=regular
#SBATCH --constraint=haswell
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/fg.out
#SBATCH -e out/fg.err
#SBATCH --time=08:00:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC

cd /global/u2/l/lonappan/workspace/S4bird

export ini=litebird.ini

srun -n 1 python foregrounds.py $ini -dust
#srun -n 1 python foregrounds.py $ini -synch