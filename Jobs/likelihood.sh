#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=64
#SBATCH --ntasks=1000
#SBATCH --cpus-per-task=1
#SBATCH -J s4bird
#SBATCH -o out/delens.out
#SBATCH -e out/delens.err
#SBATCH --time=00:30:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc

conda activate PC2

cd /global/u2/l/lonappan/workspace/New_s4bird/s4bird

export ini=delensing.ini


mpirun -np 100 python libparam.py $ini -lh 