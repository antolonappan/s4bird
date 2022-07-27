#!/bin/bash
#SBATCH --qos=debug
#SBATCH --constraint=haswell
#SBATCH --nodes=4
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=1
#SBATCH -J qclds
#SBATCH -o out/qclds.out
#SBATCH -e out/qclds.err
#SBATCH --time=00:05:00
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=anto.lonappan@sissa.it


source /global/homes/l/lonappan/.bashrc
conda activate PC2
cd /global/u2/l/lonappan/workspace/s4bird/s4bird

export ini=litebird1.ini


mpirun -np $SLURM_NTASKS python quest.py $ini -qclds
