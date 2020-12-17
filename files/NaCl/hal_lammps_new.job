#!/bin/bash


#SBATCH -n 28                         # Request X X cores
#SBATCH -p parallel-28
#SBATCH -t 168:00:00                   # The job can take at most X wall-clock hours.
#SBATCH -J NaCl_2                         # Jobname

##SBATCH --constraint="XeonE51650v2"


lmp=/home/noura/LAMMPS/tests/src_v05

mpi=/usr/local/openmpi-1.8.4-ifort/bin

$mpi/mpirun -np 28 $lmp/lmp_mpi < Simulation_new.in



sleep 2


exit 0

