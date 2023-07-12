# Parental analysis
The code for jointly analysing direct, indirect and parent-of-origin effects is written in python using MPI, and was tested with python 3.11.1 and openmpi 4.1.4.

## 1. Set up python environment

> module load python/3.11.1
> module load openmpi/4.1.4
> python -m venv *nameofyourenv*
source *nameofyourenv*/bin/activate
pip install -U pip
pip install numpy loguru mpi4py

`
