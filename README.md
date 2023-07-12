# Parental analysis
The code for jointly analysing direct, indirect and parent-of-origin effects is written in python using MPI, and was tested with python 3.11.1 and openmpi 4.1.4.

### 1. Set up python environment

```
module load python/3.11.1
module load openmpi/4.1.4
python -m venv *nameofyourenv*
source *nameofyourenv*/bin/activate
pip install -U pip
pip install numpy scipy matplotlib loguru mpi4py welford zarr dask pandas tqdm scikit-allel
deactivate
```
### 2. Get code

```
git clone https://github.com/medical-genomics-group/parental
```

### 3. Run code
Load modules and source pyenv:
```
module load python/3.11.1
module load openmpi/4.1.4
source *nameofyourenv*/bin/activate
```
The commands for how to run a certain program are given at the beginning of each program alongside with the explanation of the input parameters.

The sequence of the programs is:
1. Preprocessing vcf files with genotype information to have the data format required by the Gibbs sampler:
   *preprocessing_data_v1.py*
   Needed input: vcf files, id file
3. Processing phenotype to have the same order as the genotype matrix: *order_phenotype.py*
4.) Gibbs:
parental_gibbs_stdX_MPI.py
1. plotting
2. predicting
