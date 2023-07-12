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

The sequence of the programs is:
1. Preprocessing vcf files with genotype information to have the data format required by the Gibbs sampler:
   *preprocessing_data_v1.py*\\
   Needed input: vcf files with genotype information, tab delimited file with id information for trios and duos
2. Calculating X.T@X: *calc_xtx.py*
3. Processing phenotype to have the same order as the genotype matrix: *order_phenotype.py*
4. Running Gibbs sampler: parental_gibbs_stdX_MPI.py
5. Plotting
6. Predicting

Common input parameters are:
```
--n number of individuals
--p number of markers
--k number of genetic components
--y path to phenotype file in txt format
--xfiles path to genotype files in zarr format; multiple files are separated by space
--dir path to output directory
--g number of markers in each group if grouping is desired; otherwise g=p (either g or gindex is needed as input parameter, not both)
--gindex txt file with information about which group each marker belongs to in the same order as the markers in the genotype matrix (either g or gindex is needed as input parameter, not both)
```
For information about which input parameters a program requires and how to run it, have a look at the first few lines of each program.
Be aware that the number of genetic components is often hard-coded in the preprocessing and plotting steps, as these programs require a certain order.

### 4. Output

