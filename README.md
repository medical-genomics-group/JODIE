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
1. **preprocessing_vcf_data.py**\
   Preprocessing vcf files with genotype information to have the data format required by the Gibbs sampler\
   Needed input:
   + vcf files with genotype information
   + tab delimited files with id information for trios (and separately for duos if needed; missing parent will be inferred)
2. **calc_xtx.py**\
   Calculating the standardized genotype matrix squared\
   Needed input:
   + genotype file created in step 1
   + list in txt format with line number of individual with missing phenotype (according to line in genotype file)
   Step needs to be rerun for different phenotypes if there are individuals with missing phenotypes that are removed
3. **parental_gibbs_sampler.py**\
   Estimating parameters using a Gibbs sampler\
   Needed input:
   + genotype file created in step 1
   + XtX file created in step 2
   + phenotype file in txt format without header in the same order as the genotype matrix
   + list in txt format with line number of individual with missing phenotype (according to line in genotype file) - these individuals will be removed from the analysis
4. **plot_all_traits_1group.py**\
   Plotting variances, covariances, correlations and sigma2 for several traits (only 1 group) 
6. Predicting

Common input parameters are:
```
--n number of individuals
--p number of markers
--k number of genetic components
--y path to phenotype file in txt format
--xfiles path to genotype files in zarr format; multiple files are separated by space
--dir path to output directory
--g number of markers in each group if grouping is desired; otherwise g=p; this assumes that markers are ordered in groups in sequence (either g or gindex is needed as input parameter, not both)
--gindex txt file with information about which group each marker belongs to in the same order as the markers in the genotype matrix (either g or gindex is needed as input parameter, not both)
```
For information about which input parameters a program requires and how to run it, have a look at the first few lines of each program.
Be aware that the number of genetic components is often hard-coded in the preprocessing and plotting steps, as these programs require a certain order.

## Output of Gibbs sampler
+ mean_betasXX.csv: posterior mean of effects after XX iterations (incl. burnin)
+ mean_prob.txt: posterior exclusion probability for each marker (all genetic components are either included or excluded)
The posterior means are saved every 500 iterations (XX denotes the iteration). 

During the burnin, the current estimate of every 500 iterations is saved to be able to restart the Gibbs sampler if necesseary. This includes:
+ beta_XX.csv.zip: effects
+ epsilonXX.txt: residual errors (can be used as phenotype file when restarting)
+ L_XX.txt: part of covariance matrix V = LDL.T
+ 

## Simulations
Two types of simulations can be generated:
1. Including a simulated genotype
2. Using real genotype data

### 1. With simulated genotype
The genotype matrix is simulated in the needed file format using **preprocessing_vcf_MC.py**. The corresponding phenotype is generated with **genY.py**.

### 2. With real genotype data
The phenotype and effects are generated using **genY.py**.

Steps 2 onwards are the same as for real data.
