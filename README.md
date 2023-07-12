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
+ mean_betas.csv.zip: posterior mean of effects where columns correspond to the genetic components and rows to the markers
+ var_betas.csv.zip: variance of effects
+ mean_prob.txt: posterior inclusion probability for each marker (how often has the marker been included in the model); the marker is either included for all genetic components or not, thus only one column
+ var_prob.txt: variance of posterior inclusion probability
+ mean_sigma2.txt: posterior mean of residual variance
+ var_sigma2.txt: variance of residual variance
+ mean_V.txt: posterior mean of variance of effects (if the variance is estimated for different groups, the variances of each group are given one after the other)
+ var_V.txt: variance of variance of effects
+ trace_sigma2.txt: residual variance for each iteration (rows = iterations)
+ trace_V.txt: variance of effects for each iteration (rows = iterations), only variances are saved
+ trace_Z.txt: number of included markers for each iteration (rows = iterations)
The posterior means are saved every 500 iterations (XX denotes the iteration). 

During the burnin, the current estimate of every 500 iterations is saved to be able to restart the Gibbs sampler if necesseary. This includes:
+ beta_XX.csv.zip: effects
+ epsilonXX.txt: residual errors (can be used as phenotype file when restarting)
+ L_XX.txt: part of covariance matrix V = LDL.T
+ prob_XX.txt: tracker if marker is included in model (1) or not (0) for iteration XX
+ sigma2_XX.txt: residual variance of current iteration
+ V_XX.txt: variance of effects of current iteration
+ Z_XX.txt: number of markers included in the model at iteration XX
+ trace_sigma2XX.txt: residual variance for each iteration until iteration XX (rows = iterations)
+ trace_Vg_XX.txt: variance of effects for each iteration until iteration XX for group g (rows = iterations), only variances are saved
+ trace_ZXX.txt: number of included markers for each iteration until iteration XX (rows = iterations)

Trace plots:
+ trace_sigma2.png: residual variance as function of iteration
+ trace_V.png: (co)variances as function of iterations
+ trace_Z.png: number of included markers as function of iterations

## Simulations
Two types of simulations can be generated:
1. Including a simulated genotype
2. Using real genotype data

### 1. With simulated genotype
The genotype matrix is simulated in the needed file format using **preprocessing_vcf_MC.py**. The corresponding phenotype is generated with **genY.py**.

### 2. With real genotype data
The phenotype and effects are generated using **genY.py**.

Steps 2 onwards are the same as for real data.
