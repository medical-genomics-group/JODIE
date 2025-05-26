# JODIE: Joint mOdel for Direct and Indirect Effects

[Code](#code)

[Simulations](#simulations)

[Association studies](#association)

[Inferring the genotype of a missing parent](#missparent)

JODIE is a joint Bayesian method (Gibbs sampler) that, for a single outcome, is able to (i) estimate the unique contribution of different genetic components (here: direct, indirect parental and parent-of-origin genetic effects) to phenotypic variation; (ii) determine the covariances between the different effects; and (iii) find shared and distinct associations between the different genetic components, while allowing for sparsity and correlations within the genomic data. Additionally, code to perform a multiple one-marker-at-a-time genome-wide association study (family GWAS, fGWAS) to jointly estimate the regression coefficients for the allelic variation in the child, mother, father, and of parent-of-origin assignment is provided.

<a name="code"/>

### Code
The code is written in python using MPI, and was tested with python/3.11.1 with openmpi/4.1.4 and python/3.12 with openmpi/4.1.6, run on a high performance computing cluster using slurm. Information about which input parameters a program requires and how to run it is also given in the first few lines of each program.
The data preparation step of the code is based on phased and imputed child-mother-father trio data in vcf format from the pipeline described in _R. Hofmeister et.al. Nature Communications 13 (1), 6668_. 


### 1. Set up python environment

```
module load python/3.12.4
module load openmpi/4.1.6
python -m venv *nameofyourenv*
source *nameofyourenv*/bin/activate
pip install -U pip
pip install numpy scipy matplotlib loguru mpi4py welford zarr==2.17.2 dask pandas tqdm scikit-allel
deactivate
```
Typcial installation times are less than a minute.
Note that the current setup needs zarr version 2. There are some incompatibilies between zarr version 2 and 3.

### 2. Get code

```
git clone https://github.com/medical-genomics-group/JODIE
```

### 3. Run code
Load modules and source pyenv:
```
module load python/3.12.4
module load openmpi/4.1.6
source *nameofyourenv*/bin/activate
```

The sequence of the programs is:

#### a.) preprocessing_vcf_data.py
   Preprocessing vcf files with genotype information to have the data format required by the Gibbs sampler, separetely for each chromosome\
   Needed input:
   + vcf files with genotype information
   + tab delimited files with id information for trios with child id as first column (and separately for duos if needed; missing parent will be inferred)
     
   Output:
   + genotype file in zarr format with child, mother, father, parent-of-origin information for each marker
   + rsid file in csv format with CHR, POS, RSID, REF, ALT information
     
   File structure for output is assumed to be outdir/chrX/ where X is the chromosome number.

#### b.) order_phenotype.py
   Ordering the phenotypes of children accroding to index file (same order as genotype file)\
   Needed input:
   + tab-delimited phenotype file with ID, VAL
   + tab delimited files with id information for trios with child id as first column (and separately for duos if needed)
     
   Output:
   + ordered phenotype file
   + rmid file with list of individuals with missing phenotype (according to line in genotype file)
     
   This step needs to be run for every phenotype.
   
#### c.) calc_xtx.py
   Calculating the standardized genotype matrix squared, separately for each chromosome\
   Needed input:
   + genotype file created in step a
   + rmid file created in step b
     
   Output:
   + standardized and squared matrix calculated for the different component of each marker in zarr format (XtX matrix)
   + rmrsid file in txt format with line number of markers that need to be removed due to zero variation
     
   This step needs to be rerun for different phenotypes if there are individuals with missing phenotypes that are removed.

#### d.) jodie.py
   Estimating parameters using a Gibbs sampler\
   Needed input is required to be able to fit into RAM:
   + genotype file created in step a
   + XtX file created in step c
   + odered phenotype file created in step b
   + rmid file created in step b - these individuals will be removed from the analysis
   + rmrsid file created in step c - these markers will be removed from the analysis
     
   Output:
   + mean_beta.csv.zip: posterior mean of effects where columns correspond to the genetic components and rows to the markers
   + var_beta.csv.zip: variance of effects
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

#### e.) fGWAS.py
Estimating regression coefficients for one marker at a time using multiple regression.\
This step is independent of jodie.py and can be run after step c.\
Needed input:
   + genotype file in zarr format
   + rsid file in csv format with CHR, POS, RSID, REF, ALT information. Note: rsid file needs to be in the same directory as genotype zarr!
   + ordered phenotype file

Output:
   + rsid file with regression coefficients and standard error for each marker and each component including intercept (coef1, stde1)

#### Common input parameters are:
```
--n number of individuals
--p number of markers
--k number of genetic components
--y path to phenotype file in txt format
--x path to genotype files in zarr format; multiple files are separated by space
--dir path to output directory
--g number of markers in each group if grouping is desired; otherwise g=p; this assumes that markers are ordered in groups in sequence (either g or gindex is needed as input parameter, not both)
--gindex txt file with information about which group each marker belongs to in the same order as the markers in the genotype matrix (either g or gindex is needed as input parameter, not both)
```
For information about which input parameters a program requires and how to run it, have a look at the first few lines of each program.
Be aware that the number of genetic components is often hard-coded in the preprocessing steps, as these programs require a certain order.

#### Limitations for JODIE:
+ JODIE is set up that all input data needs to fit into RAM. 
+ NaN values within the genotype type can be taken into account with a slightly different setup. Code for nan value is denoted with *_nan*. There is an additional step to be run after step b:\
  **get_mean_std_rmrsids_nan.py**\
   Calculating mean and standard deviations for each column, looping over all chromosome, and merging rmrsid files\
   Needed input:
   + genotype files created in step a
   + rmrsid_pheno.txt file created in step b needs to be in the same directory as genotpye files
     
   Output:
   + zarr file with means of columns
   + zarr file with standard devations of columns
   + merged list of rsids to remove
     
   This step needs to be rerun for different phenotypes if there are individuals with missing phenotypes that are removed.  

<a name="simulations"/>

## Simulations
Two types of simulations can be generated:

a.) Including a simulated genotype\
b.) Using real genotype data (not inlcuding NaN values)

#### a.) With simulated genotype
The genotype matrix is simulated in the needed file format using **preprocessing_vcf_MC.py**. The corresponding phenotype is generated with **genY.py**.


#### b.) With real genotype data
The phenotype and effects are generated using **genY.py**, assuming that there are no NaN values within the genotype data.

### Timing for simulated example
Simulating a genotype with 20,000 markers for 5,000 indiviudals takes less than 20s; generating a phenotype for the genotype less than 10s; and calculating XtX less than 30s. Running JODIE without MPI for this example takes about 11s per iteration, while fGWAs takes less than 2 min. This was tested on an Apple MacBookPro.

<a name="association"/>

## Association studies
It is possible to perform association studies with the framework. However, special care needs to be taken if the associations are tested for each genetic component individually. The model is set up so that markers are either included in the model for all genetic components or not included at all. Therefore, if a marker is included with a high posterior inclusion probability, one needs to check for each of the genetic components if the effect size +/- standard deviation includes 0. If 0 is covered by effect size +/- standard deviation, there is no association.

<a name="missparent"/>

## Inferring the genotype of a missing parent
It is possible to infer the genotype of a missing parent in **preprocessing_vcf_data.py** using the information provided by trios and Bayes theorem. To do so, a list with the id information for duos needs to be provided additionally to the one of the trios.
Be aware that the analysis for duos, even with inferring the missing parent, will be biased.

---
In case of questions or problems, please contact ilse.kraetschmer@ist.ac.at

Reference:\
Separating direct, indirect and parent-of-origin genetic effects in the human population\
Ilse Krätschmer, Laura Hegemann, Robin Hofmeister, Elizabeth C. Corfield, Mahdi Mahmoudi, Olivier Delaneau, Ole A. Andreassen, Archie Campbell, Caroline Hayward, Estonian Biobank Research Team, Riccardo E. Marioni, Eivind Ystrom, Alexandra Havdahl, Matthew R. Robinson\
doi: https://doi.org/10.1101/2025.04.28.650988 
