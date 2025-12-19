# JODIE: Joint mOdel for Direct and Indirect Effects

[Code](#code)

[Data preparation](#dataprep)

[JODIE](#jodie)

[Within-trio one-marker-at-a-time GWAS (fGWAS)](#fgwas)

[Simulations](#simulations)

[Inferring the genotype of a missing parent](#missparent)

JODIE is a joint Bayesian method (Gibbs sampler) that, for a single outcome, is able to (i) estimate the unique contribution of different genetic components (here: direct, indirect parental and parent-of-origin genetic effects) to phenotypic variation; (ii) determine the covariances between the different effects; and (iii) find shared and distinct associations between the different genetic components, while allowing for sparsity and correlations within the genomic data. Additionally, code to perform a within-trio one-marker-at-a-time genome-wide association study (family GWAS, fGWAS) to jointly estimate the regression coefficients for the allelic variation in the child, mother, father, and of parent-of-origin assignment is provided.

<a name="code"/>

### Code
The code is written in python using MPI, and was tested with python/3.12.8 and openmpi/4.1.8 on a high performance computing cluster using slurm. 

### 1. Set up python environment

```
module load python/3.12.8
module load openmpi/4.1.8
python -m venv *nameofyourenv*
source *nameofyourenv*/bin/activate
pip install -U pip
pip install numpy scipy matplotlib loguru mpi4py welford zarr dask pandas tqdm scikit-allel
deactivate
```
Typcial installation times are less than a minute.
Note that the current setup needs zarr version 3. If you are using zarr version2, switch to tag zarr2.

### 2. Get code

```
git clone https://github.com/medical-genomics-group/JODIE
```

### 3. Run code

Load modules and source pyenv:
```
module load python/3.12.8
module load openmpi/4.1.8
source *nameofyourenv*/bin/activate
```

Information about which input parameters a program requires and how to run it is given in the first few lines of each program and below.

<a name="dataprep"/>

## Data preparation

The data preparation step is based on phased and imputed child-mother-father trio data in vcf format from the pipeline described in _R. Hofmeister et.al. Nature Communications 13 (1), 6668_. 

The following code needs to be run in order:

### a.) preprocessing_vcf_data.py

   Preprocessing vcf files with genotype information to have the data format required by JODIE and fGWAS. For a large amount of SNPs, this is best done separetely for each chromosome.\

   ```
   python preprocessing_vcf_data.py --index_trios Trios.ped --inputfiles indir/chr1.vcf.gz --dir outdir/  
   ```

   Needed input:
   + vcf files with genotype information
   + tab delimited files with id information for trios with child id as first column (and separately for duos if needed; missing parent will be inferred).
     
   Output:
   + genotype file in zarr format with child, mother, father, parent-of-origin information for each marker ({outdir}/genotype.zarr)
   + rsid file in csv format with CHR, POS, RSID, REF, ALT information ({outdir}/rsids.csv)
   
   Important notes:
   + Be aware that the number of genetic components, k, is hard-coded here, as this code requires a certain order.
   + File structure for output is assumed to be outdir/chrX/ where X is the chromosome number if chromosomes are processed separately.


### b.) order_phenotype.py
   Ordering the phenotypes of children according to the index file, so that they are in the same order as genotype files. This step needs to be run for every phenotype.\
   
   ```  
   python order_phenotype.py --y phenotype.tsv --index_trios trios.ped --pheno pheno --outdir dir/
   ```

   Needed input:
   + tab-delimited phenotype file with ID, VAL
   + tab delimited files with id information for trios with child id as first column (and separately for duos if needed)
     
   Output:
   + ordered phenotype file ({outdir}/ordered_{pheno}.txt)
   + rmid file with list of individuals with missing phenotype according to their row number in genotype file({outdir}/rm_id_ordered_{pheno}.txt)
     

### c.) calc_xtx.py

   Calculating the standardized genotype matrix squared, separately for each chromosome. For a large amount of SNPs, this is best done separetely for each chromosome. This step needs to be rerun for different phenotypes if there are individuals with missing phenotypes that are removed.\

   ```
   python calc_xtx.py --n 10000 --p 100000 --k 4 --pheno pheno --xfile genotype.zarr/ --dir outdir/ --rmid rm_id_ordered_{pheno}.txt
   ```
   
   Needed input:
   + genotype file created in step a
   + rmid file created in step b (optional, only needed if individuals are removed due to missing phenotypes)
     
   Output:
   + standardized and squared matrix calculated for the different component of each marker in zarr format ({outdir}/XtX_{pheno}.zarr)
   + rmrsid file in txt format with column number of markers that need to be removed due to zero variation ({outdir}/rmrsid_{pheno}.txt)

   The following command was used to produce the XtX_pheno.zarr file in MC/input/ using the provided MC/input/genotype.zarr
   ```
   python calc_xtx.py --n 10000 --p 20000 --k 4 --pheno pheno --xfile MC/input/genotype.zarr/ --dir MC/input/
   ```

### d.) get_mean_std_rmrsids.py

   Calculating means and standard deviations for each column, looping over all chromosome, and merging rmrsid files. This step needs to be rerun for different phenotypes if there are individuals with missing phenotypes that are removed.\

   ```
   python get_mean_std_rmrsids.py --indir indir/ --outdir outdir/  --pheno pheno --nchr 22
   ```

   Needed input:
   + genotype files created in step a - provide only the path without genotype.zarr
   + rmrsid_pheno.txt file created in step c, which is assumed to be in the same directory as genotpye files
     
   Output:
   + zarr file with means of columns ({outdir}/mean.zarr)
   + zarr file with standard devations of columns ({outdir}/std.zarr)
   + merged list of rsids to remove ({outdir}/rmrsid_{pheno}.txt)
     
   The following command was used to produce the mean.zarr, std.zarr and rmrsid_pheno.txt files in MC/input/.
   ```
   python get_mean_std_rmrsids.py --indir MC/input --outdir MC/input/  --pheno pheno --nchr 1
   ```


<a name="jodie"/>

## JODIE

JODIE estimates direct, indirect maternal and paternal and parent-of-origin effects and the population variance matrix due to this effects. 
An example data set simulated according to the section Simulations is provided in the directory MC.
All data preparation steps have been ran for the simulated data so that JODIE can be tested immediately using the following command.

   ```
   python jodie.py --n 10000 --p 20000 --k 4 --g 20000 --iters 500 --burnin 100 --x MC/input/genotype.zarr/ --y MC/true/V5/phenotype.txt --resultdir MC/results/ --xtx MC/input/XtX_pheno.zarr/ --sfile MC/input/std.zarr --mfile MC/input/mean.zarr --rmrsid MC/input/rmrsid.txt
   ```
   
   Needed input is required to be able to fit into RAM:
   + genotype file created in step a
   + odered phenotype file created in step b
   + XtX file created in step c
   + mean file created in step d
   + std file created in step d
   + rmid file created in step b (optional, only needed if individuals are removed due to missing phenotypes)
   + rmrsid file created in step d (optional, only needed if markers are removed due to no variation)
     
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

   During the burnin, the current estimate of every 500 iterations is saved to be able to restart the Gibbs sampler if necessary. This includes:
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

### Important notes on JODIE:

   + JODIE is not limited to estimate the four genetic components (direct, maternal, paternal, parent-of-origin). The number of genetic components has to reflect what is provided in the input genotype file, but can be varied with the input parameter --k. The genotype file is required to be of the dimensions (n x kp), where n is the number of individuals, p the number of markers, and k the number of genetic components. The order of the columns is genetic component 1 for marker 1, genetic component 2 for marker 1, ..., genetic component k for marker 1, genetic component 1 for marker 2, ..., genetic component k for marker p.
   + JODIE is set up that all input data needs to fit into RAM. 
   + The time complexity of JODIE is of order O(npk), where n is the number of individuals, p the number of markers, and k the number of genetic components.
   + Restarting JODIE requires the needed files to be in {resultdir} and the names be the ones for iteration 1000. It is suggested to create a new {resultdir} and copy the files need for restarting (V_1000.txt, sigma2_1000.txt, beta_1000.csv.zip, L_1000.txt, Z_1000.txt)there. These will be overwritten.


### Association studies with JODIE
It is possible to perform association studies with the framework. However, special care needs to be taken if the associations are tested for each genetic component individually. The model is set up so that markers are either included in the model for all genetic components or not included at all. Therefore, if a marker is included with a high posterior inclusion probability, one needs to check for each of the genetic components if the effect size +/- standard deviation includes 0. If 0 is covered by effect size +/- standard deviation, there is no association.


<a name="fgwas"/>

## Within-trio one-marker-at-a-time GWAS (fGWAS)

fGWAS estimates regression coefficients for one marker at a time using multiple regression.\
The code fGWAS.py is independent of jodie.py, but uses the same inputs and can be run after data preparation step c. fGWAS can be tested immediately using the provided example data and the following command:

```
python fGWAS.py --indir MC/input/ --yfile MC/true/V5/phenotype.txt --outdir MC/results/
```

Needed input:
   + genotype files created in step a - provide only the path without genotype.zarr
   + rsid file in csv format with CHR, POS, RSID, REF, ALT information. Note: rsid file needs to be in the same directory as genotype zarr!
   + ordered phenotype file created in step b

Output:
   + rsid file with regression coefficients and standard error for each marker and each component including intercept which corresponds to the columns coef1 and stde1 ({outdir}+'/fGWAS_{pstart}-{pend}.csv.zip, where {pstart} is 0 and {pend} is the total number of markers if these are not provided as input arguments)


<a name="simulations"/>

## Simulations
Two types of simulations can be generated:

a.) Including a simulated genotype\
b.) Using real genotype data (not inlcuding NaN values)

#### a.) With simulated genotype
The genotype matrix is simulated in the needed file format using **preprocessing_vcf_MC.py**. The corresponding phenotype is generated with **genY.py**.

The following commands will generate genotypes and phenotypes with 10,000 individuals and 20,000 markers of which 1000 are causal, using the realistic variance scenario. This setup is identical to what was used to generate the example data in the directory MC.

```
python preprocessing_vcf_MC.py --n 10000 --p 20000 --ntrios 10000 --na 0 --outdir
python genY.py --n 10000 --p 20000 --pc 1000 --dir outdir/ --x indir/genotype.zarr/  --scen 4
```

#### b.) With real genotype data
The phenotype and effects are generated using **genY.py**, assuming that there are no NaN values within the genotype data.

```
python genY.py --n 10000 --p 20000 --pc 1000 --dir outdir/ --x indir/genotype.zarr/  --scen 4
```

### Timing for simulated example
Simulating a genotype with 20,000 markers for 10,000 indiviudals takes about 1 min; generating a phenotype for the genotype less than 10s; calculating XtX about 2 min; and preparing mean and std files about less than 30s. Running JODIE without MPI for this example takes about 15s per iteration, while fGWAs takes less than 4 min. This was tested on an Apple MacBookPro.


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
