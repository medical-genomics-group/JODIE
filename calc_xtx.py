# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy zarr dask
```
python calc_xtx.py --n 10000 --p 100000 --k 4 --pheno pheno --xfile genotype.zarr/ --dir outputdir/ --rmid list_missingpheno.txt
```
--n number of individuals (required)
--p number of markers (required)
--k number of genetic components (k=2,3,4); default=4
--x genotype file created in preprocessing_vcf_data.py (required)
--dir path to output directory (required)
--rmid list in txt format with line number of individual with missing phenotype (according to line in genotype file)
--pheno name of phenotype for output filename
"""
import sys
import argparse
import numpy as np
import scipy.stats as stats
from loguru import logger
import zarr
import dask.array as da


def main(n, p, k, xfile, dir, rmid, pheno):

    # open zarr file
    z = zarr.open(xfile, mode='r')
    xdata = da.from_zarr(z)
    X = xdata.compute()
    X = X.astype('float')
    ## delete rows where phenotype is na
    if rmid is not None:
        lines = list(np.loadtxt(rmid, delimiter=",").astype('int'))
        lines = [l for l in lines if l < n]
        X = np.delete(X, lines, axis=0)
        n = len(X)
        logger.info(f"After removing individuals without phenotype: {n=}")
    # replace 9 with nans to compute mean and std
    X = np.where(np.equal(X,9), np.nan, X)
    ## check for columns with no variance due to removed lines
    #logger.info(f"{np.unique(np.nanstd(X, axis=0))=}")
    #logger.info(f"{np.unique(np.nanmean(X, axis=0))=}")
    sd = np.nanstd(X, ddof=1, axis=0)
    did= []
    for i in range(0,k):
        did = np.append(did, np.array(np.where(sd[i::k]==0)).reshape(-1))
    did = np.unique(did)
    ## if there are columns with no variance, remove makers (i.e k columns per marker)
    lid = []
    if len(did) > 0:
        logger.info(f"Before removing columns: {X.shape=}")
        for i in range(len(did)):
            lid = np.append(lid, np.arange(did[i]*k, k*(did[i]+1)))
        lid = lid.astype(int)
        #logger.info(f"{lid=}")
        X = np.delete(X, lid, axis=1)
        logger.info(f"After removing columns: {X.shape=}")
        p -= len(did)
    logger.info(f"{p=}")
    # replace nans with mean to calculate xtx
    mean = np.nanmean(X, axis=0)
    X = np.where(np.isnan(X), mean, X)
    XtX = np.zeros((p*k,k))
    logger.info(f"{XtX.shape=}")
    for j in range(p):
        X[:, j*k:k*(j+1)] = stats.zscore(X[:, j*k:k*(j+1)], axis=0, ddof=1)
        XtX[j*k:k*(j+1),:] = np.matmul(X[:,j*k:k*(j+1)].T, X[:,j*k:k*(j+1)])
    # logger.info(f"CHECK IF NANS ARE IN XTX: {np.unique(XtX)=}")
    # save xtx
    if zarr.__version__.startswith('3'):
        logger.info("zarr 3")
        zxtx = zarr.create_array(store=f'{dir}/XtX_'+pheno+'.zarr', shape=XtX.shape, chunks=(10000,k), dtype='float')
        zxtx[:] = XtX
    elif zarr.__version__.startswith('2'):
        logger.info("zarr 2")
        zxtx = zarr.array(XtX, chunks=(1000,None)) ## zarr2
        zarr.save(dir+'/XtX_'+pheno+'.zarr', zxtx) ## zarr2
    logger.info(f"{zxtx.info=}")
    
    # save index of removed columns
    if len(did) > 0:
        logger.info(f"{lid=}")
        np.savetxt(f"{dir}/rmrsid_{pheno}.txt", lid, delimiter=",", fmt="%i")
    

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating XtX.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--k', type=int, default=4, help='number of family member incl. POO (2,3 or 4; default=4)')
    parser.add_argument('--x', type=str, help='genotype matrix filename (zarr files)', required = True)
    parser.add_argument('--dir', type=str, help='path to storage directory', required = True)
    parser.add_argument('--rmid', type=str, help='list of ids to delete (default is None)')
    parser.add_argument('--pheno', type=str, help='name of phenotype', required=True)
    args = parser.parse_args()
    logger.info(args)

    logger.remove()
    logger.add(
        sys.stderr,
        backtrace=True,
        diagnose=True,
        colorize=True,
        level=str("debug").upper(),
    )
    np.set_printoptions(precision=6, suppress=True)
    main(n = args.n, # number of individuals
        p = args.p,  # number of markers
        k = args.k, # number of traits
        xfile = args.x, # genotype file
        dir = args.dir, # path to results directory
        rmid = args.rmid,
        pheno = args.pheno,
        ) 
    logger.info("Done.")
