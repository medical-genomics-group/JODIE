# -*- coding: utf-8 -*-
"""
Install dependencies:
```
pip install numpy loguru pandas dask pathlib tqdm statsmodels zarr dask
```
python linreg_1snp.py --indir --yfile --outdir --pstart --pend --k
--indir path to zarr directory without genotype.zarr
--yfile  path to standardized phenotype file
--k (default 4) k-1 is imprinting 
--outdir name should containg chr if X is split by chromosome
--pstart number of starting marker in file (if file needs to be split, default = 0) 
--pend   number of ending marker in file (if file needs to be split, if file is read in as whole, pend = total number of markers in file)

Note: rsids file needs to be in the same directory as genotype zarr
MAF is not correct for nan values!
"""
## remove multithreading
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
import numpy as np
import zarr
import pandas as pd
import dask.array as da
from loguru import logger
import pathlib
from tqdm import trange
#import statsmodels.formula.api as smf
from scipy.linalg import lstsq
from numpy.linalg import solve

## OLS using lstsq
def fit(X, y, n, k):
    # Solve using scipy.linalg.lstsq with lapack_driver="gelsy"
    params, _, rank, s = lstsq(X, y, lapack_driver="gelsy")
    residuals = y - X @ params
    rss = np.dot(residuals, residuals)
    # Estimate variance of residuals
    sigma_squared = rss / (n - k)
    # Compute (X^T X)^-1 efficiently
    xtx = X.T @ X
    xtx_inv = solve(xtx, np.eye(xtx.shape[0]))
    # Compute standard errors
    bse = np.sqrt(np.diag(sigma_squared * xtx_inv))
    #logger.info(f"{params=}, {bse=}") 
    return params, bse

def main(indir, yfile, outdir, k, pstart, pend, rmid):

    # open zarr files
    logger.info(f"{indir=}")
    dir=indir+"/genotype.zarr/"
    logger.info(f"{dir=}")
    z = zarr.open(dir, mode='r')
    xdata = da.from_zarr(z)
    ## get number of markers and individuals
    n, p = xdata.shape
    p = p//k
    ## remove individuals
    if rmid is not None:
        lines = list(np.loadtxt(rmid, delimiter=",").astype('int'))
        lines = [l for l in lines if l < n]
        xdata = np.delete(xdata, lines, axis=0)
        n, _ = xdata.shape
    ## set pend to all markers if variable is not given
    if pend is None:
        pend=p
    logger.info(f"Processing markers {pstart} to {pend-1}.")
    logger.info(f"{n=}, {p=}, {pstart=}, {pend=}, {xdata=}")
    ## open phenotype file
    y = np.loadtxt(yfile)

    ## open rsid file
    rsids = pd.read_csv(indir+"/rsids.csv", sep="\t")
    rsids = rsids[pstart:pend]
    ## add empty columns
    rsids['MAF'] = None
    mX = ['meanX'+str(i) for i in range(1,k+1)]
    sX = ['stdX'+str(i) for i in range(1,k+1)]
    rsids[mX] = None
    rsids[sX] = None
    coefs = ['coef'+str(i) for i in range(1,k+2)]
    stdes = ['stde'+str(i) for i in range(1,k+2)]
    rsids[coefs] = None
    rsids[stdes] = None

    # create Vandermonde matrix for OLS: intercept = 1
    # Add constant (intercept term) using pre-allocation for efficiency
    X = np.ones((n, k + 1))

    ## run OLS for each marker
    ## running count
    l = 0
    for j in trange(pstart, pend):

        ## replace X with actual data
        x = xdata[:,j*k:(j+1)*k].compute()
        ## replace 9 with nan
        x = np.where(np.equal(x,9), np.nan, x)
        ## calc maf
        rsids.loc[l, 'MAF'] = np.nanmean(x[:,1])/2
        # standardize X
        mean = np.nanmean(x, axis=0)
        std = np.nanstd(x, ddof=1, axis=0)
        rsids.loc[l,mX] = mean
        rsids.loc[l,sX] = std
        #logger.info(f"{mean=}, {std=}")
        x = (x - mean)/std
        ## replace nans with 0
        x = np.nan_to_num(x)
        X[:,1:k+1] = x
        ## get fit coefficients and standard errors from OLS
        rsids.iloc[l,-2*(k+1):-(k+1)], rsids.iloc[l,-(k+1):] = fit(X, y, n, k)
        #logger.info(f"{rsids=}")   
        l+=1

    # make sure output directory exists 
    pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
    ## save rsids, pos, chr in order of the markers occuring
    rsids.to_csv(outdir+'/linreg'+str(pstart)+'-'+str(pend)+'.csv.zip', compression="zip", index=False, sep="\t")

##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Run linear regression one marker at a time.')
    parser.add_argument('--indir', type=str, help='path to input directory with zarr files', required = True)
    parser.add_argument('--yfile', type=str, help='path to phenotype file', required = True)
    parser.add_argument('--outdir', type=str, help='output directory', required = True)
    parser.add_argument('--k', type=int, default=4, help="number of genetic components")
    parser.add_argument('--pstart', type=int, default=0, help="number of starting marker (default=0)")
    parser.add_argument('--pend', type=int, help="number of end marker - remember python starts at 0")
    parser.add_argument('--rmid', type=str, help='list of ids to delete (default is None)')
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
    main(
        indir = args.indir,
        yfile = args.yfile,
        outdir = args.outdir,
        k=args.k,
        pstart=args.pstart,
        pend=args.pend,
        rmid=args.rmid,
        )
    logger.info("Done.")
