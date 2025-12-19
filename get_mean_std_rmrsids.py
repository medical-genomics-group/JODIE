# -*- coding: utf-8 -*-
"""
Install dependencies:
```
pip install numpy loguru zarr
```
python get_mean_std_rmrsids_nan.py --indir indir/ --outdir outputdir/ --rmid rmid.txt --pheno pheno
```
--indir directory where genotype files are stored without subdirectories with chrX where X = 1..nchr (required)
    when nchr=1, indir does not need to have a subdirectory named chr1
--outdir output directory (required)
--nchr  total number of chromsomes (default=22)
--k number of genetic components (k=2,3,4); default=4
Needs to be run for each phenotype.
ATTENTION: rmrsid_pheno.txt needs to exist and be in the same directory as the genotype files!
"""
import os
import sys
import argparse
import numpy as np
from loguru import logger
import dask.array as da
import zarr
from pathlib import Path

def main(indir, outdir, nchr, rmid, pheno):

    ### storage
    mean = []
    std = []
    rsids = []
    p = np.zeros(nchr+1)

    ## open rmid file
    if rmid is not None:
        lines = list(np.loadtxt(rmid, delimiter=",").astype('int'))

    ## loop through chromosomes
    for i in range(1,nchr+1):
        ## open zarr files
        path=indir+"/chr"+str(i)+"/" if nchr > 1 else indir
        logger.info(f"{path=}")
        z = zarr.open(path+"/genotype.zarr/", mode='r')
        xdata = da.from_zarr(z)
        X = xdata.compute()
        X = X.astype('float')
        X = np.where(np.equal(X,9), np.nan, X)
        n, p[i] = X.shape
        logger.info(f"{X.shape=}")
    
        ## delete rows where phenotype is na
        if rmid is not None:
            lines = [l for l in lines if l < n]
            X = np.delete(X, lines, axis=0)
            n = len(X)
            logger.info(f"{n=}")
        
        # calculate mean and std and append
        s = np.nanstd(X, ddof=1, axis=0)
        std = np.append(std, s)
        m = np.nanmean(X, axis=0)
        mean = np.append(mean, m)
        logger.info(f"{len(s)=}, {len(m)=}")

        ## make sure file exists and open files
        name = path+"/rmrsid_"+pheno+".txt"
        logger.info(f"{name=}")
        if os.path.isfile(name):
            #logger.info(f"{path+"/rmrsid_"+pheno+".txt"}")
            id = np.loadtxt(name, delimiter=",")
            # shift index by the sum of markers in the chromosomes before
            ## each marker is represented by k indices due to the 4 genetic components which is already taken into account in p
            id += (np.sum(p[:i]))
            logger.info(f"Shift markers by {np.sum(p[:i])}")
            rsids = np.append(rsids, id)
    
    # save mean and std as zarr
    zarr.save(outdir+"/mean.zarr", mean)
    zarr.save(outdir+"/std.zarr", std)
    #zm = zarr.array(mean) ## zarr2
    #logger.info(f"{zm=}")
    #zarr.save(outdir+"/mean.zarr", zm) ## zarr2
    #zs = zarr.array(std) ## zarr2
    #logger.info(f"{zs=}")
    #zarr.save(outdir+"/std.zarr", zs) ## zarr2

    # save rsids
    logger.info(f"{rsids=}")
    np.savetxt(outdir+"/rmrsid.txt", rsids, delimiter=",", fmt="%i")


##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Getting means, std.dev. and rmrsids for all chromosomes.')
    parser.add_argument('--indir', type=str, help='path to input directory with zarr files (without chr)', required = True)
    parser.add_argument('--outdir', type=str, help='output directory', required = True)
    parser.add_argument('--nchr', type=int, default=22, help="number of chromosomes (default=22)")
    parser.add_argument('--rmid', type=str, help='list of ids to delete (default is None)')
    parser.add_argument('--pheno', type=str, help='name of phenotype (same as in calc_xtxt)', required=True)
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
        outdir = args.outdir,
        nchr=args.nchr,
        rmid=args.rmid,
        pheno=args.pheno,
        )
    logger.info("Done.")