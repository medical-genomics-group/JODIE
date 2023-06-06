"""

Install dependencies:
```
pip install numpy loguru scipy zarr dask
```
run with n processes:
python calc_XtX.py --n 10000 --p 20000 --x MC/genotype.zarr --dir MC/ --pheno test
--rmid  
"""

import sys
import argparse
import numpy as np
import scipy.stats as stats
from loguru import logger
import zarr
import dask.array as da


def main(n, p, k, xfile, dir, rmid, pheno):

    z = zarr.open(xfile, mode='r')
    xdata = da.from_zarr(z)
    X = xdata.compute()
    X = X.astype('float')
    logger.info(f"{np.unique(np.nanstd(X, axis=0))=}")
    #logger.info(f"{np.unique(np.nanmean(X, axis=0))=}")
    ## delete rows where phenotype is na
    if rmid is not None:
        lines = list(np.loadtxt(rmid, delimiter=",").astype('int'))
        lines = [l for l in lines if l < n]
        X = np.delete(X, lines, axis=0)
        n = len(X)
    ## check for possible nan columns due to removed lines
    logger.info(f"{np.unique(np.nanstd(X, axis=0))=}")
    #logger.info(f"{np.unique(np.nanmean(X, axis=0))=}")
    sd = np.nanstd(X, ddof=1, axis=0)
    did0 = np.array(np.where(sd[0::k]==0)).reshape(-1)
    did1 = np.array(np.where(sd[1::k]==0)).reshape(-1)
    did2 = np.array(np.where(sd[2::k]==0)).reshape(-1)
    did3 = np.array(np.where(sd[3::k]==0)).reshape(-1)
    did = np.unique(np.concatenate([did0, did1, did2, did3]))
    logger.info(f"{did0=}")
    logger.info(f"{did1=}")
    logger.info(f"{did2=}")
    logger.info(f"{did3=}")
    logger.info(f"{did=}")
    lid = []
    if len(did) > 0:
        logger.info(f"{X.shape=}")
        for i in range(len(did)):
            lid = np.append(lid, np.arange(did[i]*k, k*(did[i]+1)))
        lid = lid.astype(int)
        logger.info(f"{lid=}")
        X = np.delete(X, lid, axis=1)
        logger.info(f"{X.shape=}")
        p -= len(did)
    logger.info(f"{p=}")
    XtX = np.zeros((p*k,k))
    logger.info(f"{XtX.shape=}")
    for j in range(p):
        X[:, j*k:k*(j+1)] = stats.zscore(X[:, j*k:k*(j+1)], axis=0, ddof=1)
        XtX[j*k:k*(j+1),:] = np.matmul(X[:,j*k:k*(j+1)].T, X[:,j*k:k*(j+1)])
    logger.info(f"{XtX=}")
    zxtx = zarr.array(XtX, chunks=(1000,None))
    logger.info(f"{zxtx.info=}")
    zarr.save(dir+'/XtX_'+pheno+'.zarr', zxtx)
    z = zarr.array(X, chunks=(None,1000))
    logger.info(f"{z.info=}")
    zarr.save(dir+'/std_genotype_'+pheno+'.zarr', z)



##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating XtX.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--k', type=int, default=4, help='number of family member incl. POO (2,3 or 4; default=4)')
    parser.add_argument('--x', type=str, help='genotype matrix filename (zarr files)', required = True)
    parser.add_argument('--dir', type=str, help='path to storage directory', required = True)
    parser.add_argument('--pheno', type=str, help='name of phenotype', required = True)
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
    main(n = args.n, # number of individuals
        p = args.p,  # number of markers
        k = args.k, # number of traits
        xfile = args.x, # genotype file
        dir = args.dir, # path to results directory
        rmid = args.rmid, # individuals with missing phenotypic values
        pheno = args.pheno,
        ) 
    logger.info("Done.")