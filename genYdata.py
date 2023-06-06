# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy
```
python genYdata.py --scen 0 --n 10512 --p 16574 --p0 15574 --dir MC/indepV/sim1/ --x /data/robin/EBB_server/OUTPUTS/Imputed_for_Matthew/chunks/chr22/X_trios_POOtrios/genotype.zarr/ --rmid /data/robin/EBB_server/OUTPUTS/pheno/rm_id_ordered_ht.txt 
"""

import sys
import argparse
import numpy as np
import scipy.stats as stats
from loguru import logger
import zarr
import dask.array as da


def main(n, groups, groups0, k, dir, xfiles, scen, rmid):

    G = len(groups)
    # random generator
    rng = np.random.default_rng()
    # read in genotype matrix
    for i in range(len(xfiles)):
        z = zarr.open(xfiles[i], mode='r')
        #logger.info(f"{i=}, {z=}")
        if i == 0:
            xdata = da.from_zarr(z)
            #logger.info(f"{i=}, {xdata=}")
        else:
            xdata = da.append(xdata,z, axis=1)
            #logger.info(f"{i=}, {xdata=}")
    xdata = xdata.astype('int8')
    logger.info(f"{xdata=}")
    ## delete rows where phenotype is na
    if rmid is not None:
        lines = list(np.loadtxt(rmid, delimiter=",").astype('int'))
        lines = [l for l in lines if l < n]
        xdata = np.delete(xdata, lines, axis=0) 
        n -= len(lines)
        logger.info(f"new number of individuals: {n=}")
    X = xdata.compute()
    # standardize X
    logger.info(f"standardize X")
    Xnorm = stats.zscore(X, axis=0, ddof=1)
    p = np.sum(groups)
    logger.info(f"Problem has dimensions {n=}, {p=}, {G=} with {groups0=} effects set to 0.")

    # var = np.array([[[0.125, 0, 0, 0],
    #                 [0, 0.125, 0, 0],
    #                 [0, 0, 0.125, 0],
    #                 [0, 0, 0, 0.125]
    #                 ],
    #                 [[0.125, -0.5*np.sqrt(0.125)*np.sqrt(0.125), -0.5*np.sqrt(0.125)*np.sqrt(0.125), -0.5*np.sqrt(0.125)*np.sqrt(0.125)],
    #                 [-0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.125, -0.5*np.sqrt(0.125)*np.sqrt(0.125), -0.5*np.sqrt(0.125)*np.sqrt(0.125)],
    #                 [-0.5*np.sqrt(0.125)*np.sqrt(0.125), -0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.125, -0.5*np.sqrt(0.125)*np.sqrt(0.125)],
    #                 [-0.5*np.sqrt(0.125)*np.sqrt(0.125), -0.5*np.sqrt(0.125)*np.sqrt(0.125), -0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.125]
    #                 ],
    #                 [[0.125, 0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.5*np.sqrt(0.125)*np.sqrt(0.125)],
    #                 [0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.125, 0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.5*np.sqrt(0.125)*np.sqrt(0.125)],
    #                 [0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.125, 0.5*np.sqrt(0.125)*np.sqrt(0.125)],
    #                 [0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.5*np.sqrt(0.125)*np.sqrt(0.125), 0.125]
    #                 ],## change data-like scenario
    #                 [[0.125, 0, 0, 0],
    #                 [0, 0.125, 0, 0],
    #                 [0, 0, 0.125, 0],
    #                 [0, 0, 0, 0.125]
    #                 ],
    #                 ])
    var = np.array([[[0.3, 0, 0, 0],
                    [0, 0.1, 0, 0],
                    [0, 0, 0.1, 0],
                    [0, 0, 0, 0.]
                    ],
                    [[0.3, 0, 0, 0],
                    [0, 0., 0, 0],
                    [0, 0, 0.1, 0],
                    [0, 0, 0, 0.]
                    ],
                    ])


    for g in range(G):
        logger.info(f"{g=}, {scen=}, {var[scen]=}")
        # calculate number of non-zero effects
        p1 = (groups[g]-groups0[g])
        # generate beta
        if p1 == 0:
            beta = np.zeros((groups0[g], k))
        else:
            beta = np.concatenate([rng.multivariate_normal(np.zeros(k), var[scen]/p1, p1), np.zeros((groups0[g], k))])
        rng.shuffle(beta)
        V = np.matmul(beta.T, beta)
        if g == 0:
            true_beta = beta.copy()
            true_V = V.copy()
        else:
            true_beta = np.concatenate([true_beta, beta])
            true_V = np.concatenate([true_V, V])

    #logger.info(f"{true_beta.shape=}")        
    logger.info(f"{true_V=}")
    logger.info(f"{np.cov(beta, rowvar=False)=}")
    logger.info(f"{np.matmul(beta.flatten().T, beta.flatten())=}")
    logger.info(f"{np.var(Xnorm@true_beta.flatten())=}")
    g = Xnorm@true_beta.flatten()

    # generate epsilon
    std = np.sqrt(1-np.var(g))
    logger.info(f"{std=}")

    epsilon = np.random.normal(0, std, n)
    true_sigma2 = np.var(epsilon)
    logger.info(f"{true_sigma2=}")
    logger.info(f"{np.std(epsilon)=}")
    # generate Y
    Y = g + epsilon
    Ynorm = stats.zscore(Y, axis=0, ddof=1)
 
    # save true values
    np.savetxt(dir+'/true_V.txt', true_V)
    np.savetxt(dir+'/true_sigma2.txt', true_sigma2.reshape(1,1))
    np.savetxt(dir+'/true_betas.txt', beta)
    np.savetxt(dir+'/true_epsilon.txt', epsilon)
    np.savetxt(dir+'/phenotype.txt', Ynorm)


#########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Data simulation.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--k', type=int, default=4, help='number of family member incl. POO (2,3 or 4; default=4)')
    parser.add_argument('--p', nargs='+', type=int, help='number of markers in each group, sums up to total number of markers', required = True)
    parser.add_argument('--p0', nargs='+', type=int, help='number of markers set to 0 in each group', required = True)
    parser.add_argument('--dir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--x', type=str, nargs='+', help='list of genotype matrix filenames (zarr files)', required = True)
    parser.add_argument('--rmid', type=str, help='list of ids to delete (default is None)')
    parser.add_argument('--scen', type=int, help="scenario for beta variance (0=independent, 1=+0.5correlation, 2=-0.5correlation, 3=data-like correlation)", required=True)
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
    np.set_printoptions(precision=9, suppress=True)
    main(n = args.n, # number of individuals
        k = args.k, # number of genetic components
        groups = np.array(args.p),  # number of markers
        groups0 = np.array(args.p0),  # number of zero markers
        dir = args.dir, # path to results directory
        xfiles = args.x, # genotype matrix file
        rmid = args.rmid,
        scen = args.scen, # scenario for V
    ) 
    logger.info("Done.")