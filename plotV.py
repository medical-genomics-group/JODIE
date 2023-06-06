# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru matplotlib
```
python plotV.py --p 5000 --k 4 --g 2 --resultsdir chr21-22/
"""

import sys
import argparse
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from loguru import logger

def main(p, k, groups, truedir, resultsdir):

    ## true values
    #file = np.loadtxt('results/MC/chr22_1000causal/data/trueV-meanV-varV_ondata.txt')
    #true_V = file[:k,:]
    #trace_V = file[k:2*k,:]
    #trace_V_std = file[2*k:,:]
    true_V = np.loadtxt(truedir+'/true_V.txt')
    trace_V = np.loadtxt(resultsdir+'/mean_V.txt')
    trace_V_std = np.loadtxt(resultsdir+'/var_V.txt')

    # improve plots by setting fontsizes and removing white space
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['ytick.labelsize']=15
    plt.rcParams['xtick.labelsize']=15
    plt.rcParams["text.usetex"]=True
    plt.rcParams['legend.handlelength'] = 0 # remove errorbar from legend


    # estimated vs true V
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = -50  # pad is in points...
    figV, plotV = plt.subplots(ncols=groups,figsize=(12*groups,10), tight_layout=True)
    labels = [r'$\mathbf{V_{11}}$', r'$\mathbf{V_{12}}$', r'$\mathbf{V_{13}}$', r'$\mathbf{V_{14}}$',
        r'$\mathbf{V_{21}}$', r'$\mathbf{V_{22}}$', r'$\mathbf{V_{23}}$', r'$\mathbf{V_{24}}$',
        r'$\mathbf{V_{31}}$', r'$\mathbf{V_{32}}$', r'$\mathbf{V_{33}}$', r'$\mathbf{V_{34}}$',
        r'$\mathbf{V_{41}}$', r'$\mathbf{V_{42}}$', r'$\mathbf{V_{43}}$', r'$\mathbf{V_{44}}$',]

    for g in range(groups):
        z = 0
        for i in range(k):
            for j in range(k):
                logger.info(f"{g=}, {i=}, {g*k+i=}, {j=}")
                if z==0:
                    plotV.errorbar(z, trace_V[g*k+i, j], yerr= np.sqrt(trace_V_std[g*k+i,j]),marker='o', color='red', mfc='None', label='est.')
                    plotV.errorbar(z, true_V[g*k+i, j], marker='x', color='gray', mfc='None', label='true')
                else:
                    plotV.errorbar(z, trace_V[g*k+i, j], yerr= np.sqrt(trace_V_std[g*k+i,j]),marker='o', color='red', mfc='None')
                    plotV.errorbar(z, true_V[g*k+i, j], marker='x', color='gray', mfc='None')
                z += 1
        title = 'chr'+str(g+22)
        plotV.set(ylabel='values', title=title)
        plotV.legend(loc=9, fontsize=18, framealpha=0., bbox_to_anchor=[0.5, 0.9])
        plt.sca(plotV)
        plotV.set(ylim=(-0.1, 0.5))
        plt.axhline(y=0., color='black', linestyle='--')
        plt.xticks(np.arange(k*k), labels, fontsize=15, weight='bold')
    figV.savefig(resultsdir+'/V.png')

##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Plotting Gibbs results.')
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--k', type=int, help='number of family members', required = True)
    parser.add_argument('--g', type=int, help='number of groups', required = True)
    parser.add_argument('--resultsdir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--truedir', type=str, help='path to directory where the true values are stored', required = True)
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
    main(p = args.p, # number of markers
        k = args.k, # number of family members
        groups = args.g, # number of groups
        resultsdir = args.resultsdir, # path to directory where results are stored
        truedir = args.truedir,
        )
    logger.info("Done.")