# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru matplotlib
```
python quick_check.py --k 4 --g 22 --resultsdir results/full/
"""

import sys
import argparse
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from loguru import logger

def main(k, groups, resultsdir, trace):

    trace_V0 = np.loadtxt(resultsdir+'/V_500.txt')
    trace_V1 = np.loadtxt(resultsdir+'/V_1000.txt')
    trace_V2 = np.loadtxt(resultsdir+'/mean_V_1500.txt')
    trace_V2_std = np.loadtxt(resultsdir+'/var_V_1500.txt')
    trace_V = np.loadtxt(resultsdir+'/mean_V.txt')
    trace_V_std = np.loadtxt(resultsdir+'/var_V.txt')
    trace_s0 = np.loadtxt(resultsdir+'/sigma2_500.txt')
    trace_s1 = np.loadtxt(resultsdir+'/sigma2_1000.txt')
    trace_s2 = np.loadtxt(resultsdir+'/mean_sigma2_1500.txt')
    trace_s2_std = np.loadtxt(resultsdir+'/var_sigma2_1500.txt')
    trace_s = np.loadtxt(resultsdir+'/mean_sigma2.txt')
    trace_s_std = np.loadtxt(resultsdir+'/var_sigma2.txt')

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
    
    labels = [r'$\mathbf{V_{11}}$', r'$\mathbf{V_{12}}$', r'$\mathbf{V_{13}}$', r'$\mathbf{V_{14}}$',
        r'$\mathbf{V_{21}}$', r'$\mathbf{V_{22}}$', r'$\mathbf{V_{23}}$', r'$\mathbf{V_{24}}$',
        r'$\mathbf{V_{31}}$', r'$\mathbf{V_{32}}$', r'$\mathbf{V_{33}}$', r'$\mathbf{V_{34}}$',
        r'$\mathbf{V_{41}}$', r'$\mathbf{V_{42}}$', r'$\mathbf{V_{43}}$', r'$\mathbf{V_{44}}$',]

    for g in range(groups):
        figV, plotV = plt.subplots(figsize=(12,10), tight_layout=True)
        z = 0
        for i in range(k):
            for j in range(k):
                logger.info(f"{g=}, {i=}, {g*k+i=}, {j=}")
                if z==0:
                    plotV.errorbar(z, trace_V0[g*k+i, j], marker='o', color='red', mfc='None', label='500 it.') #yerr=np.sqrt(trace_V_std[g*k+i, j])
                    plotV.errorbar(z, trace_V1[g*k+i, j], marker='x', color='blue', mfc='None', label='1000 it.')
                    plotV.errorbar(z, trace_V2[g*k+i, j], yerr=np.sqrt(trace_V2_std[g*k+i, j]), marker='v', color='green', mfc='None', label='1500 it.')
                    plotV.errorbar(z, trace_V[g*k+i, j], yerr=np.sqrt(trace_V_std[g*k+i, j]), marker='^', color='black', mfc='None', label='post.')
                else:
                    plotV.errorbar(z, trace_V0[g*k+i, j],  marker='o', color='red', mfc='None') #yerr=np.sqrt(trace_V_std[g*k+i, j]),
                    plotV.errorbar(z, trace_V1[g*k+i, j], marker='x', color='blue', mfc='None')
                    plotV.errorbar(z, trace_V2[g*k+i, j], yerr=np.sqrt(trace_V2_std[g*k+i, j]), marker='v', color='green', mfc='None')
                    plotV.errorbar(z, trace_V[g*k+i, j], yerr=np.sqrt(trace_V_std[g*k+i, j]), marker='^', color='black', mfc='None')
                z += 1
        title = 'group '+str(g)
        plotV.set(ylabel='values', title=title)
        plotV.legend(loc=9, fontsize=18, framealpha=0., bbox_to_anchor=[0.5, 0.9])
        plt.sca(plotV)
        plotV.set(ylim=(-0.2, 0.5))
        plt.axhline(y=0., color='black', linestyle='--')
        plt.xticks(np.arange(k*k), labels, fontsize=15, weight='bold')
        figV.savefig(resultsdir+'/V_it_'+str(g)+'group.png')
        plt.close()

    figs, plots = plt.subplots(figsize=(10,10), tight_layout=True)
    plots.errorbar(0, trace_s0, marker='o', color='red', mfc='None', label='500 it.') #yerr=np.sqrt(trace_s_std)
    plots.errorbar(0, trace_s1, marker='x', color='blue', mfc='None', label='1000 it.')
    plots.errorbar(0, trace_s2, yerr=np.sqrt(trace_s2_std), marker='v', color='green', mfc='None', label='1500 it.')
    plots.errorbar(0, trace_s, yerr=np.sqrt(trace_s_std), marker='^', color='black', mfc='None', label='post.')
    plots.legend(loc=9, fontsize=18, framealpha=0., bbox_to_anchor=[0.5, 0.9])
    plt.sca(plots)
    plots.set(ylim=(-0., 1.), xlabel=r'$\mathbf{\sigma^{2}}$')
    figs.savefig(resultsdir+'/sigma2_it_'+str(g)+'group.png')
    plt.close()

    if trace==True:
        for g in range(groups):
            fig, plot = plt.subplots(figsize=(12,10), tight_layout=True)
            tr = np.loadtxt(resultsdir+'/trace_V'+str(g)+'_1000.txt')[:1000]
            plt.plot(tr, label='group '+str(g))
            plot.set(ylim=(-0., 0.5), ylabel='V')
            plt.legend()
            fig.savefig(resultsdir+'/traceV'+str(g)+'_it1000.png')
            plt.close()
            
        fig, plot = plt.subplots(figsize=(12,10), tight_layout=True)
        trZ = np.loadtxt(resultsdir+'/trace_Z.txt')#[:1000]
        plt.plot(trZ)
        plot.set(ylabel='Z')
        #plt.legend()
        fig.savefig(resultsdir+'/traceZ.png')
        plt.close()

        fig, plot = plt.subplots(figsize=(12,10), tight_layout=True)
        trs2 = np.loadtxt(resultsdir+'/trace_sigma21000.txt')[:1000]
        plt.plot(trs2)
        plot.set(ylim=(-0., 1),ylabel='sigma2')
        #plt.legend()
        fig.savefig(resultsdir+'/trace_sigma2_it1000.png')
        plt.close()

##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Plotting Gibbs results.')
    parser.add_argument('--k', type=int, help='number of family members', required = True)
    parser.add_argument('--g', type=int, help='number of groups', required = True)
    parser.add_argument('--resultsdir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--trace', type=bool, default=False, help="plot traces")
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
        k = args.k, # number of family members
        groups = args.g, # number of groups
        resultsdir = args.resultsdir, # path to directory where results are stored
        trace = args.trace,
        )
    logger.info("Done.")