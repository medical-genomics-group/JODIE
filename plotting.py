# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru matplotlib
```
python plotting.py --p 10000 --k 3 --g 1 --inputdir MC --resultsdir results
"""

import sys
import argparse
import json
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from loguru import logger

def get_id(filename):
    with open(filename) as f:
        list = json.load(f)
    f.close()
    return list

def main(p, k, groups, inputdir, resultsdir):

    ## load files with indices
    id1 = get_id(inputdir+"/id1.txt")
    id2 = get_id(inputdir+"/id2.txt")
    id3 = get_id(inputdir+"/id3.txt")
    id4 = get_id(inputdir+"/id4.txt")
    id5 = get_id(inputdir+"/id5.txt")
    id6 = get_id(inputdir+"/id6.txt")

    X0 = np.load(inputdir+"/genotype.npz")
    Xx = np.array(X0.f.arr_0) #, dtype='int32')
    n = Xx.shape[0]
    logger.info(f"{n=}")
    X = np.zeros((n, k*p))
    for i in range(p):
        X[id1[i], k*i] = 1
        X[id2[i], k*i] = -1
        X[id3[i], k*i+1] = 1
        X[id4[i], k*i+1] = -1
        X[id5[i], k*i+2] = 1
        X[id6[i], k*i+2] = -1
    X = stats.zscore(X, axis=0, ddof=1)
    scale = tuple((np.dot(Xx[:,i], X[:,i])/(n-1)) for i in range(k*p))

    ## true values
    true_V = np.loadtxt(inputdir+'/true_V.txt')
    true_sigma2 = np.loadtxt(inputdir+'/true_sigma2.txt')
    true_beta = np.loadtxt(inputdir+'/true_betas.txt')
    #estimated values
    mean_beta = np.loadtxt(resultsdir+'/mean_beta.txt')
    mean_betaX = np.loadtxt('resultsX/mean_beta.txt')
    trace_V = np.loadtxt(resultsdir+'/mean_V.txt')
    trace_V_std = np.loadtxt(resultsdir+'/var_V.txt')
    trace_sigma2 = np.loadtxt(resultsdir+'/mean_sigma2.txt')
    trace_sigma2_std = np.loadtxt(resultsdir+'/var_sigma2.txt')

    true_alpha = (np.array(scale)*true_beta.flatten()).reshape(p,k)
    #true_V = np.matmul(true_alpha.T, true_alpha)

    MSEa = np.sum(np.abs(mean_beta - true_alpha), axis=0)
    MSEb = np.sum(np.abs(mean_betaX - true_beta), axis=0)
    MSEab = np.sum(np.abs(mean_beta - true_beta), axis=0)
    logger.info(f"{MSEa=}")
    logger.info(f"{MSEb=}")
    logger.info(f"{MSEab=}")
    MSEa2 = np.sum((mean_beta - true_alpha)**2, axis=0)
    MSEb2 = np.sum((mean_betaX - true_beta)**2, axis=0)
    MSEab2 = np.sum((mean_beta - true_beta)**2, axis=0)
    logger.info(f"{MSEa2=}")
    logger.info(f"{MSEb2=}")
    logger.info(f"{MSEab2=}")



    # improve plots by setting fontsizes and removing white space
    plt.rcParams['axes.labelsize'] = 18
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['ytick.labelsize']=15
    plt.rcParams['xtick.labelsize']=15
    plt.rcParams["text.usetex"]=True
    plt.rcParams['legend.handlelength'] = 0 # remove errorbar from legend
 
    # estimated vs true beta
    fig, ax = plt.subplots(figsize=(8,8), tight_layout=True)
    ax.scatter(x=true_beta[:,0], y=mean_beta[:,0], facecolors='none', edgecolors='grey', label="child")
    ax.scatter(x=true_beta[:,1], y=mean_beta[:,1], marker="x", color="red", label="mother")
    ax.scatter(x=true_beta[:,2], y=mean_beta[:,2], facecolors='none', edgecolors='blue', label="father")
    ax.set(xlabel="true effect sizes", ylabel="estimated effect sizes")
    ax.legend(loc=4, fontsize=18, framealpha=0.)
    #plt.margins(0,0)
    ax.set(xlim=(-0.2, 0.2), ylim=(-0.2, 0.2))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', ls="--")
    fig.savefig(resultsdir+'/est_vs_true_beta.png')
    ax.set(xlim=(-0.02, 0.02), ylim=(-0.02, 0.02))
    fig.savefig(resultsdir+'/est_vs_true_beta_zoom.png')

    # estimated vs true alpha
    fig, ax = plt.subplots(figsize=(8,8), tight_layout=True)
    ax.scatter(x=true_alpha[:,0], y=mean_beta[:,0], facecolors='none', edgecolors='grey', label="child")
    ax.scatter(x=true_alpha[:,1], y=mean_beta[:,1], marker="x", color="red", label="mother")
    ax.scatter(x=true_alpha[:,2], y=mean_beta[:,2], facecolors='none', edgecolors='blue', label="father")
    ax.set(xlabel="true effect sizes", ylabel="estimated effect sizes")
    ax.legend(loc=4, fontsize=18, framealpha=0.)
    #plt.margins(0,0)
    ax.set(xlim=(-0.2, 0.2), ylim=(-0.2, 0.2))
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color='black', ls="--")
    fig.savefig(resultsdir+'/est_vs_true_alpha.png')
    ax.set(xlim=(-0.02, 0.02), ylim=(-0.02, 0.02))
    fig.savefig(resultsdir+'/est_vs_true_alpha_zoom.png')

    # estimated vs true V
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlepad'] = -50  # pad is in points...
    figV, plotV = plt.subplots(ncols=2,figsize=(22,10), tight_layout=True)
    labels = [r'$\mathbf{V_{11}}$', r'$\mathbf{V_{12}}$', r'$\mathbf{V_{13}}$', r'$\mathbf{V_{21}}$', r'$\mathbf{V_{22}}$', r'$\mathbf{V_{23}}$', r'$\mathbf{V_{31}}$', r'$\mathbf{V_{32}}$', r'$\mathbf{V_{33}}$']
    for g in range(groups):
        z = 0
        for i in range(k):
            for j in range(k):
                if z==0:
                    plotV[g].errorbar(z, trace_V[g*k+i, j], yerr= np.sqrt(trace_V_std[g*k+i,j]),marker='o', color='red', mfc='None', label='est.')
                    plotV[g].errorbar(z, true_V[g*k+i, j], marker='x', color='black', label='true')
                else:
                    plotV[g].errorbar(z, trace_V[g*k+i, j], yerr= np.sqrt(trace_V_std[g*k+i,j]),marker='o', color='red', mfc='None')
                    plotV[g].errorbar(z, true_V[g*k+i, j], marker='x', color='black')
                z += 1
        title = 'group '+str(g)
        plotV[g].set(ylabel='values', title=title)
        plotV[g].legend(loc=9, fontsize=18, framealpha=0., bbox_to_anchor=[0.5, 0.9])
        plt.sca(plotV[g])
        plt.xticks(np.arange(k*k), labels, fontsize=15, weight='bold')
    figV.savefig(resultsdir+'/V.png')

    # estimated vs true sigma
    figS, plotS = plt.subplots(figsize=(10,9.6), tight_layout=True)
    plotS.errorbar(1, trace_sigma2, yerr=np.sqrt(trace_sigma2_std), marker='o', color='red', mfc='None', label='est.')
    plotS.errorbar(1, true_sigma2, marker='x', color='black', label='true')
    plotS.set(ylabel='values', xlabel=r'$\mathbf{\sigma^{2}}$', ylim=(0.,1.))
    figS.legend(fontsize=18, framealpha=0.)
    figS.savefig(resultsdir+'/sigma2.png')

##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Plotting Gibbs results.')
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--k', type=int, help='number of family members', required = True)
    parser.add_argument('--g', type=int, help='number of groups', required = True)
    parser.add_argument('--inputdir', type=str, help='path to directory where true values are stored', required = True)
    parser.add_argument('--resultsdir', type=str, help='path to directory where the results are stored', required = True)
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
        inputdir = args.inputdir, # path to directory where true values are stored
        resultsdir = args.resultsdir, # path to directory where results are stored
        )
    logger.info("Done.")