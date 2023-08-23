# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru matplotlib
```
python plot_all_traits_1group.py 
--resultsdir Estonia/height-full-1group/ Estonia/bmi-full-1group/ Estonia/bp-full-1group/ 
--traits height BMI BP --outdir Estonia/plots
````
--resultsdir paths to directories where results are stored (required)
--traits name of traits to put on legend (required)
--outdir name of directory where plots are stored (required)
"""

import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

def main(k, resultsdir, traits, outdir):

    # # improve plots by setting fontsizes and removing white space
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['ytick.labelsize']=18
    plt.rcParams['xtick.labelsize']=18
    plt.rcParams['legend.handlelength'] = 0 # remove errorbar from legend
    plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams["text.usetex"]=True
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    #plt.rcParams['axes.titlepad'] = -20  # pad is in points...

    # V labels
    labels = [r'$\mathbf{\boldsymbol{\beta}_{c}^2}$', r'$\mathbf{\boldsymbol{\beta}_{m}^2}$', r'$\mathbf{\boldsymbol{\beta}_{f}^2}$', r'$\mathbf{\boldsymbol{\beta}_{i}^2}$',
            r'$\mathbf{\boldsymbol{\beta}_{c}\boldsymbol{\beta}_{m}}$', r'$\mathbf{\boldsymbol{\beta}_{c}\boldsymbol{\beta}_{f}}$', r'$\mathbf{\boldsymbol{\beta}_{i}\boldsymbol{\beta}_{c}}$', 
            r'$\mathbf{\boldsymbol{\beta}_{i}\boldsymbol{\beta}_{m}}$', r'$\mathbf{\boldsymbol{\beta}_{i}\boldsymbol{\beta}_{f}}$', r'$\mathbf{\boldsymbol{\beta}_{m}\boldsymbol{\beta}_{f}}$',]

    # parameters for plotting
    order=np.array([[0,0], [1,1], [2,2], [3,3], [1,0], [2,0], [3,0], [3,1], [3,2], [2,1]])
    shift=0.05
    marker=['o', 's', 'v', '^', '*']
    color=['gray', 'orange', 'tab:cyan', 'mediumvioletred', 'darkgreen']

    # V
    figV, plotV = plt.subplots(figsize=(15,5), tight_layout=True)
    for i in range(len(resultsdir)):
        mean_V = np.loadtxt(resultsdir[i]+'mean_V.txt')
        var_V = np.loadtxt(resultsdir[i]+'var_V.txt')
        logger.info(f"{mean_V=}")
        # change sign of V_fi
        mean_V[3,2] *= -1
        mean_V[2,3] *= -1
        mean_V = np.tril(mean_V)
        var_V = np.tril(var_V)
        logger.info(f"{mean_V=}")
        #logger.info(f"{var_V=}")

        z = 0
        for row, col in order:
            #logger.info(f"{z+i*shift=}, {mean_V[row, col]=}, {var_V[row, col]=}")
            if z==0:
                plotV.errorbar(z+i*shift-(5*shift/2), mean_V[row, col], yerr=np.sqrt(var_V[row, col]), marker=marker[i], color=color[i], label=traits[i])
            else:
                plotV.errorbar(z+i*shift-(5*shift/2), mean_V[row, col], yerr=np.sqrt(var_V[row, col]), marker=marker[i], color=color[i])
            z += 1

    plotV.set(ylabel='values')
    plt.title("Estonian Biobank, 10,512 trios", loc='right')
    plotV.legend(loc=1, fontsize=18, framealpha=0.)#, bbox_to_anchor=[0.9, 0.9])
    plt.sca(plotV)
    plotV.set(ylim=(-0.05, 0.25))
    plt.axhline(y=0., color='black', linestyle='--')
    plt.axvline(x=3.5, color='black', linestyle='--')
    plt.text(6, 0.225, "covariances", fontsize=18)
    plt.text(1, 0.225, "variances", fontsize=18)
    plt.xticks(np.arange(10), labels, fontsize=15, weight='bold')
    figV.savefig(outdir+'/V_alltraits.png')
    plt.close()

    # V normalized by h2 and correlations
    ## propagate error bars by using upper and lower error
    figV, plotV = plt.subplots(figsize=(15,5), tight_layout=True)
    for i in range(len(resultsdir)):
        mean_V = np.loadtxt(resultsdir[i]+'mean_V.txt')
        var_V = np.loadtxt(resultsdir[i]+'var_V.txt')
        
        # correlation matrix
        logger.info(f"{mean_V=}")
        corr = ((1/np.sqrt(np.diag(mean_V)))*mean_V).T*(1/np.sqrt(np.diag(mean_V)))
        logger.info(f"{corr=}")
        # calculate error bands for correlation
        upper_V = mean_V + np.sqrt(var_V)
        upper_corr = ((1/np.sqrt(np.diag(upper_V)))*upper_V).T*(1/np.sqrt(np.diag(upper_V)))
        logger.info(f"{upper_corr=}")
        lower_V = mean_V - np.sqrt(var_V)
        lower_corr = ((1/np.sqrt(np.diag(lower_V)))*lower_V).T*(1/np.sqrt(np.diag(lower_V)))
        logger.info(f"{lower_corr=}")
        upper_cband = np.abs(upper_corr - corr)
        lower_cband = np.abs(- lower_corr + corr)
        
        # normalize variances by h2
        mean_V = np.tril(mean_V)
        var_V = np.tril(var_V)
        h2 = np.sum(mean_V) - mean_V[2,1] - mean_V[3,0]
        upper_band = np.abs(np.tril(lower_V) - mean_V)/h2
        lower_band = np.abs(- np.tril(upper_V) + mean_V)/h2
        mean_V /= h2
       
        # change sign of V_fi for plotting
        mean_V[3,2] *= -1
        corr[3,2] *= -1

        z = 0
        for row, col in order:
            #logger.info(f"{z+i*shift=}, {mean_V[row, col]=}, {var_V[row, col]=}")
            if row == col:
                V = mean_V
                err = np.array(list(zip([lower_band[row,col], upper_band[row,col]])))
            else:
                V = corr
                err = np.array(list(zip([lower_cband[row,col], upper_cband[row,col]])))

            if z==0:
                plotV.errorbar(z+i*shift-(5*shift/2), V[row, col], yerr=err, marker=marker[i], color=color[i], label=traits[i])
            else:
                plotV.errorbar(z+i*shift-(5*shift/2), V[row, col], yerr=err, marker=marker[i], color=color[i])
            z += 1

    plotV.set(ylabel='values')
    plt.title("Estonian Biobank, 10,512 trios", loc='right')
    plotV.legend(loc=3, fontsize=18, framealpha=0.)#, bbox_to_anchor=[0.9, 0.9])
    plt.sca(plotV)
    plotV.set(ylim=(-1., 1))
    plt.yticks(np.arange(-1., 1.1, step=0.2))
    plt.axhline(y=0., color='black', linestyle='--')
    plt.axvline(x=3.5, color='black', linestyle='--')
    plt.text(0.75, 0.85, r"variances scaled by $h^2$", fontsize=18)
    plt.text(6, 0.85, "correlations", fontsize=18)
    plt.xticks(np.arange(10), labels, fontsize=15, weight='bold')
    figV.savefig(outdir+'/Vnorm_alltraits.png')

    # sigma
    figs, plots = plt.subplots(figsize=(8,8), tight_layout=True)
    for i in range(len(resultsdir)):
        mean_s = np.loadtxt(resultsdir[i]+'mean_sigma2.txt')
        var_s = np.loadtxt(resultsdir[i]+'var_sigma2.txt')
        plots.errorbar(i*0.05-0.125, mean_s, yerr=np.sqrt(var_s), marker=marker[i], color=color[i], label=traits[i])
    plots.legend(loc=1, fontsize=18, framealpha=0.)#, bbox_to_anchor=[0.9, 0.9])
    plt.sca(plots)
    plt.title("Estonian Biobank, 10,512 trios", loc='right')
    plt.yticks(np.arange(0, 1.1, step=0.1))
    plt.axhline(y=0.5, color='black', linestyle='--')
    plots.set(ylim=(-0., 1.), xlim=(-0.5, 0.5), ylabel='values', xlabel=r'$\mathbf{\boldsymbol{\sigma}^{2}}$')
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off
    figs.savefig(outdir+'/sigma2_alltraits.png')
    plt.close()

##########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Plotting final results.')
    parser.add_argument('--resultsdir', nargs='+', type=str, help='paths to directories where the results are stored', required = True)
    parser.add_argument('--traits', nargs='+', type=str, help='name of traits in labels', required = True)
    parser.add_argument('--outdir', type=str, help='path to output directory', required=True)
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
        k = 4,
        resultsdir = args.resultsdir, # path to directory where results are stored
        traits = args.traits,
        outdir = args.outdir,
        )
    logger.info("Done.")
