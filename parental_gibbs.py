# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy tqdm mpi4py welford matplotlib
```
run with 4 processes (given by -n):
mpiexec -n 4 python -m mpi4py parental_gibbs.py --n 20000 --p 10000 --g 10000 --iters 5 --burnin 1 --x MC/genotype.npz --y MC/phenotype.txt --resultdir results --inputdir MC
"""
"""
Needed input: 
1. index files of contrasted matrices (with consecutive numbering) always iterating 1 and -1
2. genotype matrix normalized (not contrasted) 
3. col std?
"""
### CHANGE: speed-up
### XtX calculation
### epsilon updating

import sys
import argparse
import json
import welford
from mpi4py import MPI
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.linalg.blas as blas
import matplotlib.pyplot as plt
from loguru import logger
from tqdm import trange


def sample_mvn(k, xe, xtx, beta, sigma2, V_inv, scale, pi_ratio, rng):

    ## sample betas from multivariate normal distribution
    # calculate omega*
    omega_star_inv = xtx/sigma2 + V_inv
    omega_star = linalg.inv(omega_star_inv)
    #logger.info(f"{omega_star=}")
    # calculate mu*: check order of multiplication
    mu_star = np.matmul(omega_star, (xe + np.matmul(xtx, beta))) / sigma2
    #logger.info(f"{mu_star=}")
    # calculate exclusion probability
    # calc term in exponential (cut of at 100 so that exp does not overflow)
    f = np.minimum(
        np.linalg.multi_dot([mu_star.T, omega_star_inv, mu_star])/2, 
        100)

    tau = calc_tau(f, pi_ratio, omega_star, V_inv)
    #logger.info(f"{tau=}")
    if tau > rng.uniform(0,1):
        tracker = 0
        beta = np.zeros((1,k))
    else:
        tracker = 1
        beta = rng.multivariate_normal(
            mu_star,
            omega_star,
            method="cholesky", 
            check_valid="ignore"
        )

    return tracker, beta


def calc_tau(f, pi_ratio, omega_star, V_inv):
    return (pi_ratio / (
        pi_ratio
        + np.linalg.det(V_inv)**(1/2)
        * np.linalg.det(omega_star)**(1/2)
        * np.exp(f) )
    )


def sample_V(beta, L, D, k, Z, a, b, s):
    # sample covariances according to
    # https://doi.org/10.1198/jcgs.2009.08095
    #logger.info(f"{np.matmul(beta.T, beta)=}")
    beta_2 = Z*np.matmul(beta.T, beta)
    ## all zero groups
    if np.all(beta_2 < 10e-15):
        V = np.zeros((k,k))
        Vinv = 10e+15*np.eye(k)
        L = np.eye(k)
        D = np.ones(k)
    ## non-zero groups
    else:
        ww = np.linalg.multi_dot([L, beta_2, L.T])

        for i in range(k):
            #sample elements of D
            D[i] = stats.invgamma.rvs(a= a/2 + Z, 
                scale=a*b/2 + ww[i, i]
                )
            #sample elements of L
            if i >= 1:
                si = np.linalg.inv((1 / D[i]) * beta_2[0:i, 0:i] + s*np.eye(i))
                mi = - (si / D[i]) @ beta_2[0:i, i]
                L[i, :i] = np.random.multivariate_normal(mi.flatten(), si).reshape((1, i))

        Vinv = np.linalg.multi_dot([L.T, np.diag(1/D), L])
        V = linalg.inv(Vinv)
        #logger.info(f"{V=}")
    
    return V, Vinv, L, D


def get_id(filename):
    with open(filename) as f:
        list = json.load(f)
    f.close()
    return list


def sample_invGamma(a, b, a1, b1):
    return stats.invgamma.rvs(a + a1, scale=(b + b1))




def main(n, p, k, groups, iters, burnin, xfile, yfile, itc, resultdir, inputdir):
    
    # random generator
    rng = np.random.default_rng()

    ## load files with indices
    id1 = get_id(inputdir+"/id1.txt")
    id2 = get_id(inputdir+"/id2.txt")
    id3 = get_id(inputdir+"/id3.txt")
    id4 = get_id(inputdir+"/id4.txt")
    id5 = get_id(inputdir+"/id5.txt")
    id6 = get_id(inputdir+"/id6.txt")
    id7 = get_id(inputdir+"/id5.txt") ##CHANGE
    id8 = get_id(inputdir+"/id6.txt") ##CHANGE

    id_plus = {0: id1, 1: id3, 2: id5, 3: id7}
    id_minus = {0: id2, 1: id4, 2: id6, 3: id8}

    # cstd = np.loadtxt(inputdir+'/col_sigmac.txt')
    # xstd = np.loadtxt(inputdir+'/col_sigmax.txt')
    # cmean = np.loadtxt(inputdir+'/col_meanc.txt')
    # xmean = np.loadtxt(inputdir+'/col_meanx.txt')

    ## true values
    true_V = np.loadtxt(inputdir+'/true_V.txt')
    true_sigma2 = np.loadtxt(inputdir+'/true_sigma2.txt')

    ## open phenotype file
    epsilon = np.loadtxt(yfile)
    # open genotype file
    contrast = 1
    X0 = np.load(xfile)
    X = np.array(X0.f.arr_0, order="F") #, dtype='int32')
    # X = np.zeros((n, k*p))
    # for i in range(p):
    #     X[id1[i], k*i] = 1
    #     X[id2[i], k*i] = -1
    #     X[id3[i], k*i+1] = 1
    #     X[id4[i], k*i+1] = -1
    #     X[id5[i], k*i+2] = 1
    #     X[id6[i], k*i+2] = -1
    # X = stats.zscore(X, axis=0, ddof=1)
    scale = np.ones(p)
    # scale = tuple((np.dot(Xx[:,i], X[:,i])/(n-1)) for i in range(k*p))
    # scale = np.array(scale)
    # logger.info(f"{scale=}")
    # #logger.info(f"{cstd/xstd=}")
    XtX = np.zeros((k*p,k))
    for j in range(p):
        XtX[j*k:k*(j+1),:] = np.matmul(X[:,j*k:k*(j+1)].T, X[:,j*k:k*(j+1)])

    ## groups
    G = len(groups)
    group_idx = np.repeat(np.arange(G), groups)
    assert p == np.sum(groups)
    true_V = np.split(true_V, G, axis=0)
    logger.info(f"Problem has dimensions {n=}, {p=}, {k=}, {G=}.")
    

    # initalize parameters
    init = {
        "beta": np.zeros((p, k)),
        "V": true_V, #np.repeat([0.5*np.eye(k)], G, axis=0),
        "sigma2": true_sigma2,
        "pi": np.repeat(np.array([[0.5, 0.5]]), G, axis=0),
        "D": np.array(G*[np.ones(k)]),
        "L": np.array(G*[np.eye(k)]),
        "mu": 0,
    }
    hypers = {
        "av": 2,
        "bv": 0.1,
        "sv": 0.0001,
        "ae": 1/n,
        "be": 0.1,
    }

    Z_sum = groups
    V_inv = np.zeros((G,k,k))
    pi_ratio = np.ones(G)

    beta = init["beta"]
    V = np.array(init["V"])
    for g in range(G):
        V_inv[g] = linalg.inv(V[g])
    sigma2 = np.array(init["sigma2"])
    pi = init["pi"]
    mu = init["mu"]
    L = init["L"]
    D = init["D"]
    logger.info(f"initialize V as {V=}")
    logger.info(f"initialize sigma2 as {sigma2=}")
        
    # generate storage using the Welford package
    w_beta = welford.Welford()
    w_V = welford.Welford()
    w_sigma2 = welford.Welford()
    # storage
    trace_V = np.zeros((iters,G,k,k))
    trace_sigma2 = np.zeros((iters))
    trace_Z = np.zeros((iters, G))

    logger.info(f"Running Gibbs with {iters=} and {burnin=}")
    for it in trange(iters, desc="Main loop"):

        # sample intercept term
        epsilon += mu
        mu = np.mean(epsilon) if it == 0 else rng.normal(np.mean(epsilon), np.sqrt(sigma2/n))
        epsilon -= mu
        # logger.info(f"{it=}, {mu=}")

        #calculate pi ratio
        pi_ratio = pi[:,0]/pi[:,1]

        ## set number of non-zero markers to 0 before each iteration
        Z = np.zeros(G, dtype='i')

        # loop trough all markers randomly
        ### needs to be changed for groups
        rj = np.arange(p)
        rng.shuffle(rj)
        for j in rj:
                
                #get group index
                g = group_idx[j]
                # calculate X.T@epsilon using indices
                # plus = tuple(epsilon[(id_plus[i][j])].sum() for i in range(k))
                # minus = tuple(epsilon[id_minus[i][j]].sum() for i in range(k))
                # if contrast:
                #     xe = 1/cstd[k*j:k*(j+1)]*(np.array(plus) - np.array(minus) - cmean[k*j:k*(j+1)] * epsilon.sum(axis=0))
                # else:
                #     xe = 1/xstd[k*j:k*(j+1)]*(np.array(plus) + 2*np.array(minus) - xmean[k*j:k*(j+1)] * epsilon.sum(axis=0))
                
                #logger.info(f"{xe=}")
                xe = np.matmul(X[:,j*k:k*(j+1)].T, epsilon)
                #xtx = np.matmul(X[:,j*k:k*(j+1)].T, X[:,j*k:k*(j+1)])
                #logger.info(f"{XtX[j*k:k*(j+1), j*k:k*(j+1)]=}")
                #logger.info(f"{xe=}")
                #logger.info(f"{xtx=}")
                prev_beta = beta[j].copy()
                #logger.info(f"{j=}, {prev_beta=}")

                ## sample beta
                tracker, beta[j] = sample_mvn(
                    k, 
                    xe, 
                    XtX[j*k:k*(j+1), :],
                    prev_beta, 
                    sigma2, 
                    V_inv[g]*Z_sum[g], 
                    scale[j*k:k*(j+1)],
                    pi_ratio[g],
                    rng)
                #beta[j] *= scale[j*k:k*(j+1)]
                #logger.info(f"{j=}, {tracker=}, {beta[j,:]=}")

                # udpate number of non-zero betas
                Z[g] += tracker
                # calculate difference in epsilon
                #epsilon += np.matmul(Xx[:,j*k:k*(j+1)], (prev_beta - beta[j, :]))
                #epsilon += blas.dgemv(1, a=Xx[:,j*k:k*(j+1)], x=(prev_beta - beta[j]))
                epsilon += blas.dgemv(1, X[:,j*k:k*(j+1)], (prev_beta - beta[j]))
                #logger.info(f"{np.var(epsilon)=}")

        # update number of non-zero markers
        Z_sum[g] = Z[g]
        logger.info(f"{Z_sum=}")
        # update pi for each group
        for g in range(G):
            if Z_sum[g] == 0:
                pi[g] = rng.dirichlet((p-Z_sum[g]-1, 1))
            elif Z_sum[g] == p:
                pi[g] = rng.dirichlet((p-Z_sum[g]+1, Z_sum[g]-1))
            else:
                pi[g] = rng.dirichlet((p-Z_sum[g], Z_sum[g]))

            #update V
            start_g = 0 if g == 0 else groups[g]
            end_g = p if g == G-1 else groups[g+1]
            V[g], V_inv[g], L[g], D[g] = sample_V(
                beta[start_g:end_g, :],
                L[g], D[g],  
                k, Z_sum[g],
                hypers["av"],
                hypers["bv"],
                hypers["sv"]
            )

        # update sigma2
        ## fast way
        sigma2 = np.dot(epsilon.T, epsilon)/np.mean(stats.chi2.rvs(n-2))
        #logger.info(f"{sigma2=}")
        # slower  
        # sigma2 = sample_invGamma(
        #     hypers["ae"], hypers["be"], 
        #     n / 2, np.dot(epsilon.T, epsilon)/ 2 
        # )
        # logger.info(f"{sigma2=}")
        #logger.info(f"{beta.shape=}")

        # store stuff
        trace_V[it] = V
        trace_sigma2[it] = sigma2
        trace_Z[it] = Z_sum
        if it >= burnin:
            w_beta.add(beta)
            w_sigma2.add(sigma2)
            w_V.add(V.reshape(G*k,k))

    mean_beta = np.array(w_beta.mean)
    var_beta = np.array(w_beta.var_s)
    mean_V = np.array(w_V.mean)
    var_V = np.array(w_V.var_s)
    mean_sigma2 = np.array(w_sigma2.mean)
    var_sigma2 = np.array(w_sigma2.var_s)
    logger.info(f"{np.mean(trace_Z[burnin:], axis=0)=}\n")
    logger.info(f"{true_V=}")
    logger.info(f"{mean_V=}")
    logger.info(f"{var_V=}")
    logger.info(f"{np.matmul(mean_beta.T, mean_beta)=}\n")
    logger.info(f"{true_sigma2=}")
    logger.info(f"{mean_sigma2=}")
    ### save
    np.savetxt(resultdir+'/mean_V.txt', mean_V)
    np.savetxt(resultdir+'/var_V.txt', var_V)
    np.savetxt(resultdir+'/mean_sigma2.txt', mean_sigma2.reshape(1,1))
    np.savetxt(resultdir+'/var_sigma2.txt', var_sigma2.reshape(1,1))
    np.savetxt(resultdir+'/mean_beta.txt', mean_beta)
    np.savetxt(resultdir+'/var_beta.txt', var_beta)
    np.savetxt(resultdir+'/trace_Z.txt', trace_Z)

    # Plotting sigma2 results
    logger.info("Plotting sigma2 results.")
    t = np.arange(iters)
    figS, axS = plt.subplots()
    axS.plot(t, trace_sigma2)
    axS.set(ylabel='sigma2', xlabel='iterations')
    axS.get_figure().savefig(resultdir+'/trace_sigma2.png')

    # Plotting V results
    logger.info("Plotting V results.")
    for g in range(G):
        figV, axV = plt.subplots()
        for i in range(k):
            for j in range(k):
                axV.plot(t, trace_V[:,g,i,j])
        axV.set(ylabel='V', xlabel='iterations')
        axV.get_figure().savefig(resultdir+'/trace_V_'+str(g)+'.png')

    # Plotting Z results
    logger.info("Plotting Z results.")
    t = np.arange(iters)
    figZ, axZ = plt.subplots()
    axZ.plot(t, trace_Z)
    axZ.set(ylabel='Z', xlabel='iterations')
    axZ.get_figure().savefig(resultdir+'/trace_Z.png')


##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parental Gibbs sampler.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--k', type=int, default=4, help='number of family member incl. POO (2,3 or 4; default=4=')
    parser.add_argument('--g', nargs='+', type=int, help='number of markers in each group', required=True)
    parser.add_argument('--iters', type=int, default=10000, help='number of iterations (default = 10000)')
    parser.add_argument('--burnin', type=int, default=1000, help='number of iterations in burnin (default = 1000)')
    parser.add_argument('--itc', type=int, default=2, help='counter for updating epsilon (default=2)')
    parser.add_argument('--x', type=str, help='genotype matrix filename in which file format?', required = True)
    parser.add_argument('--y', type=str, help='phenotype matrix filename in which file format?', required = True)
    parser.add_argument('--resultdir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--inputdir', type=str, help='path to directory where indices are stored')
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
        iters = args.iters, # number of iterations
        burnin = args.burnin, # number of iterations in burnin period
        groups = np.array(args.g), # number of markers in each group
        itc = args.itc, # counter for updating epsilon (after number of processes times itc markers)
        xfile = args.x, # genotype file
        yfile = args.y, # phenotype file
        resultdir = args.resultdir, # path to results directory
        inputdir = args.inputdir # path to directory with true values
        ) 
    logger.info("Done.")
