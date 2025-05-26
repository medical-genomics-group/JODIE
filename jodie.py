# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy tqdm mpi4py welford matplotlib zarr dask pandas
```
execute with mpi with n processes:
mpiexec -n 4 python -m mpi4py jodie.py 
--n 10000 --p 20000 --k 4 --g 20000 
--iters 500 --burnin 100 
--x genotype.zarr/ --y phenotype.txt --resultdir results/
--rmid  list_missingpheno.txt
````
Options:
--n number of individuals (required)
--p number of markers (required)
--k number of genetic components (k=2,3,4); default=4
--xfile genotype file created in preprocessing_vcf_data.py (required)
--xtx squared genotype file created in calc_xtx.py (required)
--y phenotype file in txt format without header with individuals ordered in the same way as in the xfiles
--resultdir path to output directory (required) 
--rmid list in txt format with line number of individual with missing phenotype (according to line in genotype file)
--iters number of total iterations for the Gibbs sampler; default = 10,000
--burnin number of iterations in the burnin; default 1,000)
--g number of markers in each group; this assumes that markers are ordered in groups in sequence (either g or gindex is needed as input parameter, not both)
--gindex txt file with information about which group each marker belongs to in the same order as the markers in the genotype matrix (either g or gindex is needed as input parameter, not both)
--itc counter for updating residuals (after number of processes times itc markers); default = 2 (if set too high, the Gibbs sampler will diverge)
--restart default = False; Gibbs sampler can be restarted using results saved at it 1000; 
CAUTION: files for restarting are taken from resultdir, but new results will be overwrite old results in resultdir
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import argparse
from mpi4py import MPI
import numpy as np
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.linalg.blas as blas
import matplotlib.pyplot as plt
import welford
from loguru import logger
from tqdm import trange
import zarr
import dask.array as da
import pandas as pd


def sample_mvn(k, xe, xtx, beta, sigma2, V_inv, pi_ratio, rng):

    ## sample betas from multivariate normal distribution
    # calculate omega*
    omega_star_inv = xtx/sigma2 + V_inv
    omega_star = linalg.inv(omega_star_inv)
    # calculate mu*: check order of multiplication
    mu_star = np.matmul(omega_star, (xe + np.matmul(xtx, beta))) / sigma2
    # calculate exclusion probability
    # calc term in exponential (cut of at 100 so that exp does not overflow)
    f = np.minimum(
        np.linalg.multi_dot([mu_star.T, omega_star_inv, mu_star])/2, 
        100)

    tau = calc_tau(f, pi_ratio, omega_star, V_inv)
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
    beta_2 = np.matmul(beta.T, beta)
    ## all zero groups or if total beta variance below 0.1%
    if np.sum(np.diag(beta_2)) <= 0.001 or Z==0:
        V = np.zeros((k,k))
        Vinv = 10e+09*np.eye(k)
        L = np.eye(k)
        D = np.ones(k)
    ## non-zero groups
    else:
        beta_2*=Z
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
    
    return V, Vinv, L, D


def sample_invGamma(a, b, a1, b1):
    return stats.invgamma.rvs(a + a1, scale=(b + b1))


def main(n, p, k, groups, gindex, iters, burnin, xfiles, xtxfiles, yfile, itc, resultdir, rmid, restart):

    ## groups
    if gindex:
        group_idx = np.loadtxt(gindex, dtype='int8')
        _, groups = np.unique(group_idx, return_counts=True)
        G = int(np.amax(group_idx)+1)
        assert p == group_idx.shape[0]
    elif len(groups) > 0:
        G = len(groups)
        group_idx = np.repeat(np.arange(G), groups)
        logger.info(f"{np.sum(groups)=}")
        assert p == np.sum(groups)
    else:
        logger.info("Neither g nor gindex has been defined. One of them is needed for processing data.")
    
    # mpi initalization
    comm = MPI.COMM_WORLD
    worldSize = comm.Get_size()
    rank = comm.Get_rank()
    # size of data blocks
    p_split = int(p/worldSize)
    if p_split * worldSize < p:
        p_split += 1
    if rank==0:
        logger.info(f"Data is split in {worldSize} * {p_split} columns.")

    # open genotype files via lazy loading
    for i in range(len(xfiles)):
        z = zarr.open(xfiles[i], mode='r')
        if i == 0:
            xdata = da.from_zarr(z)
        else:
            xdata = da.append(xdata,z, axis=1)
    xdata = xdata.astype('int8')
    logger.info(f"{xdata=}")
    # add columns of 0 for even split
    if p_split*worldSize-p > 0:
        az = da.zeros((xdata.shape[0], (p_split*worldSize-p)*k), dtype = 'int8')
        if rank == 0:
            logger.info(f"Added {p_split*worldSize-p} columns of zeros to x.")
        xdata = da.concatenate([xdata, az], axis=1)
        group_idx = np.append(group_idx, np.ones((p_split*worldSize-p))*G)
        group_idx = group_idx.astype(int)
        if rank==0:
            logger.info(f"{group_idx.shape=}")
            logger.info(f"{G=}, {group_idx=}")
    ## delete rows where phenotype is na
    if rmid is not None:
        lines = list(np.loadtxt(rmid, delimiter=",").astype('int'))
        lines = [l for l in lines if l < n]
        xdata = np.delete(xdata, lines, axis=0) 
        n -= len(lines)
    ## split data for each process
    X_split = xdata[:,rank*k*p_split:(rank+1)*k*p_split].compute()
    ## open xtx files
    ## split XtX for each process
    for i in range(len(xtxfiles)):
        zxtx = zarr.open(xtxfiles[i], mode='r')
        if i == 0:
            xtxdata = da.from_zarr(zxtx)
        else:
            xtxdata = np.append(xtxdata, zxtx, axis=0)
    xtxdata = xtxdata.astype('float32')
    XtX = xtxdata[rank*k*p_split:(rank+1)*k*p_split].compute()

    if rank==0:
        logger.info(f"Problem has dimensions {n=}, {p=}, {k=}, {G=}.")
    # random generator
    rng = np.random.default_rng()

    # initialize some parameters
    beta = None
    tracker = None
    Z_sum = np.ones(G, dtype='i')*int(p/G)
    Z = np.zeros(G, dtype='i')
    epsilon = np.zeros((n), dtype='float64')
    sigma2 = np.zeros((1), dtype='float64')
    V_inv = np.zeros((G,k,k), dtype='float64')
    pi_ratio = np.ones(G, dtype='float64')

    if rank == 0:
        ## open phenotype file
        epsilon = np.loadtxt(yfile)[:n]
        logger.info(f"{epsilon.shape=}")
       
        # initalize parameters
        init = {
            "beta": np.zeros((p_split*worldSize, k)),
            "V": np.repeat([0.5/G*np.eye(k)], G, axis=0),
            "sigma2": 0.5,
            "pi": np.repeat(np.array([[0.9, 0.1]]), G, axis=0), #[0.5, 0.5]
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
        
        if restart==False:
            beta = init["beta"]
            V = np.array(init["V"])
            sigma2 = np.array(init["sigma2"])
            pi = init["pi"]
            L = init["L"]
        else:
            V = np.loadtxt(resultdir+'/V_1000.txt').reshape(G,k,k)
            sigma2 = np.loadtxt(resultdir+'/sigma2_1000.txt')
            beta = pd.read_csv(resultdir+'/beta_1000.csv.zip', compression='zip').to_numpy()           
            beta = np.concatenate([beta, np.zeros((p_split*worldSize-p, k))], axis=0)
            L = np.loadtxt(resultdir+'/L_1000.txt').reshape(G,k,k)
            Z_sum = np.loadtxt(resultdir+'/Z_1000.txt').astype(np.int32).reshape(G)
            pi = init["pi"]
            for g in range(G):
                pi[g] = rng.dirichlet((groups[g]-Z_sum[g], Z_sum[g]))

        mu = init["mu"]
        D = init["D"]
        tracker = np.zeros(p_split*worldSize)
        for g in range(G):
            V_inv[g] = linalg.inv(V[g])
        logger.info(f"initialize V as {V=}")
        logger.info(f"initialize sigma2 as {sigma2=}")
            
        # generate storage using the Welford package
        w_beta = welford.Welford()
        w_V = welford.Welford()
        w_sigma2 = welford.Welford()
        w_tracker = welford.Welford()
        # storage
        trace_V = np.zeros((iters,G,k,k))
        trace_sigma2 = np.zeros((iters))
        trace_Z = np.zeros((iters, G))
    
    # send data
    # initialize recvbuf on all processes
    tracker_split = np.zeros(p_split)
    beta_split = np.zeros(p_split*k)
    comm.Scatterv([beta, p_split*k, MPI.DOUBLE], beta_split, root=0)
    beta_split = beta_split.reshape(p_split, k)

    ## start iterations
    if rank==0:
        logger.info(f"Running Gibbs with {iters=} and {burnin=}")
    for it in trange(iters, desc="Main loop"):

        comm.Barrier()
        if rank==0:
            #calculate pi ratio
            pi_ratio = pi[:,0]/pi[:,1]

            # sample intercept term
            epsilon += mu
            mu = np.mean(epsilon) if it == 0 else rng.normal(np.mean(epsilon), np.sqrt(sigma2/n))
            epsilon -= mu

            # flatten matrices to vectors for sending
            V_inv = V_inv.flatten()
        
        #send relevant information to all processes
        comm.Bcast([epsilon, MPI.DOUBLE], root=0)
        comm.Bcast([sigma2, MPI.DOUBLE], root=0)
        comm.Bcast([V_inv, MPI.DOUBLE], root=0)
        comm.Bcast([pi_ratio, MPI.DOUBLE], root=0)
        comm.Bcast([Z_sum, MPI.INT], root=0)

        # reshape flattened vectors
        if rank== 0:
            V_inv = V_inv.reshape((G,k,k))

        ## containers for differences in epsilon
        diff = np.zeros((n))
        diff_sum = np.zeros((n))
        ## set number of non-zero markers to 0 before each iteration
        Z = np.zeros(G, dtype='i')
        #keep track of number of processed markers
        counter = 0 

        # loop trough all markers randomly
        rj = np.arange(0, p_split)
        rng.shuffle(rj)
        for j in rj:
            # check if marker is outside of range
            gj = j + p_split*rank
            if gj >= p:
                beta_split[j] = np.zeros((1,k))
                counter += 1
            else:     
                #get group index
                g = group_idx[gj]
                # standardize X
                X = (X_split[:, j*k:k*(j+1)] - np.nanmean(X_split[:, j*k:k*(j+1)], axis=0))/np.nanstd(X_split[:, j*k:k*(j+1)], ddof=1, axis=0)
                # calculate X.T@epsilon
                #xe = np.matmul(X.T, epsilon)
                xe = blas.dgemv(1.0, X, epsilon, trans=1)
                prev_beta = beta_split[j].copy()

                ## sample beta
                ### condition if Z = 0
                temp = V_inv[g] if Z_sum[g]==0 else V_inv[g]*Z_sum[g]
                tracker_split[j], beta_split[j] = sample_mvn(
                    k, 
                    xe, 
                    XtX[j*k:(j+1)*k,:],
                    prev_beta, 
                    sigma2, 
                    temp, #V_inv[g]*Z_sum[g], 
                    pi_ratio[g],
                    rng)

                # calculate difference in epsilon
                #diff += np.matmul(X, (prev_beta - beta_split[j]))
                diff += blas.dgemv(1, X, (prev_beta - beta_split[j]))
                # udpate number of non-zero betas
                Z[g] += tracker_split[j]
                counter += 1
            
            #receive and sum up diff after each process processed two markers
            if counter%itc==0 or counter == p_split:
                comm.Barrier()
                comm.Reduce(diff, diff_sum, MPI.SUM, root=0)
                                    
                if rank==0:
                    epsilon = epsilon + diff_sum
                    diff_sum = np.zeros((n))
                comm.Bcast(epsilon, root=0)
                diff = np.zeros((n))
            
        comm.Barrier()
        # sum up number of non-zero effects
        comm.Reduce(Z, Z_sum, MPI.SUM, root=0)
        # pull together betas and tracker
        comm.Gatherv(sendbuf=beta_split, recvbuf=(beta, p_split*k), root = 0)
        comm.Gatherv(sendbuf=tracker_split, recvbuf=tracker, root = 0) 

        if rank == 0:
            beta = beta.reshape((p_split*worldSize, k))

            # update pi for each group
            for g in range(G):
                if Z_sum[g] == 0:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g]-1, 1))
                elif Z_sum[g] == groups[g]:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g]+1, Z_sum[g]-1))
                else:
                    pi[g] = rng.dirichlet((groups[g]-Z_sum[g], Z_sum[g]))

                #update V
                start_g = 0 if g == 0 else np.sum(groups[:g])
                end_g = np.sum(groups[:g+1])
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

            # store stuff
            trace_V[it] = V
            trace_sigma2[it] = sigma2
            trace_Z[it] = Z_sum
            if (it%500==0) and (it > 0) and (it < burnin):
                dfm = pd.DataFrame(beta[:p], columns=['child', 'mother', 'father', 'poo'])
                dfm.to_csv(resultdir+'/beta_'+str(it)+'.csv.zip', index=False, compression='zip', sep=',')
                np.savetxt(resultdir+'/V_'+str(it)+'.txt', V.reshape(G*k,k))
                np.savetxt(resultdir+'/sigma2_'+str(it)+'.txt', sigma2.reshape(1,1))
                np.savetxt(resultdir+'/Z_'+str(it)+'.txt', trace_Z[it])
                np.savetxt(resultdir+'/prob_'+str(it)+'.txt', tracker[:p])
                np.savetxt(resultdir+'/L_'+str(it)+'.txt', L.reshape(G*k,k))
                np.savetxt(resultdir+'/epsilon'+str(it)+'.txt', epsilon)
                np.savetxt(resultdir+'/trace_Z'+str(it)+'.txt', trace_Z)
                np.savetxt(resultdir+'/trace_sigma2'+str(it)+'.txt', trace_sigma2)
                for g in range(G):
                    np.savetxt(resultdir+'/trace_V'+str(g)+'_'+str(it)+'.txt', trace_V[:,g].diagonal(0,1,2))

            if it >= burnin:
                w_beta.add(beta[:p])
                w_sigma2.add(sigma2)
                w_V.add(V.reshape(G*k,k))
                w_tracker.add(tracker[:p])
                if (it == burnin):
                    dfm = pd.DataFrame(beta[:p], columns=['child', 'mother', 'father', 'poo'])
                    dfm.to_csv(resultdir+'/beta_'+str(it)+'.csv.zip', index=False, compression='zip', sep=',')
                    np.savetxt(resultdir+'/V_'+str(it)+'.txt', V.reshape(G*k,k))
                    np.savetxt(resultdir+'/sigma2_'+str(it)+'.txt', sigma2.reshape(1,1))
                    np.savetxt(resultdir+'/Z_'+str(it)+'.txt', trace_Z[it])
                    np.savetxt(resultdir+'/prob_'+str(it)+'.txt', tracker[:p])
                    np.savetxt(resultdir+'/L_'+str(it)+'.txt', L.reshape(G*k,k))
                    np.savetxt(resultdir+'/epsilon'+str(it)+'.txt', epsilon)
                    np.savetxt(resultdir+'/trace_Z'+str(it)+'.txt', trace_Z)
                    np.savetxt(resultdir+'/trace_sigma2'+str(it)+'.txt', trace_sigma2)
                    for g in range(G):
                        np.savetxt(resultdir+'/trace_V'+str(g)+'_'+str(it)+'.txt', trace_V[:,g].diagonal(0,1,2))
            
            if (it%500==0) and (it > burnin):    
                logger.info(f"{it=}")
                mean_beta = np.array(w_beta.mean)
                var_beta = np.array(w_beta.var_s)
                mean_V = np.array(w_V.mean)
                var_V = np.array(w_V.var_s)
                mean_sigma2 = np.array(w_sigma2.mean)
                var_sigma2 = np.array(w_sigma2.var_s)
                mean_tracker = np.array(w_tracker.mean)
                var_tracker = np.array(w_tracker.var_s)
                logger.info(f"{np.mean(trace_Z[burnin:], axis=0)=}\n")
                logger.info(f"{mean_V=}")
                logger.info(f"{var_V=}")
                logger.info(f"{np.matmul(mean_beta.T, mean_beta)=}\n")
                logger.info(f"{mean_sigma2=}")
                ### save
                dfm = pd.DataFrame(mean_beta, columns=['child', 'mother', 'father', 'poo'])
                dfm.to_csv(resultdir+'/mean_beta_'+str(it)+'.csv.zip', index=False, compression='zip', sep=',')
                dfv = pd.DataFrame(var_beta, columns=['child', 'mother', 'father', 'poo'])
                dfv.to_csv(resultdir+'/var_beta_'+str(it)+'.csv.zip', index=False, compression='zip', sep=',')
                np.savetxt(resultdir+'/mean_V_'+str(it)+'.txt', mean_V)
                np.savetxt(resultdir+'/var_V_'+str(it)+'.txt', var_V)
                np.savetxt(resultdir+'/mean_sigma2_'+str(it)+'.txt', mean_sigma2.reshape(1,1))
                np.savetxt(resultdir+'/var_sigma2_'+str(it)+'.txt', var_sigma2.reshape(1,1))
                np.savetxt(resultdir+'/trace_Z.txt', trace_Z)
                np.savetxt(resultdir+'/mean_prob_'+str(it)+'.txt', mean_tracker)
                np.savetxt(resultdir+'/var_prob_'+str(it)+'.txt', var_tracker)
                np.savetxt(resultdir+'/trace_sigma2.txt', trace_sigma2)
                for g in range(G):
                    np.savetxt(resultdir+'/trace_V'+str(g)+'_'+str(it)+'.txt', trace_V[:, g].diagonal(0,1,2))

    ## iterations finished
    ## print out numbers
    if rank == 0:
        mean_beta = np.array(w_beta.mean)
        var_beta = np.array(w_beta.var_s)
        mean_V = np.array(w_V.mean)
        var_V = np.array(w_V.var_s)
        mean_sigma2 = np.array(w_sigma2.mean)
        var_sigma2 = np.array(w_sigma2.var_s)
        mean_tracker = np.array(w_tracker.mean)
        var_tracker = np.array(w_tracker.var_s)
        logger.info(f"{np.mean(trace_Z[burnin:], axis=0)=}\n")
        logger.info(f"{mean_V=}")
        logger.info(f"{var_V=}")
        logger.info(f"{np.matmul(mean_beta.T, mean_beta)=}\n")
        logger.info(f"{mean_sigma2=}")
        ### save
        dfm = pd.DataFrame(mean_beta, columns=['child', 'mother', 'father', 'poo'])
        dfm.to_csv(resultdir+'/mean_beta.csv.zip', index=False, compression='zip', sep=',')
        dfv = pd.DataFrame(var_beta, columns=['child', 'mother', 'father', 'poo'])
        dfv.to_csv(resultdir+'/var_beta.csv.zip', index=False, compression='zip', sep=',')
        np.savetxt(resultdir+'/mean_V.txt', mean_V)
        np.savetxt(resultdir+'/var_V.txt', var_V)
        np.savetxt(resultdir+'/mean_sigma2.txt', mean_sigma2.reshape(1,1))
        np.savetxt(resultdir+'/var_sigma2.txt', var_sigma2.reshape(1,1))
        np.savetxt(resultdir+'/trace_Z.txt', trace_Z)
        np.savetxt(resultdir+'/mean_prob.txt', mean_tracker)
        np.savetxt(resultdir+'/var_prob.txt', var_tracker)
        np.savetxt(resultdir+'/trace_sigma2.txt', trace_sigma2)
        if G==1:
            np.savetxt(resultdir+'/trace_V.txt', trace_V.reshape(iters,k*k))
        else:
            np.savetxt(resultdir+'/trace_V.txt', trace_V[:, g].diagonal(0,1,2))

        # Plotting sigma2 results
        logger.info("Plotting sigma2 results.")
        t = np.arange(iters)
        figS, axS = plt.subplots()
        axS.plot(t, trace_sigma2)
        axS.set(ylabel='sigma2', xlabel='iterations')
        axS.get_figure().savefig(resultdir+'/trace_sigma2.png')
        plt.close()

        # Plotting V results
        logger.info("Plotting V results.")
        for g in range(G):
            figV, axV = plt.subplots()
            for i in range(k):
                for j in range(k):
                    axV.plot(t, trace_V[:,g,i,j], label=str(i)+str(j))
            axV.set(ylabel='V', xlabel='iterations')
            axV.legend(ncol=4, loc=1)
            axV.get_figure().savefig(resultdir+'/trace_V_'+str(g)+'.png')
            plt.close()

        # Plotting Z results
        logger.info("Plotting Z results.")
        t = np.arange(iters)
        figZ, axZ = plt.subplots()
        axZ.plot(t, trace_Z)
        axZ.set(ylabel='Z', xlabel='iterations')
        axZ.get_figure().savefig(resultdir+'/trace_Z.png')
        plt.close()

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parental Gibbs sampler.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--k', type=int, default=4, help='number of family member incl. POO (2,3 or 4; default=4)')
    parser.add_argument('--g', nargs='+', type=int, help='number of markers in each group')
    parser.add_argument('--gindex', type=str, help='file with group index information; groups need to be in order!')
    parser.add_argument('--iters', type=int, default=10000, help='number of iterations (default = 10000)')
    parser.add_argument('--burnin', type=int, default=1000, help='number of iterations in burnin (default = 1000)')
    parser.add_argument('--itc', type=int, default=2, help='counter for updating epsilon (default=2)')
    parser.add_argument('--x', type=str, nargs='+', help='list of genotype matrix filenames (zarr files)', required = True)
    parser.add_argument('--xtx', type=str, nargs='+', help='list of xtx files (zarr files)', required = True)
    parser.add_argument('--y', type=str, help='phenotype matrix filename (txt file)', required = True)
    parser.add_argument('--resultdir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--rmid', type=str, help='list of ids to delete (default is None)')
    parser.add_argument('--restart', type=bool, default=False, help='restart with burnin values (default=False)')
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
        gindex = args.gindex, # group indices
        itc = args.itc, # counter for updating epsilon (after number of processes times itc markers)
        xfiles = args.x, # genotype file
        xtxfiles = args.xtx, # genotype file
        yfile = args.y, # phenotype file
        resultdir = args.resultdir, # path to results directory
        rmid = args.rmid, # rows to remove from genotype
        restart = args.restart, #boolean for restart
        ) 
    logger.info("Done.")
