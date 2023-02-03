# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru scipy
```
python gen_data.py --n 20000 --p 60000 --p0 56000 --k --dir MC
"""

import sys
import argparse
import numpy as np
import scipy.stats as stats
from loguru import logger


def gen_data(n, p, rng):
    prob = 0.5
    ## Simulate parents alleles
    # mother
    xm_a1 = rng.binomial(1, prob, size=(n, p))
    xm_a2 = rng.binomial(1, prob, size=(n, p))
    Xm = xm_a1 + xm_a2
    # father
    xf_a1 = rng.binomial(1, prob, size=(n, p))
    xf_a2 = rng.binomial(1, prob, size=(n, p))
    Xf = xf_a1 + xf_a2
    ## Indicator of transmission of alleles from parents to offspring
    r_m = rng.binomial(1, prob, size=(n, p))
    r_f = rng.binomial(1, prob, size=(n, p))
    ## Simulated alleles of child
    xc_a1 = r_m * xm_a2 + (1 - r_m) * xm_a1
    xc_a2 = r_f * xf_a2 + (1 - r_f) * xf_a1
    Xc = xc_a1 + xc_a2
    return Xc, Xm, Xf


def main(n, p, k, p0, dir):

    logger.info(f"Problem has dimensions {n=}, {p=} with {p0=} effects set to 0.")

    contrast = 0
    # random generator
    rng = np.random.default_rng()
    # generate genotpye matrices
    Xc, Xm, Xf = gen_data(n, p, rng)
    # Xpo = np.zeros((n,p))
    # Xpo = np.where((np.equal(Xc,1)) & (np.equal(Xm,1)), (Xc-Xf), Xpo)
    # Xpo = np.where((np.equal(Xc,1)) & (np.equal(Xf,1)), (Xm-Xc), Xpo)
    # Xpo = np.where((np.equal(Xc,1)) & (np.equal(Xm,2)) & (np.equal(Xf,0)), 1, Xpo)
    # Xpo = np.where((np.equal(Xc,1)) & (np.equal(Xm,0)) & (np.equal(Xf,2)), -1, Xpo)
    # Xpo = np.where((np.equal(Xc,1)) & (np.equal(Xm,1)) & (np.equal(Xf,1)), 1, Xpo)

    # reorder Xc, Xm, Xf to X
    X = np.zeros((n, k*p))
    X[:,0::k]=Xc
    X[:,1::k]=Xm
    X[:,2::k]=Xf
    #X[:,3::k]=Xpo
    #logger.info(f"{Xc=}")
    #logger.info(f"{Xm=}")
    #logger.info(f"{Xf=}")
    xmean = np.mean(X, axis=0)
    xstd = np.std(X, axis=0)

    # contrasts
    # direct effects
    Cc = np.where((np.equal(Xm,1)) & (np.equal(Xf,1)), (Xc-Xm), 0)
    #logger.info(f"{Cc=}")
    # maternal
    Cm = np.where((np.equal(Xc,1)) & (np.equal(Xf,1)), (Xm-Xc), 0)
    #logger.info(f"{Cm=}")
    # paternal
    Cf = np.where((np.equal(Xc,1)) & (np.equal(Xm,1)), (Xf-Xc), 0)
    #logger.info(f"{Cf=}")
    # parent of origin
    #Cpo = np.where((np.equal(Xc,1)) & (np.equal(Xm,1)) & (np.equal(Xf,1)), 1, 0)
    #logger.info(f"{Cpo=}")

    # get indices for +-1 for contrasted matrices 
    ## save list of indices for each marker (marker is given by row in file)
    id_c1 = []
    id_c2 = []
    id_m1 = []
    id_m2 = []
    id_f1 = []
    id_f2 = []
    #id_po1 = []
    #id_po2 = []
    if contrast:
        logger.info("Save contrasted indices.")
        for i in range(p):
            lc1 = list(np.where(Cc[:,i]==1)[0])
            id_c1.append(lc1)
            lc2 = list(np.where(Cc[:,i]==-1)[0])
            id_c2.append(lc2)
            lm1 = list(np.where(Cm[:,i]==1)[0])
            id_m1.append(lm1)
            lm2 = list(np.where(Cm[:,i]==-1)[0])
            id_m2.append(lm2)
            lf1 = list(np.where(Cf[:,i]==1)[0])
            id_f1.append(lf1)
            lf2 = list(np.where(Cf[:,i]==-1)[0])
            id_f2.append(lf2)
            # lpo1 = list(np.where(Cpo[:,i]==1)[0])
            # id_po1.append(lpo1)
            # lpo2 = list(np.where(Cpo[:,i]==-1)[0])
            # id_po2.append(lpo2)
    else:
        logger.info("Save not contrasted indices.")
        for i in range(p):
            lc1 = list(np.where(Xc[:,i]==1)[0])
            id_c1.append(lc1)
            lc2 = list(np.where(Xc[:,i]==2)[0])
            id_c2.append(lc2)
            lm1 = list(np.where(Xm[:,i]==1)[0])
            id_m1.append(lm1)
            lm2 = list(np.where(Xm[:,i]==2)[0])
            id_m2.append(lm2)
            lf1 = list(np.where(Xf[:,i]==1)[0])
            id_f1.append(lf1)
            lf2 = list(np.where(Xf[:,i]==2)[0])
            id_f2.append(lf2)
    
    #child
    fc1 = open(dir+'/id1.txt', 'w')
    fc1.write(str(id_c1))
    fc1.close()
    fc2 = open(dir+'/id2.txt', 'w')
    fc2.write(str(id_c2))
    fc2.close()
    #mother
    fm1 = open(dir+'/id3.txt', 'w')
    fm1.write(str(id_m1))
    fm1.close()
    fm2 = open(dir+'/id4.txt', 'w')
    fm2.write(str(id_m2))
    fm2.close()
    #father
    ff1 = open(dir+'/id5.txt', 'w')
    ff1.write(str(id_f1))
    ff1.close()
    ff2 = open(dir+'/id6.txt', 'w')
    ff2.write(str(id_f2))
    ff2.close()
    #poo
    # fpo1 = open(dir+'/idpo_1.txt', 'w')
    # fpo1.write(str(id_po1))
    # fpo1.close()
    # fpo2 = open(dir+'/idpo_2.txt', 'w')
    # fpo2.write(str(id_po2))
    # fpo2.close()

    C = np.zeros((n, k*p))#, dtype='int32')
    C[:,0::k]=Cc
    C[:,1::k]=Cm
    C[:,2::k]=Cf
    #C[:,3::k]=Cpo
    #Cnorm = stats.zscore(C, axis=0, ddof=1)
    #col_mean = np.mean(C, axis=0)
    #col_sigma = np.std(C, axis=0)
    #logger.info(f"{col_mean=}")
    #logger.info(f"{col_sigma=}")
    #logger.info(f"{np.mean(Cnorm, axis=0)=}")
    #logger.info(f"{np.std(Cnorm, axis=0)=}")
    # logger.info(f"{np.matmul(Cnorm.T, Cnorm)=}")
    #logger.info(f"{Cnorm=}")

    #Xnorm = stats.zscore(C, axis=0, ddof=1)
    Xnorm = stats.zscore(X, axis=0, ddof=1)
    # logger.info(f"{Xnorm=}")
    # #logger.info(f"{np.mean(Xnorm, axis=0)=}")
    # #logger.info(f"{np.std(Xnorm, axis=0)=}")
    # cmean = np.mean(Cnorm, axis=0)
    # cstd = np.std(Cnorm, axis=0)
    # mean = xmean - cmean
    # std = cstd/xstd
    # #logger.info(f"{Cnorm*std=}")
    # t = Cnorm*std
    # logger.info(f"{mean=}")
    # logger.info(f"{std=}")
    # for i in range(p):
    #     logger.info(f"{np.unique(Cnorm[:,i])=}")
    #     logger.info(f"{np.unique(Xnorm[:,i])=}")

    # logger.info(f"{np.matmul(Xnorm.T, Xnorm)/np.matmul(Cnorm.T, Cnorm)=}")

    ### generate beta
    V1 = np.array([[0.15, -0.5*np.sqrt(0.15)*np.sqrt(0.25), 0],[-0.5*np.sqrt(0.15)*np.sqrt(0.25), 0.25, 0], [0., 0., 0.1]])
    V2 = np.array([[0.1, 0.5*np.sqrt(0.1)*np.sqrt(0.15), 0],[0.5*np.sqrt(0.1)*np.sqrt(0.15), 0.15, 0.5*np.sqrt(0.15)*np.sqrt(0.2)], [0, 0.5*np.sqrt(0.15)*np.sqrt(0.2), 0.2]])
    # number of non-zero effects for each group
    p1 = int((p-p0)/2)
    b1 = rng.multivariate_normal(np.zeros(k), V1/p1, p1)
    b2 = rng.multivariate_normal(np.zeros(k), V2/p1, p1)
    b0 = np.zeros((int(p0/2), k))
    true_beta1 = np.concatenate((b1, b0))
    rng.shuffle(true_beta1)
    true_V1 = np.matmul(true_beta1.T, true_beta1)
    true_beta2 = np.concatenate((b2, b0))
    rng.shuffle(true_beta2)
    true_V2 = np.matmul(true_beta2.T, true_beta2)
    beta = np.concatenate([true_beta1, true_beta2])
    true_V = np.concatenate([true_V1, true_V2])
    logger.info(f"{true_V=}")
    logger.info(f"{np.cov(beta, rowvar=False)=}")
    logger.info(f"{np.matmul(beta.flatten().T, beta.flatten())=}")
    logger.info(f"{np.var(Xnorm@beta.flatten())=}")
    # generate epsilon
    std = np.sqrt(1-np.var(Xnorm@beta.flatten()))
    logger.info(f"{std=}")
    epsilon = np.random.normal(0, std, n)
    true_sigma2 = np.var(epsilon)
    logger.info(f"{true_sigma2=}")
    logger.info(f"{np.std(epsilon)=}")
    # generate Y
    Y = np.matmul(Xnorm, beta.flatten()) + epsilon
    Ynorm = stats.zscore(Y, axis=0, ddof=1)
 
    # save true values
    np.savetxt(dir+'/true_V.txt', true_V)
    np.savetxt(dir+'/true_sigma2.txt', true_sigma2.reshape(1,1))
    np.savetxt(dir+'/true_betas.txt', beta)
    np.savetxt(dir+'/true_epsilon.txt', epsilon)
    np.savetxt(dir+'/phenotype.txt', Ynorm)
    np.savez_compressed(dir+'/genotype.npz', Xnorm) 
    # save sigma, mu 
    # np.savetxt(dir+'/col_meanc.txt', col_mean)
    # np.savetxt(dir+'/col_sigmac.txt', col_sigma)
    np.savetxt(dir+'/col_meanx.txt', xmean)
    np.savetxt(dir+'/col_sigmax.txt', xstd)


#########################
if __name__ == "__main__":
    # input arguments
    parser = argparse.ArgumentParser(description='Data simulation.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--k', type=int, default=4, help='number of family members + POO (can  be 2, 3, 4; default = 4)')
    parser.add_argument('--p0', type=int, help='number of markers set to 0', required = True)
    parser.add_argument('--dir', type=str, help='path to directory where the results are stored', required = True)
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
        p = args.p,  # number of markers
        k = args.k, # number of family members
        p0 = args.p0, # number of markers set to 0
        dir = args.dir # path to results directory
    ) 
    logger.info("Done.")