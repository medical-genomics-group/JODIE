#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru tqdm zarr
```
python preprocessing_vcf_MC.py --n 5000 --p 20000 --ntrios 5000 --na 0 --maf 0.2 --dir outdir
````
--n number of individuals (required)
--p number of markers (required)
--ntrios number of trios (required)
--na number of not genotyped mothers, fathers are calculated as n - na - ntrios; set to 0 if only trios should be generated; missing parent will be inferred
--maf minor allele frequency used for all markers; default = 0.2
--dir output directory
"""
import sys
import argparse
import numpy as np
from loguru import logger
import zarr
from tqdm import trange

def gen_data(n, p, prob, rng):
    trans = 0.5
    ## Simulate parents alleles
    # mother
    xm_a1 = rng.binomial(1, prob, size=(n, p))
    xm_a2 = rng.binomial(1, prob, size=(n, p))
    # father
    xf_a1 = rng.binomial(1, prob, size=(n, p))
    xf_a2 = rng.binomial(1, prob, size=(n, p))
    ## Indicator of transmission of alleles from parents to offspring
    r_m = rng.binomial(1, trans, size=(n, p))
    r_f = rng.binomial(1, trans, size=(n, p))
    ## Simulated alleles of child
    xc_a1 = r_m * xm_a2 + (1 - r_m) * xm_a1 #allele from mother
    xc_a2 = r_f * xf_a2 + (1 - r_f) * xf_a1 #allele from father
    X = np.zeros((p,3*n,2), dtype='int8')
    X[:,:n,0] = xc_a1.T
    X[:,:n,1] = xc_a2.T
    X[:,n:2*n,0] = xm_a1.T
    X[:,n:2*n,1] = xm_a2.T
    X[:,2*n:,0] = xf_a1.T
    X[:,2*n:,1] = xf_a2.T
    return X

def main(n, p, na, n_trios, k, prob, dir):

    logger.info(f"{zarr.__version__=}")
    missing = True if n_trios!=n else False

    ## possible genetic combinations of child/mother/father
    combinations = np.array([[0,0], [0,1], [1,0], [1,2], [2,1], [2,2], [1,1]])
    # trios for imputation of mother; father available
    trios_mother = np.array([[0,0,0], [0,0,1], 
        [1,1,0], [1,0,2], 
        [2,1,1], [2,1,2],
        [1,0,1], [1,2,1]]
    )
    # trios for imputation of father; mother available
    trios_father = np.array([[0,0,0], [0,1,0], 
        [1,0,1], [1,2,0], 
        [2,1,1], [2,2,1],
        [1,1,0], [1,1,2]]
    )
    # combinations of NA data
    na_mother = np.array([[0,9,0], [0,9,1], 
        [1,9,0], [1,9,2], 
        [2,9,1], [2,9,2],
        [1,9,1], [1,9,1]]
    )
    na_father = np.array([[0,0,9], [0,1,9], 
        [1,0,9], [1,2,9], 
        [2,1,9], [2,2,9],
        [1,1,9], [1,1,9]]
    )
    # shift
    shift = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    # initialize numpy random
    rng = np.random.default_rng()

    # generate data
    gt = gen_data(n, p, prob, rng)
    id = np.arange(3*n).reshape(n, 3, order='F')
    #logger.info(f"{id=}")
    xa = gt[:,:,0] + gt[:,:,1]
    #logger.info(f"{xa=}")
    n_duos_m=na
    n_duos_f=n-n_trios-na
    logger.info(f"{n=}, {p=}, {n_trios=}, {n_duos_m=}, {n_duos_f=}")

    # add line with nans to be able to indicate missing data
    a = np.ones((p,1)) * 9
    xa = np.concatenate((xa,a), axis=1)
    x = np.zeros((p*k,n), dtype='int8')
    # fill x with child, mother, father genotype
    for i in range(k-1):
        x[i::k] = xa[:,id[:,i]]
    # add parent of origin information
    xpoo = np.zeros((p, n), dtype='int8')
    ## 1 if minor allele is coming from the mother
    ## return two arrays with indices for dim1 and dim2
    wm = np.where((np.equal(gt[:,id[:,0],0],1) & np.equal(gt[:,id[:,0],1],0)))
    xpoo[wm[0], wm[1]] = 1
    ## -1 if minor allele is coming from the father
    wf = np.where((np.equal(gt[:,id[:,0],0],0) & np.equal(gt[:,id[:,0],1],1)))
    xpoo[wf[0], wf[1]] = -1
    x[(k-1)::k] = xpoo
    #logger.info(f"{x=}")
    if zarr.__version__.startswith('3'):
        logger.info(f"zarr 3")
        z = zarr.create_array(store=f'{dir}/genotype.zarr', shape=x.T.shape, chunks=(n, 1000), dtype='int8')
        z[:] = x.T
    elif zarr.__version__.startswith('2'):
        logger.info(f"zarr 2")
        z = zarr.array(x.T, chunks=(None,1000)) ##zarr2
        zarr.save('genotype.zarr', z) ## zarr2
    logger.info(f"{z.info=}")

    # simulate missing data
    if missing:
        x[1::k, n_trios:n_trios+na] = 9
        x[2::k, n_trios+na:n] = 9
        #logger.info(f"{x.T=}")

        logger.info(f"Imputing duos")
        # loop through markers
        for j in trange(p, desc="Main loop"):

            rj = j*k
            # get counts for 0, 1, 2 and 9 for each marker and trio
            vc, cc = np.unique(x[rj], return_counts=True)
            vm, cm = np.unique(x[rj+1], return_counts=True)
            vf, cf = np.unique(x[rj+2], return_counts=True)
            # calculate number of mothers and fathers
            nm = n if len(np.argwhere(vm==9)) == 0 else n - cm[(np.argwhere(vm==9)).item()]     # number of mothers in full dataset
            nf = n if len(np.argwhere(vf==9)) == 0 else n - cf[(np.argwhere(vf==9)).item()]     # number of fathers in full dataset
            # get occurences of 0,1,2 in trios
            vm_trios, cm_trios_ = np.unique(x[rj+1,:n_trios], return_counts=True)
            vf_trios, cf_trios_ = np.unique(x[rj+2,:n_trios], return_counts=True)

            # containers for prior probabilities
            ppA = np.zeros(3)
            # containers for counts in trios
            cm_trios = np.zeros(3)
            cf_trios = np.zeros(3)
            for i in range(3):
                # prior counts A calculated from children, mothers and fathers
                ppA[i] = 0 if len(np.argwhere(vm==i)) == 0 else cm[(np.argwhere(vm==i)).item()]
                ppA[i] = ppA[i] if len(np.argwhere(vf==i)) == 0 else ppA[i]+cf[(np.argwhere(vf==i)).item()]
                ppA[i] = ppA[i] if len(np.argwhere(vc==i)) == 0 else ppA[i]+cc[(np.argwhere(vc==i)).item()]

                # check that all values are there, otherwise set count to 0
                # counts of mother being 0,1,2 in trios
                cm_trios[i] = 0 if len(np.argwhere(vm_trios==i)) == 0 else cm_trios_[(np.argwhere(vm_trios==i)).item()]
                # counts of father being 0,1,2 in trios
                cf_trios[i] = 0 if len(np.argwhere(vf_trios==i)) == 0 else cf_trios_[(np.argwhere(vf_trios==i)).item()]
            # divide prior counts by total number of individuals to get prior probability A
            ppA /= (nm+nf+n)
                
            # loop through different combinations and calculate their probabilities
            for i in range(len(combinations)):

                # determine how many NA values need to be replaced
                id_na_m = np.argwhere((na_mother[i]==x[rj:rj+3].T).all(axis=1)) # number of NA mothers
                id_na_f = np.argwhere((na_father[i]==x[rj:rj+3].T).all(axis=1)) # number of NA fathers
                nm_na = len(id_na_m)
                nf_na = len(id_na_f)
                #logger.info(f"{nm_na=}, {nf_na=}")
                if (nm_na==0 and nf_na==0):
                    #logger.info(f"Skipped combination {combinations[i]}")
                    continue 
                    
                # figure out index of all child/mother and child/father combinations
                duos_id_m = np.argwhere((combinations[i]==x[rj:rj+3:2].T).all(axis=1))  # for mother NA
                duos_id_f = np.argwhere((combinations[i]==x[rj:rj+2].T).all(axis=1))    # for father NA
                    
                ## homozygotes need to be split according to parent of origin information
                if (combinations[i]==np.array([1,1])).all() and (len(duos_id_m) > 0 or len(duos_id_f) > 0):
                    # if minor allele from mother == 1, poo = x[rj+3]== 1 
                    # if minor allele from father == 1, poo = x[rj+3] == -1
                    duos_id_m0 = np.argwhere((np.array([1,-1])==x[rj+2:rj+4].T).all(axis=1))     # duos where Xf = 1, POO = -1 (Xm=0), mother NA
                    duos_id_m1 = np.argwhere((np.array([1,1])==x[rj+2:rj+4].T).all(axis=1))      # duos where Xf = 1, POO = +1 (Xm=2), mother NA
                    duos_id_f0 = np.argwhere((np.array([1,1])==x[rj+1:rj+4:2].T).all(axis=1))    # duos where Xm = 1, POO = +1 (Xf=0), father NA
                    duos_id_f1 = np.argwhere((np.array([1,-1])==x[rj+1:rj+4:2].T).all(axis=1))   # duos where Xm = 1, POO = -1 (Xf=2), father NA
                        
                    # calulate prior probabilities B
                    # from number of occurences divided by all number of available duos (=number of available mothers or fathers)
                    ppB_m0 = len(duos_id_m0)/nf # occurences of child+father / total number of child+father duos; split by POO
                    ppB_m1 = len(duos_id_m1)/nf
                    ppB_f0 = len(duos_id_f0)/nm # occurences of child+mother / total number of child+mother duos; split by POO
                    ppB_f1 = len(duos_id_f1)/nm

                    # calculate likelihood from trios; split by POO
                    Lm0 = 0 if cm_trios[trios_mother[i,1]]==0 else len(np.argwhere((trios_mother[i]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cm_trios[trios_mother[i,1]]
                    Lm1 = 0 if cm_trios[trios_mother[i+1,1]]==0 else len(np.argwhere((trios_mother[i+1]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cm_trios[trios_mother[i+1,1]]
                    Lf0 = 0 if cf_trios[trios_father[i,2]]==0 else len(np.argwhere((trios_father[i]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cf_trios[trios_father[i,2]]
                    Lf1 = 0 if cf_trios[trios_father[i+1,2]]==0 else len(np.argwhere((trios_father[i+1]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cf_trios[trios_father[i+1,2]]


                    # determine how many NA values need to be replaced
                    id_na_m0 = np.argwhere((np.array([1, 9, 1, -1])==x[rj:rj+4].T).all(axis=1)) # 1, 9, 1, -1 (Xm = 0)
                    id_na_m1 = np.argwhere((np.array([1, 9, 1, +1])==x[rj:rj+4].T).all(axis=1)) # 1, 9, 1, +1 (Xm = 2) 
                    id_na_f0 = np.argwhere((np.array([1, 1, 9, +1])==x[rj:rj+4].T).all(axis=1)) # 1, 1, 9, +1 (Xf = 0)
                    id_na_f1 = np.argwhere((np.array([1, 1, 9, -1])==x[rj:rj+4].T).all(axis=1)) # 1, 1, 9, -1 (Xf = 2) 
                    nm_na0 = len(id_na_m0)
                    nm_na1 = len(id_na_m1)
                    nf_na0 = len(id_na_f0)
                    nf_na1 = len(id_na_f1)
                    # draw from binomial distribution according to posterior probability
                    # add shift for those values where the possiblities are 1 and 2
                    if nm_na0 > 0:
                        pm0 = ppA[trios_mother[i,1]]*Lm0/ppB_m0
                        if pm0 > 1:
                            #logger.info(f"reset pm0 to 1: {j=}, {ppA=}, {ppB_m0=}, {Lm0=}, {pm0=}")
                            pm0 = 1
                        mother = rng.binomial(1, pm0, size= nm_na0)
                        x[rj+1, id_na_m0] = mother.reshape(nm_na0,1)+shift[i]
                    if nm_na1 > 0:
                        pm1 = ppA[trios_mother[i+1,1]]*Lm1/ppB_m1
                        # pm1 gives probability of getting Xm = 2
                        # 1-pm1 gives probabiltiy of getting Xm = 1
                        # use 1-pm1 to make it consistent the other combinations
                        if pm1 > 1:
                            #logger.info(f"reset pm1 to 1: {j=}, {ppA=}, {ppB_m1=}, {Lm1=}, {pm1=}")
                            pm1 = 1
                        mother = rng.binomial(1, (1-pm1), size= nm_na1)
                        x[rj+1, id_na_m1] = mother.reshape(nm_na1,1)+shift[i+1]
                    if nf_na0 > 0:
                        pf0 = ppA[trios_father[i,2]]*Lf0/ppB_f0
                        if pf0 > 1:
                            #logger.info(f"reset pf0 to 1: {j=}, {ppA=}, {ppB_f0=}, {Lf0=}, {pf0=}")
                            pf0 = 1
                        father = rng.binomial(1, pf0, size= nf_na0)+shift[i]
                        x[rj+2, id_na_f0] = father.reshape(nf_na0,1)
                    if nf_na1 > 0:
                        pf1 = ppA[trios_father[i+1,2]]*Lf1/ppB_f1
                        #logger.info(f"{j=}, {af[j]=}, {ppA=}, {ppB_f1=}, {Lf1=}, {pf1=}")
                        # pf1 gives probability of getting Xf = 2
                        # 1-pf1 gives probabiltiy of getting Xf = 1
                        # use 1-pf1 to make it consistent the other combinations
                        if pf1 > 1:
                            #logger.info(f"reset pf1 to 1: {j=}, {ppA=}, {ppB_f1=}, {Lf1=}, {pf1=}")
                            pf1 = 1
                        father = rng.binomial(1, (1-pf1), size= nf_na1)+shift[i+1]
                        x[rj+2, id_na_f1] = father.reshape(nf_na1,1)

                else:
                    # calculate prior probability B
                    # from number of occurences divided by all number of available duos (=number of available mothers or fathers)
                    ppB_m = len(duos_id_m)/nf    # occurences of child+father / total number of child+father duos
                    ppB_f = len(duos_id_f)/nm    # occurences of child+mother / total number of child+mother duos
                    #logger.info(f"{combinations[i]=}, {ppB_m=}, {ppB_f=}")

                    # calculate likelihood from trios
                    Lm = 0 if cm_trios[trios_mother[i,1]]==0 else len(np.argwhere((trios_mother[i]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cm_trios[trios_mother[i,1]]
                    Lf = 0 if cf_trios[trios_father[i,2]]==0 else len(np.argwhere((trios_father[i]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cf_trios[trios_father[i,2]]
                    #logger.info(f"{Lm=}, {Lf=}")

                    # for each combination there are two possibilities
                    # calculate posterior probability for one possibility, then the other one has prob. 1-p
                    # posterior = priorA * likelihood / priorB
                    pm = ppA[trios_mother[i,1]]*Lm/ppB_m
                    pf = ppA[trios_father[i,2]]*Lf/ppB_f
                    #logger.info(f"{j=}, {pm=}, {pf=}")

                    # draw from binomial distribution according to posterior probability
                    # add shift for those values where the possiblities are 1 and 2
                    if nm_na > 0:
                        if pm > 1:
                            #logger.info(f"reset pm to 1: {j=}, {ppA=}, {ppB_m=}, {Lm=}, {pm=}")
                            pm = 1
                        mother = rng.binomial(1, pm, size= nm_na)
                        x[rj+1, id_na_m] = mother.reshape(nm_na,1)+shift[i]
                    if nf_na > 0:
                        if pf > 1:
                            #logger.info(f"reset pf to 1: {j=}, {ppA=}, {ppB_f=}, {Lf=}, {pf=}")
                            pf = 1
                        father = rng.binomial(1, pf, size= nf_na)+shift[i]
                        x[rj+2, id_na_f] = father.reshape(nf_na,1)
        
        if zarr.__version__.startswith('3'):
            logger.info(f"zarr 3")
            z = zarr.create_array(store=f'{dir}/genotype_imputed.zarr', shape=x.T.shape, chunks=(n, 1000), dtype='int8')
            z[:] = x.T
        elif zarr.__version__.startswith('2'):
            logger.info(f"zarr 2")               
            z = zarr.array(x.T, chunks=(None,1000)) ##zarr2
            zarr.save('genotype_imputed.zarr', z) ##zarr2
        logger.info(f"{z.info=}")

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generating MC.')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)
    parser.add_argument('--ntrios', type=int, help='number of trios', required=True)
    parser.add_argument('--na', type=int, help='number of not genotyped mothers, fathers are calculated as n - na - ntrios', required=True)
    parser.add_argument('--maf', type=float, default = 0.2, help='minor allele frequency (default=0.2)')
    parser.add_argument('--dir', type=str, help='path to output directory', required=True)
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
        n = args.n,
        p = args.p,
        n_trios = args.ntrios,
        na = args.na, 
        k = 4, # number of genetic components
        prob = args.maf,
        dir = args.dir,
        ) 
    logger.info("Done.")
