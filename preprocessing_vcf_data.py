#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Install dependencies:
```
pip install numpy loguru scikit-allel zarr pandas tqdm
```
python preprocessing_vcf_data.py 
--index_trios Trios.ped
--index_duos Duos.ped
--inputfiles path-to-file/chr1.vcf.gz path-to-file/chr2.vcf.gz path-to-file/chr3.vcf.gz
--dir path-to-output-directory/
--only_trios True
````
--index_trios tab delimited file with id information for trios (required)
--index_duos tab delimited file with id information for duos, incl. nan values for the missing parent
--dir path to output directory where zarr files are stored (required)
--only_trios process only trios, by default false
"""
import sys
import argparse
import numpy as np
from loguru import logger
import allel
import zarr
import pandas as pd
from tqdm import trange

def main(inputfiles, index_trios, index_duos, only_trios, dir, k):

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

    ## if no index file for duos is provided, set flag only_trios to true
    if index_duos is None:
        only_trios=True
        logger.info("No index file for duos was provided. Only processing data for trios.")

    # read in indices
    if only_trios:
        idfile = (pd.read_csv(index_trios, delimiter="\t", header = None)).values
        n = idfile.shape[0]
        n_trios = n
    else: 
        idfile_trios = pd.read_csv(index_trios, delimiter="\t", header = None)
        idfile_duos = pd.read_csv(index_duos, delimiter="\t", header = None, keep_default_na=False)
        #replace NA values with NAN to avoid confusion with individuals' id
        idfile_duos[idfile_duos=='NA']='NAN'
        logger.info(f"{idfile_trios=}")
        logger.info(f"{idfile_duos=}")
        # get number of individuals and markers
        n_trios = len(idfile_trios)
        n = n_trios + len(idfile_duos)
        ## cocatenate idfiles and remove headers
        idfile = np.concatenate([(idfile_trios.values), (idfile_duos.values)], axis=0) 
    logger.info(f"Processing {n=} individuals wiht {n_trios=} trios.")
    logger.info(f"{idfile=}")
    ### container for indices
    id = -np.ones((n,k-1), dtype='int')

    ##loop over multiple input files
    for l in inputfiles:

        logger.info(f"{l=}")
        #read in vcf file
        # 1st dim  = variants, 2nd dim = samples
        callset = allel.read_vcf(l, fields=['calldata/GT', 'samples', 'variants/AF'])
        logger.info(f"{callset['samples']=}")
        af = callset['variants/AF'][:,0]
        #logger.info(f"{af=}")
        ## get indices only once after first file has loaded
        if l == inputfiles[0]:
            ## add a NAN value to sample ids so that NA have last index
            if only_trios:
                samples = callset['samples']
            else:
                samples = np.concatenate([callset['samples'], np.array(['NAN'])])
            sorter = np.argsort(samples)
            logger.info(f"{samples.shape=}")
            logger.info(f"{sorter=}")
            id = sorter[np.searchsorted(samples, idfile, sorter=sorter)]
        logger.info(f"{id=}")
        ## get genotype data as genotype array (2 alleles)
        gt = allel.GenotypeArray(callset['calldata/GT'])
        logger.info(f"{gt=}")
        p = gt.n_variants
        logger.info(f"{n=}, {p=}, {n_trios=}")

        ## combine 2 alleles to 0,1,2
        xa = gt[:,:,0] + gt[:,:,1]
        logger.info(f"{np.unique(xa)=}")
        logger.info(f"{xa=}")
        ## replace NAN indexed with -1 or -2 with 9
        xa = np.where(np.equal(xa, -1), 9, xa)
        xa = np.where(np.equal(xa, -2), 9, xa)
        # add line with nans to be able to indicate missing data
        a = np.ones((p,1)) * 9
        xa = np.concatenate((xa,a), axis=1)
        x = np.zeros((p*k,n), dtype='int8')
        # fill x with child, mother, father genotype
        for i in range(k-1):
            x[i::k] = xa[:,id[:,i]]
        # add parent of origin information
        # add parent of origin information
        xpoo = np.zeros((p, n), dtype='int8')
        ## 1 if minor allele is coming from the mother
        ## return two arrays with indices for dim1 and dim2
        wm = np.where((np.equal(gt[:,id[:,0],0],1) & np.equal(gt[:,id[:,0],1],0)))
        #logger.info(f"{wm=}")
        xpoo[wm[0], wm[1]] = 1
        #logger.info(f"{np.where(xpoo==1)=}")
        ## -1 if minor allele is coming from the father
        wf = np.where((np.equal(gt[:,id[:,0],0],0) & np.equal(gt[:,id[:,0],1],1)))
        xpoo[wf[0], wf[1]] = -1
        #logger.info(f"{wf=}")
        #logger.info(f"{np.where(xpoo==-1)=}")
        x[(k-1)::k] = xpoo
        logger.info(f"{x=}")

        ## remove markers depending on trio data only
        sd = np.nanstd(x[:, 0:n_trios], axis=1)
        logger.info(f"{np.unique(sd)=}")
        did = []
        for i in range(0,k):
            did = np.append(did, np.array(np.where(sd[i::k]==0)).reshape(-1))
            did = np.append(did, np.array(np.where(np.isnan(sd[i::k]))).reshape(-1))
        did = np.unique(did)
        logger.info(f"{did=}")
        if len(did) > 0:
            logger.info(f"{x.shape=}")
            lid = []
            for i in range(len(did)):
                lid = np.append(lid, np.arange(did[i]*k, k*(did[i]+1)))
            lid = lid.astype(int)
            logger.info(f"{lid=}")
            x = np.delete(x, lid, axis=0)
            logger.info(f"{x.shape=}")
        logger.info(f"{np.unique(x)=}")
       
        ## impute missing data for duos
        ## loop through markers
        if only_trios is False:
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
                #logger.info(f"{j=}, {af[j]=}, {ppA=}")
                
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
                        #duos_id_m0 = duos_id_m[np.argwhere((x[rj+3,duos_id_m]==-1).all(axis=1))]    # 1 came from father (Xm=0)
                        #duos_id_m1 = duos_id_m[np.argwhere((x[rj+3,duos_id_m]==1).all(axis=1))]     # 1 came from mother (Xm=2)
                        #duos_id_f0 = duos_id_f[np.argwhere((x[rj+3,duos_id_f]==1).all(axis=1))]     # 1 came from mother (Xf=0)
                        #duos_id_f1 = duos_id_f[np.argwhere((x[rj+3,duos_id_f]==-1).all(axis=1))]    # 1 came from father (Xf=2)
                        
                        # calulate prior probabilities B
                        # from number of occurences divided by all number of available duos (=number of available mothers or fathers)
                        ppB_m0 = len(duos_id_m0)/nf # occurences of child+father / total number of child+father duos; split by POO
                        ppB_m1 = len(duos_id_m1)/nf
                        ppB_f0 = len(duos_id_f0)/nm # occurences of child+mother / total number of child+mother duos; split by POO
                        ppB_f1 = len(duos_id_f1)/nm
                        #logger.info(f"{combinations[i]=}, {ppB_m0=}, {ppB_m1=}, {ppB_f0=}, {ppB_f1=}")
                        logger.info(f"{j=}, {len(duos_id_m)=}, {len(duos_id_m0)=}, {len(duos_id_m1)=}")
                        logger.info(f"{j=}, {len(duos_id_f)=}, {len(duos_id_f0)=}, {len(duos_id_f1)=}")
                        logger.info(f"{np.unique(x[rj,duos_id_m1], return_counts=True)=}, {np.unique(x[rj,duos_id_m0], return_counts=True)=}")
                        logger.info(f"{np.unique(x[rj,duos_id_f1], return_counts=True)=}, {np.unique(x[rj,duos_id_f0], return_counts=True)=}")

                        # calculate likelihood from trios; split by POO
                        Lm0 = 0 if cm_trios[trios_mother[i,1]]==0 else len(np.argwhere((trios_mother[i]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cm_trios[trios_mother[i,1]]
                        Lm1 = 0 if cm_trios[trios_mother[i+1,1]]==0 else len(np.argwhere((trios_mother[i+1]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cm_trios[trios_mother[i+1,1]]
                        Lf0 = 0 if cf_trios[trios_father[i,2]]==0 else len(np.argwhere((trios_father[i]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cf_trios[trios_father[i,2]]
                        Lf1 = 0 if cf_trios[trios_father[i+1,2]]==0 else len(np.argwhere((trios_father[i+1]==x[rj:rj+3,:n_trios].T).all(axis=1)))/cf_trios[trios_father[i+1,2]]

                        # for each combination there are two possibilities
                        # calculate posterior probability for one possibility, then the other one has prob. 1-p
                        # posterior = priorA * likelihood / priorB
                        #pm0 = ppA[trios_mother[i,1]]*Lm0/ppB_m0
                        #pm1 = ppA[trios_mother[i+1,1]]*Lm1/ppB_m1
                        #pf0 = ppA[trios_father[i,2]]*Lf0/ppB_f0
                        #pf1 = ppA[trios_father[i+1,2]]*Lf1/ppB_f1
                        #logger.info(f"{pm0=}, {pm1=}, {pf0=}, {pf1=}")

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
                            #logger.info(f"{j=}, {af[j]=}, {ppA=}, {ppB_m0=}, {Lm0=}, {pm0=}")
                            if pm0 > 1:
                                logger.info(f"reset pm0 to 1: {j=}, {af[j]=}, {ppA=}, {ppB_m0=}, {Lm0=}, {pm0=}")
                                pm0 = 1
                            mother = rng.binomial(1, pm0, size= nm_na0)
                            x[rj+1, id_na_m0] = mother.reshape(nm_na0,1)+shift[i]
                        if nm_na1 > 0:
                            pm1 = ppA[trios_mother[i+1,1]]*Lm1/ppB_m1
                            #logger.info(f"{j=}, {af[j]=}, {ppA=}, {ppB_m1=}, {Lm1=}, {pm1=}")
                            # pm1 gives probability of getting Xm = 2
                            # 1-pm1 gives probabiltiy of getting Xm = 1
                            # use 1-pm1 to make it consistent the other combinations
                            if pm1 > 1:
                                logger.info(f"reset pm1 to 1: {j=}, {af[j]=}, {ppA=}, {ppB_m1=}, {Lm1=}, {pm1=}")
                                pm1 = 1
                            mother = rng.binomial(1, (1-pm1), size= nm_na1)
                            x[rj+1, id_na_m1] = mother.reshape(nm_na1,1)+shift[i+1]
                        if nf_na0 > 0:
                            pf0 = ppA[trios_father[i,2]]*Lf0/ppB_f0
                            #logger.info(f"{j=}, {af[j]=}, {ppA=}, {ppB_f0=}, {Lf0=}, {pf0=}")
                            if pf0 > 1:
                                logger.info(f"reset pf0 to 1: {j=}, {af[j]=}, {ppA=}, {ppB_f0=}, {Lf0=}, {pf0=}")
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
                                logger.info(f"reset pf1 to 1: {j=}, {af[j]=}, {ppA=}, {ppB_f1=}, {Lf1=}, {pf1=}")
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
                        #logger.info(f"{j=}, {af[j]=}, {pm=}, {pf=}")

                        # draw from binomial distribution according to posterior probability
                        # add shift for those values where the possiblities are 1 and 2
                        if nm_na > 0:
                            if pm > 1:
                                logger.info(f"reset pm to 1: {j=}, {af[j]=}, {ppA=}, {ppB_m=}, {Lm=}, {pm=}")
                                pm = 1
                            mother = rng.binomial(1, pm, size= nm_na)
                            x[rj+1, id_na_m] = mother.reshape(nm_na,1)+shift[i]
                        if nf_na > 0:
                            if pf > 1:
                                logger.info(f"reset pf to 1: {j=}, {af[j]=}, {ppA=}, {ppB_f=}, {Lf=}, {pf=}")
                                pf = 1
                            father = rng.binomial(1, pf, size= nf_na)+shift[i]
                            x[rj+2, id_na_f] = father.reshape(nf_na,1)


        if l == inputfiles[0]:
            z = zarr.array(x.T, chunks=(None,1000))
        else:
            z.append(x.T, axis=1)

    logger.info(f"{z.info=}")
    zarr.save(dir+'/genotype.zarr', z)


##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing.')
    parser.add_argument('--inputfiles', type=str, nargs='+', help='name of input files', required=True)
    parser.add_argument('--index_trios', type=str, help='List of ordered indices for trios.', required=True)
    parser.add_argument('--index_duos', type=str, help='List of ordered indices for duos.')
    parser.add_argument('--dir', type=str, help='Path to output directory.', required=True)
    parser.add_argument('--only_trios', type=bool, default=False, help='Only process trios (default is false)')
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
    main(inputfiles = args.inputfiles, # inputfile
        index_trios = args.index_trios, # list of indices
        index_duos = args.index_duos, # list of indices
        only_trios = args.only_trios, # boolean to only process trios
        dir = args.dir, # path to output directory
        k = 4, # number of genetic components
        ) 
    logger.info("Done.")
