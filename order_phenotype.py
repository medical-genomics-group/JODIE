#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy loguru pandas
```
python order_phenotype.py 
--y phenotype.tsv --index_trios trios.ped 
--index_duos duos.ped --pheno pheno
--outdir dir/

--y           phenotype file in tab-delimited tsv or csv format with ID, VAL or tab-delimited .phen file with ID, FID, VAL with header (required)
--index_trios trio index file, same as for preparing_vcf_data.py; first column is child id (required)
--index_duos  duos index file, same as for preparing_vcf_data.py; first column is child id 
--outdir      name of output directory (required)
--pheno       name of phenotype for output file (required)
Phenotype file will first contain trios, then duos, ordered according to index.
"""
import numpy as np
import sys
import argparse
from loguru import logger
import pandas as pd


def main(yfile, index_trios, index_duos, outdir, pheno):

    # read in indices for child
    if index_duos is not None:
        id = pd.concat(
            (
                pd.read_csv(index_trios, delimiter="\t", names=["ID_STR"], usecols=[0]),
                pd.read_csv(index_duos, delimiter="\t", names=["ID_STR"], usecols=[0]),
            ),
            axis=0,  ignore_index=True)
    else:
        id = pd.read_csv(index_trios, delimiter="\t", names=["ID_STR"], usecols=[0]) 
    logger.info(f"{id=}")

    # read in phenotype file, add NAN value for missing values
    if yfile.endswith(".phen"):
        y = pd.concat(
            (
                pd.read_csv(yfile, delimiter="\t", names = ["ID_STR", "ID_STR1", "VALUE"], dtype={"ID_STR": str, "ID_STR1": str, "VALUE": float}, usecols=[0, 2]),
                pd.DataFrame([{"ID_STR": np.nan, "VALUE": np.nan}]),
            )
        )
    else:
        y = pd.concat(
            (
                pd.read_csv(yfile, delimiter="\t", names=["ID_STR", "VALUE"], dtype={"ID_STR": str, "VALUE": float}),
                pd.DataFrame([{"ID_STR": np.nan, "VALUE": np.nan}]),
            )
        )
    logger.info(f"{y=}")
    ## assign values to id 
    id = id.assign(
        ID_STR=lambda x: np.where(
            x["ID_STR"].isin(y["ID_STR"].unique()), x["ID_STR"], np.nan
        )
    )
    result = y.set_index("ID_STR").loc[id["ID_STR"]]
    logger.info(f"{result=}")
    id_na = np.where(result['VALUE'].isna())
    logger.info(f"{id_na=}")
    logger.info(f"{result.dropna().values=}")
    logger.info(f"{result.dropna().values.shape=}")
    np.savetxt(f"{outdir}/ordered_{pheno}.txt", result.dropna().values)
    np.savetxt(f"{outdir}/rm_id_ordered_{pheno}.txt", id_na, fmt='%i', delimiter=",")
    

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing phenotype.')
    parser.add_argument('--y', type=str, help='name of input files', required=True)
    parser.add_argument('--index_trios', type=str, help='List of ordered indices for trios.', required=True)
    parser.add_argument('--index_duos', type=str, default=None, help='List of ordered indices for duos.')
    parser.add_argument('--outdir', type=str, help='name of output directory.', required=True)
    parser.add_argument('--pheno', type=str, help='name of phenotype for output file.', required=True)
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
    main(yfile = args.y, # inputfile
        index_trios = args.index_trios, # list of indices
        index_duos = args.index_duos, # list of indices
        outdir = args.outdir, # output directory
        pheno = args.pheno,
        ) 
    logger.info("Done.")
