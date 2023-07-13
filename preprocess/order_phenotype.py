#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Install dependencies:
```
pip install numpy scikit-allel loguru pandas
```
python order_phenotype.py --y phenotype.tsv --index_trios trios.ped --index_duos duos.ped --name ordered_phenotype.txt

lines = list(np.loadtxt("rm_id_bmi.txt", delimiter=",").astype('int'))
x = np.delete(x, lines, axis=0)
"""
import numpy as np
import sys
import argparse
from loguru import logger
import allel
import pandas as pd


def main(yfile, index_trios, index_duos, name):

    # read in indices for child
    id = pd.concat(
        (
            pd.read_csv(index_trios, delimiter="\t", names=["ID_STR"], usecols=[0]),
            pd.read_csv(index_duos, delimiter="\t", names=["ID_STR"], usecols=[0]),
        ),
        axis=0,  ignore_index=True)
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
    np.savetxt(name, result.dropna().values)
    np.savetxt("rm_id_"+name, id_na, fmt='%i', delimiter=",")
    
    

##########################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocessing.')
    parser.add_argument('--y', type=str, help='name of input files', required=True)
    parser.add_argument('--index_trios', type=str, help='List of ordered indices for trios.', required=True)
    parser.add_argument('--index_duos', type=str, help='List of ordered indices for duos.', required=True)
    parser.add_argument('--name', type=str, help='name of output file.', required=True)
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
        name = args.name, # output filename
        ) 
    logger.info("Done.")
