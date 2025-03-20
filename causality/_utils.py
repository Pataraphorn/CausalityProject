import os
current_dir = os.getcwd()

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)

tqdm.pandas()


def load_df_from_path(path: str, type: str = None):
    print(f"Load data from type:{type} in path:{path} to Dataframe")
    if type == "parquet":
        df = pd.read_parquet(path)
    elif type == "npz":
        npz = np.load(path, allow_pickle=True)
        df = pd.DataFrame({file: npz[file] for file in npz.files})
    elif type == "pickle":
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
    return df
