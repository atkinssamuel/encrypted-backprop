from src.constants import Files, Directories
from os import path
import pandas as pd
import zipfile
import os

def preprocess():
    if not path.exists(Files.raw_data):
        with zipfile.ZipFile(Files.zipped_data, 'r') as zip_ref:
            zip_ref.extractall(Directories.raw)

    if not path.exists(Directories.data):
        os.mkdir(Directories.data)

    if not path.exists(Files.balanced_data) or True:
        df = pd.read_csv(Files.raw_data)
        df = pd.DataFrame()


        df.to_pickle(Files.balanced_data)
