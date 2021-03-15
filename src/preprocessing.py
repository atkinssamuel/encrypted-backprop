from src.constants import Files, Directories
from os import path
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

import pandas as pd
import zipfile
import os

SEED = 88


def rebalance(df):
    """
    Re-balances a DataFrame by oversampling using SMOTE
    :param df: Pandas DataFrame
    :return: Pandas DataFrame
    """
    sm = SMOTE()
    X, y = sm.fit_resample(df.drop('Class', axis=1), df['Class'])
    return pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)


def standardize(df):
    """
    Standardizes the non-predictor columns in the input DataFrame.
    :param df: Pandas DataFrame
    :return: Pandas DataFrame
    """
    df_inputs = df.drop("Class", axis=1)
    df_inputs_columns = df_inputs.columns
    df_outputs = df["Class"]
    standardizer = preprocessing.StandardScaler().fit(df_inputs)
    df_inputs = standardizer.transform(df_inputs)
    return pd.concat([pd.DataFrame(df_inputs, columns=df_inputs_columns), df_outputs], axis=1)


def preprocess():
    """
    This function unzips and extracts data into a DataFrame. Then, this function re-balances and standardizes the raw
    data before storing the result in the "data" folder.
    :return: None
    """
    if not path.exists(Files.raw_data):
        with zipfile.ZipFile(Files.zipped_data, 'r') as zip_ref:
            zip_ref.extractall(Directories.raw)

    if not path.exists(Directories.data):
        os.mkdir(Directories.data)

    if not path.exists(Files.balanced_data):
        df = pd.read_csv(Files.raw_data)

        df = rebalance(df)
        df = standardize(df)

        df.to_pickle(Files.balanced_data)

