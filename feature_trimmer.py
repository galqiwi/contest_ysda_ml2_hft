import numpy as np

from hft_model import SavableModel, TrainableModel
from utils import get_basic_asset_columns
from constants import TARGET_NAME

from typing import List, Optional
import logging
import pandas as pd
import os
import json
from tqdm import tqdm


class FeatureTrimmer(SavableModel, TrainableModel):
    CONFIG_NAME: str = 'config.json'

    assets: List[str]
    log: logging.Logger

    def __init__(self, trim_sigmas: int, log: logging.Logger):
        self.log = log
        self.trim_sigmas = trim_sigmas
        self.min_value_by_feature_name = None
        self.max_value_by_feature_name = None

    def fit(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame]):
        assert df_val is None
        assert TARGET_NAME not in df_train.columns
        mean = pd.Series(np.mean(df_train.values, axis=0), index=df_train.columns).astype(np.float64)
        # for some reason df_train.std() works really bad
        std = pd.Series(np.std(df_train.values, axis=0), index=df_train.columns).astype(np.float64)
        self.min_value_by_feature_name = dict(mean - std * self.trim_sigmas)
        self.max_value_by_feature_name = dict(mean + std * self.trim_sigmas)

    def predict(self, df: pd.DataFrame) -> pd.Series:
        assert set(self.min_value_by_feature_name.keys()) == set(df.columns)
        for feature in df.columns:
            max_value = self.max_value_by_feature_name[feature]
            min_value = self.min_value_by_feature_name[feature]
            too_large_samples = (df[feature] > max_value)
            too_small_samples = (df[feature] > min_value)
            df[feature][too_large_samples] = max_value
            df[feature][too_small_samples] = min_value

    def save(self, dir_path: str):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        with open(os.path.join(dir_path, self.CONFIG_NAME), 'w') as file:
            json.dump({
                'min_value_by_feature_name': self.min_value_by_feature_name,
                'max_value_by_feature_name': self.max_value_by_feature_name,
                'trim_sigmas': self.trim_sigmas,
            }, file)

    def load(self, dir_path: str):
        assert os.path.isdir(dir_path)
        with open(os.path.join(dir_path, self.CONFIG_NAME), 'r') as file:
            file_data = json.load(file)
        assert file_data['trim_sigmas'] == self.trim_sigmas
        self.min_value_by_feature_name = file_data['min_value_by_feature_name']
        self.max_value_by_feature_name = file_data['max_value_by_feature_name']
