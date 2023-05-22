import numpy as np

from hft_model import SavableModel
from features.mid import Mid
from features.wap import Wap
from features.moving_average import MovingAverage
from features.diff import Diff
from features.delta import Delta
from features.autocorrelation import WindowedAutoCorrelation
from features.money_flow_multiplier import MoneyFlowMultiplier
from utils import get_asset_name, get_basic_asset_columns

from typing import List
import logging
import pandas as pd
import os
import json
import datetime
import constants
import hashlib

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(x):
        return x


class FeatureGenerator(SavableModel):
    CONFIG_NAME: str = 'config.json'

    assets: List[str]
    log: logging.Logger

    def __init__(self, assets: List[str], log: logging.Logger):
        self.log = log
        self.assets = assets

    @staticmethod
    def _do_add_features_to_asset(asset_df: pd.DataFrame) -> pd.DataFrame:
        asset_name = get_asset_name(asset_df)
        df = asset_df.copy()
        columns_to_drop = []

        mid = Mid().add_to_df(df)
        columns_to_drop.append(mid)
        for smoothness in [10, 50, 90, 99]:
            MovingAverage(base_feature_name=mid, smoothness=smoothness).add_to_df(df, subtract=mid)
        for window in range(1, 10):
            Delta(base_feature_name=mid, window=window).add_to_df(df)


        wap = Wap().add_to_df(df)
        wap_mid_diff = Diff(wap, mid).add_to_df(df)
        for window in range(1, 10):
            Delta(base_feature_name=wap, window=window).add_to_df(df)
            Delta(base_feature_name=wap_mid_diff, window=window).add_to_df(df)
        columns_to_drop.append(wap)


        for column in get_basic_asset_columns(asset_name):
            if not column.endswith('price'):
                continue
            Diff(column, mid).add_to_df(df)
            columns_to_drop.append(column)

        df.drop(columns=columns_to_drop, inplace=True)

        assert len(set(df.dtypes)) == 1, f'failed dtypes test {set(df.dtypes)}'

        assert df.isna().sum().sum() == 0, f'failed nan test: \n{df.isna().sum()}'
        return df

    @staticmethod
    def _get_df_hash(asset_df: pd.DataFrame):
        asset_df = asset_df.head(constants.FEATURE_GENERATOR_HASH_PREFIX)
        df = FeatureGenerator._do_add_features_to_asset(asset_df)

        columns = list(df.columns)
        shape = list(df.shape)
        np.random.seed(0)
        idx = np.random.randint(0, len(df), constants.FEATURE_GENERATOR_HASH_SAMPLES)
        hashes = list(pd.util.hash_pandas_object(df.iloc[idx]))

        output = json.dumps([columns, shape, hashes])
        return hashlib.sha256(output.encode()).hexdigest()

    @staticmethod
    def _add_features_to_asset(asset_df: pd.DataFrame) -> pd.DataFrame:
        return FeatureGenerator._do_add_features_to_asset(asset_df)

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        self.log.debug(f"generating features")
        dfs = []
        for asset in self.assets:
            dfs.append(self._add_features_to_asset(df[get_basic_asset_columns(asset)]))
            self.log.debug(f"got features for asset {asset}")

        output = pd.concat(dfs, axis=1)

        assert len(output.columns) == sum([len(asset_df.columns) for asset_df in dfs])
        assert len(output) == len(df)

        self.log.debug(f'got {len(output.columns)} features')

        assert output.isna().sum().sum() == 0, 'failed nan test'
        return output

    def save(self, dir_path: str):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        with open(os.path.join(dir_path, self.CONFIG_NAME), 'w') as file:
            json.dump({
                'assets': self.assets,
            }, file)

    def load(self, dir_path: str):
        assert os.path.isdir(dir_path)
        with open(os.path.join(dir_path, self.CONFIG_NAME), 'r') as file:
            assert json.load(file) == {
                'assets': self.assets
            }
