import pandas as pd

from utils import get_asset_name
from features.base_feature import BaseFeature
import numpy as np


class Delta(BaseFeature):
    def __init__(self, base_feature_name: str, window: int):
        assert 0 <= window
        self.base_feature_name = base_feature_name
        self.window = window

    def calculate(self, asset_df: pd.DataFrame):
        output_name = f'{self.base_feature_name}_delta_{self.window}'
        asset_name = get_asset_name(asset_df)
        assert self.base_feature_name.startswith(asset_name)

        if self.window >= len(asset_df):
            return pd.Series(np.zeros(shape=(len(asset_df),)), index=asset_df.index), output_name

        assert self.window < len(asset_df)

        input_series_values = asset_df[self.base_feature_name].values
        if len(input_series_values) == 0:
            return pd.Series(), output_name

        output = input_series_values[self.window:] - input_series_values[:-self.window]
        assert len(output) == len(asset_df) - self.window
        output = np.pad(output, pad_width=(self.window, 0), mode='constant', constant_values=(0, 0))
        assert len(output) == len(asset_df)
        return pd.Series(output, index=asset_df.index), output_name
