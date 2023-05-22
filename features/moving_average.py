import pandas as pd

from utils import get_asset_name, assert_equal_indexes
from features.base_feature import BaseFeature


class MovingAverage(BaseFeature):
    def __init__(self, base_feature_name: str, smoothness: int):
        assert 0 <= smoothness <= 100
        self.base_feature_name = base_feature_name
        self.smoothness = smoothness

    def calculate(self, asset_df: pd.DataFrame):
        feature_name = f'{self.base_feature_name}_moving_avg_{self.smoothness}'
        asset_name = get_asset_name(asset_df)
        assert self.base_feature_name.startswith(asset_name)

        input_series = asset_df[self.base_feature_name]
        if len(input_series) == 0:
            return pd.Series(), feature_name

        output = input_series.ewm(alpha=(100 - self.smoothness) / 100, adjust=False).mean().astype(input_series.dtype)
        assert_equal_indexes(output, asset_df)

        return output, feature_name
