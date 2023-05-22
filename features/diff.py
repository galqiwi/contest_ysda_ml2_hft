import pandas as pd

from typing import Tuple
from utils import get_asset_name
from features.base_feature import BaseFeature


class Diff(BaseFeature):
    def __init__(self, a: str, b: str):
        self.a = a
        self.b = b

    def calculate(self, asset_df: pd.DataFrame) -> Tuple[pd.Series, str]:
        asset_name = get_asset_name(asset_df)
        assert self.a.startswith(asset_name)
        assert self.b.startswith(asset_name)
        output = asset_df[self.a] - asset_df[self.b]
        return output, f'{asset_name}_diff({self.a}, {self.b})'
