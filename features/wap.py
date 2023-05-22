import pandas as pd

from typing import Tuple
from utils import get_entry_name, get_asset_name
from features.base_feature import BaseFeature


class Wap(BaseFeature):
    def calculate(self, asset_df: pd.DataFrame) -> Tuple[pd.Series, str]:
        asset_name = get_asset_name(asset_df)
        output = (
            (
                asset_df[get_entry_name(asset_name, 'ask', 0, 'price')] *
                asset_df[get_entry_name(asset_name, 'ask', 0, 'qty')]
            ) + (
                asset_df[get_entry_name(asset_name, 'bid', 0, 'price')] *
                asset_df[get_entry_name(asset_name, 'bid', 0, 'qty')]
            )
        ) / (
            asset_df[get_entry_name(asset_name, 'ask', 0, 'qty')] +
            asset_df[get_entry_name(asset_name, 'bid', 0, 'qty')]
        )

        return output, f'{asset_name}_wap'
