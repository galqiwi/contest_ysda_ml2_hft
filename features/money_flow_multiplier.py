import pandas as pd

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from utils import get_entry_name, get_asset_name
from features.base_feature import BaseFeature
from numba import njit


class MoneyFlowMultiplier(BaseFeature, ABC):
    def __init__(self, window_size: int):
        self.window_size = window_size

    @staticmethod
    @njit
    def _do_calculate(low_array: np.array, high_array: np.array, window_size: int) -> np.array:
        assert low_array.shape == high_array.shape
        output = np.zeros(shape=(len(low_array),), dtype=low_array.dtype)
        for i in range(window_size - 1, len(low_array)):
            # window is [window_begin, window_end) == [i - window_size + 1, i + 1)
            # so, window has size window_size and ends at i-th element
            window_begin, window_end = i - window_size + 1, i + 1
            low_window = low_array[window_begin: window_end]
            high_window = high_array[window_begin: window_end]
            low, high = np.min(low_window), np.max(high_window)
            last = (low_window[-1] + high_window[-1]) / 2
            if high - low == 0:
                continue
            output[i] = (2 * last - high - low) / (high - low)
        return output

    def calculate(self, asset_df: pd.DataFrame) -> Tuple[pd.Series, str]:
        asset_name = get_asset_name(asset_df)
        feature_name = f'{asset_name}_money_flow_multiplier_{self.window_size}'

        output = self._do_calculate(
            low_array=asset_df[get_entry_name(
                asset_name=asset_name,
                book_side='bid',
                idx=0,
                entry_type='price'
            )].values,
            high_array=asset_df[get_entry_name(
                asset_name=asset_name,
                book_side='ask',
                idx=0,
                entry_type='price'
            )].values,
            window_size=self.window_size
        )
        assert output.shape == (len(asset_df),)
        output = pd.Series(output, index=asset_df.index)

        return output, feature_name
