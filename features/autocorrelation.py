import pandas as pd

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
from utils import get_entry_name, get_asset_name
from features.base_feature import BaseFeature
from numba import njit


class WindowedAutoCorrelation(BaseFeature, ABC):
    def __init__(self, window_size: int, shift: int, base_feature: str):
        self.window_size = window_size
        self.shift = shift
        self.base_feature = base_feature
        assert self.shift < self.window_size

    @staticmethod
    @njit
    def _do_calculate(values: np.array, window_size: int, shift: int) -> np.array:
        output = np.zeros(shape=(len(values),), dtype=values.dtype)
        for i in range(window_size - 1, len(values)):
            # window is [window_begin, window_end) == [i - window_size + 1, i + 1)
            # so, window has size window_size and ends at i-th element
            window_begin, window_end = i - window_size + 1, i + 1
            window = values[window_begin: window_end]
            output[i] = (
                np.sum((
                    (window[:-shift] - np.mean(window[:-shift])) *
                    (window[shift:] - np.mean(window[shift:]))
                )) / (len(window) - shift)
            )
        return output

    def calculate(self, asset_df: pd.DataFrame) -> Tuple[pd.Series, str]:
        asset_name = get_asset_name(asset_df)
        feature_name = f'{asset_name}_autocorr_{self.base_feature}_{self.window_size}_{self.shift}'

        output = self._do_calculate(
            values=asset_df[self.base_feature].values,
            window_size=self.window_size,
            shift=self.shift,
        )
        assert output.shape == (len(asset_df),)
        output = pd.Series(output, index=asset_df.index)

        return output, feature_name
