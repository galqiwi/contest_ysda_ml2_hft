from abc import ABC, abstractmethod

from utils import assert_equal_indexes, get_asset_name
from typing import Tuple, Optional
import pandas as pd
import numpy as np


class BaseFeature(ABC):
    @abstractmethod
    def calculate(self, asset_df: pd.DataFrame) -> Tuple[pd.Series, str]:
        pass

    def add_to_df(self, asset_df: pd.DataFrame, norm_by: Optional[str] = None, subtract: Optional[str] = None) -> str:
        asset_name = get_asset_name(asset_df)

        values, name = self.calculate(asset_df=asset_df)
        assert_equal_indexes(values, asset_df)
        assert name not in asset_df.columns, f'already got {name}'
        assert name.startswith(asset_name)

        if subtract is not None:
            values -= asset_df[subtract]

        if norm_by is not None:
            values /= asset_df[norm_by]

        asset_df[name] = values  # .astype(np.float16)
        return name
