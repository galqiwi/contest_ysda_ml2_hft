from utils import bootstrap_r2_score

import pandas as pd
from typing import Optional
from typing import Tuple, Union
from abc import ABC, abstractmethod
from constants import TARGET_NAME


class HFTModel(ABC):
    @abstractmethod
    def predict(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class TrainableModel(HFTModel):
    @abstractmethod
    def fit(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame]):
        pass


class SavableModel(HFTModel):
    @abstractmethod
    def save(self, dir_path: str):
        pass

    @abstractmethod
    def load(self, dir_path: str):
        pass


def test_model(model: HFTModel, df: pd.DataFrame) -> Tuple[float, float]:
    prediction = model.predict(df.drop(columns=[TARGET_NAME]))
    assert isinstance(prediction, pd.Series)
    return bootstrap_r2_score(
        target=df[TARGET_NAME],
        prediction=prediction
    )
