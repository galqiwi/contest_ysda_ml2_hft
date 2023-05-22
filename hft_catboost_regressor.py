import os.path

from hft_model import SavableModel, TrainableModel
from feature_generator import FeatureGenerator
from utils import CatBoostLogger
from catboost import CatBoostRegressor, Pool
from catboost.utils import get_gpu_device_count
from typing import List, Dict, Optional, Any, Union
import logging
import pandas as pd
from constants import TARGET_NAME
import json


class HFTCatboostRegressor(SavableModel, TrainableModel):
    CONFIG_FILENAME = 'config.json'
    MODEL_FILENAME = 'regressor.catboost'

    def __init__(self, config: Dict[str, Any], log: logging.Logger):
        self.log = log
        self.config = config
        self.feature_generator = FeatureGenerator(assets=config['assets'], log=log)
        self.regressor = CatBoostRegressor(
            iterations=config['n_iterations'],
            verbose=False,
            use_best_model=True,
            task_type=('GPU' if get_gpu_device_count() > 0 else None),
            eval_metric='R2'
        )
        self.do_save_feature_importance = config['feature_importance']['do_save']
        self.feature_importance_save_path = config['feature_importance']['path']

    def fit(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame]):
        train_target = df_train[TARGET_NAME]
        val_target = df_val[TARGET_NAME]
        df_train = self.feature_generator.predict(df_train.drop(columns=[TARGET_NAME]))
        df_val = self.feature_generator.predict(df_val.drop(columns=[TARGET_NAME]))

        self.regressor.fit(
            Pool(df_train, label=train_target),
            eval_set=Pool(df_val, label=val_target),
            verbose=True,
            log_cout=CatBoostLogger(self.log),
            log_cerr=CatBoostLogger(self.log),
        )
        feature_importance_df: pd.DataFrame = pd.DataFrame({
            'feature': self.regressor.feature_names_,
            'feature_importance': self.regressor.get_feature_importance(),
        }).sort_values('feature_importance', ascending=False)
        feature_importance_df = feature_importance_df[feature_importance_df['feature_importance'] > 0]
        feature_importance_df = feature_importance_df.reset_index(drop=True)
        self.log.info(feature_importance_df.to_string())
        if self.do_save_feature_importance:
            feature_importance_df.to_csv(self.feature_importance_save_path)

    def predict(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        df = self.feature_generator.predict(df)
        return pd.Series(self.regressor.predict(df), index=df.index)

    def save(self, dir_path: str):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        with open(os.path.join(dir_path, self.CONFIG_FILENAME), 'w') as file:
            json.dump({
                'config': self.config,
            }, file)
        self.regressor.save_model(os.path.join(dir_path, self.MODEL_FILENAME))

    def load(self, dir_path: str) -> "SavableModel":
        assert os.path.isdir(dir_path)
        with open(os.path.join(dir_path, self.CONFIG_FILENAME), 'r') as file:
            assert json.load(file) == {
                'config': self.config,
            }
        self.regressor.load_model(os.path.join(dir_path, self.MODEL_FILENAME))
