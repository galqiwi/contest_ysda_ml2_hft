import pandas as pd
from utils import DummyLogger, read_config
import torch
from hft_ml_regressor import HFTMlRegressor


config = read_config()
regressor = HFTMlRegressor(
    config=config['ml']['config'],
    training_save_path=config['ml']['path'],
    last_epoch_save_path=config['ml']['last_epoch_path'],
    log=DummyLogger,
    use_wandb=True
)
regressor.load(config['ml']['path'])


def get_predict(df: pd.DataFrame):
    df.set_index(inplace=True, keys='timestamp')
    df.sort_index()
    return regressor.predict(df)
