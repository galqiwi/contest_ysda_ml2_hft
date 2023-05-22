import os

import pandas as pd
import numpy as np
from typing import List, Dict, Any
import itertools
from sklearn.metrics import r2_score
import logging
import uuid
import sys
import json
from constants import TARGET_NAME, MAX_BOOK_DEPTH, EQUAL_INDEXES_ASSERT_SAMPLES

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(x):
        return x


def read_data(data_dir='data_by_days', percentage=None, sample_from_start=True):
    files = os.listdir(data_dir)
    if percentage is not None:
        n_files = round(len(files) * percentage)
        n_files = max(n_files, 1)
        n_files = min(n_files, len(files))
        if sample_from_start:
            files = files[:n_files]
        else:
            files = files[-n_files:]

    output = []
    for file in files:
        assert file.startswith('train_')
        assert file.endswith('.feather')
        output.append(pd.read_feather(os.path.join(data_dir, file)))
    output = pd.concat(output)
    output.reset_index(drop=True, inplace=True)

    output.set_index('timestamp', inplace=True)
    output.sort_index(inplace=True)

    return output


def assert_equal_indexes(a, b):
    assert a.index.shape == b.index.shape

    n_corners_check = min(len(a), EQUAL_INDEXES_ASSERT_SAMPLES)
    idx_to_check = []
    idx_to_check.extend(range(n_corners_check))
    idx_to_check.extend(range(len(a) - n_corners_check, len(a)))
    idx_to_check.extend(np.random.randint(low=0, high=len(a), size=EQUAL_INDEXES_ASSERT_SAMPLES))

    assert (
        np.array(a.index[idx_to_check]).astype(int) ==
        np.array(b.index[idx_to_check]).astype(int)
    ).all()


def get_entry_name(asset_name: str, book_side: str, idx: int, entry_type: str):
    assert book_side in ('ask', 'bid')
    assert 0 <= idx < MAX_BOOK_DEPTH
    return f'{asset_name}_{book_side}{idx}_{entry_type}'


def get_basic_asset_columns(asset_name: str) -> List[str]:
    return [
        get_entry_name(
            asset_name=asset_name,
            book_side=book_side,
            idx=idx,
            entry_type=entry_type
        )
        for (book_side, idx, entry_type) in itertools.product(
            ('ask', 'bid'),
            range(MAX_BOOK_DEPTH),
            ('price', 'qty')
        )
    ]


def list_assets(df: pd.DataFrame) -> List[str]:
    assets = set([column.split('_')[0] for column in df.columns if column != 'timestamp' and column != TARGET_NAME])
    assets = sorted(assets)
    return assets


def count_moving_average(input_series: pd.Series, smoothness: float = 0.5) -> pd.Series:
    if len(input_series) == 0:
        return pd.Series()
    output = []
    values = input_series.values
    avg = values[0]
    for value in values:
        avg = avg * smoothness + value * (1 - smoothness)
        output.append(avg)
    return pd.Series(output)


def bootstrap_r2_score(target, prediction, bootstrap_samples=10, verbose=False):
    assert_equal_indexes(target, prediction)
    assert len(target) == len(prediction)
    arr_size = len(target)

    r2_scores = []

    counting_iterator = range(bootstrap_samples)
    if verbose:
        counting_iterator = tqdm(counting_iterator)

    for _ in counting_iterator:
        idx = np.random.choice(np.arange(arr_size), size=arr_size)
        r2_scores.append(r2_score(target[idx], prediction[idx]))

    return (
        np.mean(r2_scores),
        np.std(r2_scores) * np.sqrt(bootstrap_samples / (bootstrap_samples - 1))
    )


def get_asset_name(asset_df: pd.DataFrame) -> str:
    asset_name = [column for column in asset_df.columns if column != 'timestamp'][0].split('_')[0]

    for column in asset_df.columns:
        if column == 'timestamp':
            continue
        assert column.startswith(asset_name + '_'), f'{column} does not start with {asset_name}'
    return asset_name


class CatBoostLogger:
    def __init__(self, log: logging.Logger):
        self.log = log

    def write(self, message: str):
        if message.endswith('\n'):
            message = message[:-1]
        self.log.info(message)


def _get_dummy_logger() -> logging.Logger:
    output = logging.getLogger(uuid.uuid4().hex)
    output.handlers.clear()
    assert len(output.handlers) == 0
    return output


def setup_logging() -> logging.Logger:
    logging.getLogger().handlers.clear()
    formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
    output = logging.getLogger('train')
    output.setLevel(logging.DEBUG)
    output.addHandler(logging.StreamHandler(sys.stdout))
    assert len(output.handlers) == 1
    output.handlers[0].setFormatter(formatter)
    return output


def read_config_fallback() -> Dict[str, Any]:
    with open('config.yml', 'r') as file:
        file_raw = file.read()
    with open('config.json', 'r') as file:
        config_cache = json.load(file)
    assert file_raw == config_cache['file_raw']
    return config_cache['value']


def read_config() -> Dict[str, Any]:
    if yaml is None:
        return read_config_fallback()
    with open('config.yml', 'r') as file:
        file_raw = file.read()
        output = yaml.safe_load(file_raw)
    with open('config.json', 'w') as file:
        json.dump({
            'file_raw': file_raw,
            'value': output,
        }, file)
    return output


DummyLogger = _get_dummy_logger()
