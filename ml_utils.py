import torch
from typing import Any, Iterable
import numpy as np
import pandas as pd
import torch.nn.functional as F

try:
    from torcheval.metrics import R2Score
except ModuleNotFoundError:
    R2Score = None

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(x):
        return x


class THFTDataset(torch.utils.data.Dataset):
    def __init__(self, features_df, target_series):
        self.features_df = features_df.values
        self.target_series = target_series.values
        n_samples, n_features = self.features_df.shape
        assert self.target_series.shape == (n_samples,)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.features_df)):
            raise IndexError
        return (
            self.features_df[idx].astype(np.float32),
            self.target_series[idx].astype(np.float32)
        )

    def __len__(self):
        return len(self.features_df)


class IteratorLimiter:
    def __init__(self, inner: Iterable[Any], max_length: int):
        self.inner = inner
        self.max_length = max_length
        self.idx = 0

    def __iter__(self):
        return IteratorLimiter(
            iter(self.inner),
            max_length=self.max_length
        )

    def __next__(self):
        if self.idx == self.max_length:
            raise StopIteration
        self.idx += 1
        return next(self.inner)


def get_adam_lr(optimizer):
    param_groups = optimizer.param_groups
    assert len(param_groups) == 1
    return param_groups[0]['lr']


def do_epoch(model, optimizer, dataloader, device, is_train=True, verbose=False, max_length=None) -> pd.Series:
    if is_train:
        model.train()
    else:
        model.eval()

    statistics = []

    samples_iterator = dataloader
    iterator_len = len(dataloader)

    if max_length is not None:
        samples_iterator = IteratorLimiter(samples_iterator, max_length=max_length)
        iterator_len = min(iterator_len, max_length)

    if verbose:
        samples_iterator = tqdm(
            samples_iterator,
            total=iterator_len
        )

    for features, targets in samples_iterator:
        features = features.to(device).to(dtype=torch.float32)
        targets = targets.to(device).to(dtype=torch.float32)

        if is_train:
            optimizer.zero_grad()

        predicts = model(features)
        loss_value = F.mse_loss(predicts, targets)

        if is_train:
            loss_value.backward()
            optimizer.step()

        r2_score = R2Score().to(device)
        r2_score.update(predicts, targets)
        r2_score = r2_score.compute()

        statistics.append({
            'loss_value': loss_value.detach().cpu().numpy(),
            'r2_score': r2_score.detach().cpu().numpy(),
        })
    return pd.DataFrame(statistics).mean()
