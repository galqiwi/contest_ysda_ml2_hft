import os.path

from hft_model import SavableModel, TrainableModel
from feature_generator import FeatureGenerator
from typing import Dict, Optional, Any, Union
from ml_utils import do_epoch, get_adam_lr
import logging
import pandas as pd
from constants import TARGET_NAME
import numpy as np
import json
import torch
import torch.nn as nn
import datetime

try:
    import wandb
except ModuleNotFoundError:
    wandb = None


class HFTLSTMModel(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 25, num_lstm_layers: int = 4, dropout: float = 0.3):
        super().__init__()
        self.in_features = in_features
        self.hidden_dim = hidden_dim
        self.input_bn = nn.BatchNorm1d(num_features=in_features, affine=False)
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_lstm_layers,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)

    def _calc_lstm_outputs(self, features):
        batch_size, series_len, n_features = features.shape
        assert n_features == self.in_features
        x = features
        assert x.shape == (batch_size, series_len, n_features)
        x = torch.permute(x, (0, 2, 1))
        assert x.shape == (batch_size, n_features, series_len)
        x = self.input_bn(x)
        assert x.shape == (batch_size, n_features, series_len)
        x = torch.permute(x, (0, 2, 1))
        assert x.shape == (batch_size, series_len, n_features)
        x, _ = self.lstm(x)
        assert x.shape == (batch_size, series_len, self.hidden_dim)
        return x

    def get_all_targets(self, features):
        batch_size, series_len, n_features = features.shape

        x = self._calc_lstm_outputs(features)
        assert x.shape == (batch_size, series_len, self.hidden_dim)
        if batch_size != 1:
            raise NotImplemented
        x = x[0]
        assert x.shape == (series_len, self.hidden_dim)
        x = self.dropout(x)
        assert x.shape == (series_len, self.hidden_dim)
        x = self.fc(x)
        assert x.shape == (series_len, 1)
        return x[:, 0]

    def forward(self, features):
        batch_size, series_len, n_features = features.shape

        x = self._calc_lstm_outputs(features)
        assert x.shape == (batch_size, series_len, self.hidden_dim)
        x = x[:, -1, :]
        assert x.shape == (batch_size, self.hidden_dim)
        x = self.dropout(x)
        assert x.shape == (batch_size, self.hidden_dim)
        x = self.fc(x)
        assert x.shape == (batch_size, 1)
        return x[:, 0]


class THFTSeriesDataset(torch.utils.data.Dataset):
    def __init__(self, features_df, target_series, series_len=32):
        self.series_len = series_len
        self.features_df = features_df.values.astype(np.float16)
        self.target_series = target_series.values.astype(np.float16)
        n_samples, n_features = self.features_df.shape
        assert self.target_series.shape == (n_samples,)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.features_df) - self.series_len + 1):
            raise IndexError

        assert 0 <= idx + self.series_len <= len(self.features_df)

        return (
            self.features_df[idx: idx + self.series_len],
            self.target_series[idx + self.series_len - 1],
        )

    def __len__(self):
        return len(self.features_df) - self.series_len + 1


class HFTMlRegressor(SavableModel, TrainableModel):
    CONFIG_FILENAME = 'config.json'
    MODEL_FILENAME = 'regressor.pt'

    def __init__(
            self,
            config: Dict[str, Any],
            training_save_path: str,
            last_epoch_save_path: str,
            log: logging.Logger,
            use_wandb: bool = False
    ):
        self.log = log
        self.use_wandb = use_wandb
        self.training_save_path = training_save_path
        self.last_epoch_save_path = last_epoch_save_path
        self.config = config
        self.feature_generator = FeatureGenerator(assets=config['assets'], log=log)
        self.batch_size = config['batch_size']
        self.hidden_dim = config['hidden_dim']
        self.num_lstm_layers = config['num_lstm_layers']
        self.adam_lr = config['adam_lr']
        self.n_epochs = config['n_epochs']
        self.n_batches_per_epoch = config['n_batches_per_epoch']
        self.scheduler_config = config['scheduler']
        self.droupout_percent = config['droupout_percent']

        sample_top_features_config = config.get('sample_top_features')

        self.do_sample_top_features = sample_top_features_config is not None
        self.n_top_features = None
        self.feature_importance_path = None
        if sample_top_features_config is not None:
            self.n_top_features = sample_top_features_config['n_top_features']
            self.feature_importance_path = sample_top_features_config['feature_importance_path']

        self.regressor = None

    @staticmethod
    def _get_device():
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _start_wandb(self, df_train, df_val):
        if not self.use_wandb:
            return None
        name = 'test-({})'.format(
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        wandb_config = self.config.copy()
        wandb_config['train_size'] = len(df_train)
        wandb_config['val_size'] = len(df_val)
        wandb_run = wandb.init(project='contest_ysda', name=name, config=wandb_config)
        wandb_run.log_code('.')
        return wandb_run

    def _log_wandb(self, message):
        if not self.use_wandb:
            return None
        wandb.log(message)

    def _preprocess_features(self, df: pd.DataFrame):
        if not self.do_sample_top_features:
            return df
        feature_importance_df = pd.read_csv(self.feature_importance_path)
        feature_importance_df = feature_importance_df.sort_values('feature_importance', ascending=False)
        assert set(feature_importance_df['feature']) == set(df.columns)
        output_features = list(feature_importance_df['feature'])
        assert len(output_features) >= self.n_top_features
        output_features = output_features[:self.n_top_features]
        return df[output_features].copy()

    def fit(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame]):
        wandb_run = self._start_wandb(df_train, df_val)

        train_target = df_train[TARGET_NAME]
        val_target = df_val[TARGET_NAME]
        df_train = self.feature_generator.predict(df_train.drop(columns=[TARGET_NAME]))
        df_val = self.feature_generator.predict(df_val.drop(columns=[TARGET_NAME]))

        df_train = self._preprocess_features(df_train)
        df_val = self._preprocess_features(df_val)

        assert len(df_val.columns) == len(df_train.columns)
        n_features = len(df_val.columns)

        device = self._get_device()
        self.log.info(f'{device=}')

        train_dataset = THFTSeriesDataset(df_train, train_target)
        val_dataset = THFTSeriesDataset(df_val, val_target)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, shuffle=True,
            batch_size=self.batch_size, num_workers=0, drop_last=True
        )
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset, shuffle=True,
            batch_size=self.batch_size, num_workers=0, drop_last=True
        )

        self.regressor = HFTLSTMModel(
            in_features=n_features,
            hidden_dim=self.hidden_dim,
            num_lstm_layers=self.num_lstm_layers,
            dropout=self.droupout_percent / 100,
        ).to(device)

        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=self.adam_lr)

        scheduler = None

        if self.scheduler_config['type'] == 'exponential':
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.scheduler_config['scheduler_step'],
                gamma=self.scheduler_config['scheduler_gamma']
            )
        elif self.scheduler_config['type'] == 'linear':
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1,
                end_factor=0,
                total_iters=self.n_epochs
            )

        assert scheduler is not None, 'scheduler is not specified'

        self.log.info("started training")

        train_stats = []
        val_stats = []
        best_score = None
        best_epoch = 0
        val_score = 0
        for epoch_idx in range(self.n_epochs):
            lr = get_adam_lr(optimizer)
            train_stat = do_epoch(
                model=self.regressor,
                optimizer=optimizer,
                dataloader=train_dataloader,
                device=device,
                is_train=True,
                verbose=False,
                max_length=self.n_batches_per_epoch,
            )
            train_stats.append(train_stat)
            val_stat = do_epoch(
                model=self.regressor,
                optimizer=None,
                dataloader=val_dataloader,
                device=device,
                is_train=False,
                verbose=False,
                max_length=self.n_batches_per_epoch,
            )
            val_stats.append(val_stat)

            scheduler.step()
            
            train_score = train_stat['r2_score']
            val_score = val_score * 0.9 + 0.1 * val_stat['r2_score']

            if best_score is None or best_score < val_score:
                best_score = val_score
                best_epoch = epoch_idx
                self.save(self.training_save_path)

            self.log.info(
                f"({epoch_idx})\t"
                f"train: {train_score:.6f}\t"
                f"val: {val_score:.6f}\t"
                f"best: {best_score:.6f} ({best_epoch})\t"
                f"lr:{lr}"
            )
            self._log_wandb({
                'lr': lr,
                'epoch_idx': epoch_idx,
                'train_r2_score': train_stat['r2_score'],
                'train_loss_value': train_stat['loss_value'],
                'val_r2_score': val_stat['r2_score'],
                'val_loss_value': val_stat['loss_value'],
                'best_val_r2_score': best_score,
            })

        self.save(self.last_epoch_save_path)

        if wandb_run is not None:
            wandb_run.finish()

        self.load(self.training_save_path)

    def predict(self, df: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        assert TARGET_NAME not in df.columns
        df = self.feature_generator.predict(df)
        df = self._preprocess_features(df)
        device = self._get_device()

        self.regressor.eval()
        with torch.no_grad():
            output = self.regressor.get_all_targets(
                torch.Tensor(df.values).to(device)[None, :, :]
            ).detach().cpu().numpy()

        return pd.Series(output, index=df.index)

    def save(self, dir_path: str):
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        with open(os.path.join(dir_path, self.CONFIG_FILENAME), 'w') as file:
            json.dump({
                'config': self.config,
                'n_features': self.regressor.in_features,
            }, file)
        torch.save(self.regressor.state_dict(), os.path.join(dir_path, self.MODEL_FILENAME))

    def load(self, dir_path: str) -> "SavableModel":
        assert os.path.isdir(dir_path)
        with open(os.path.join(dir_path, self.CONFIG_FILENAME), 'r') as file:
            file_data = json.load(file)
            assert file_data['config'] == self.config
            n_features = file_data['n_features']

        self.regressor = HFTLSTMModel(
            in_features=n_features,
            hidden_dim=self.hidden_dim,
            num_lstm_layers=self.num_lstm_layers,
            dropout=self.droupout_percent / 100,
        ).to(self._get_device())

        self.regressor.load_state_dict(torch.load(
            os.path.join(dir_path, self.MODEL_FILENAME),
            map_location=self._get_device()
        ))
