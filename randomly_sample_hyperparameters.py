import subprocess
import os
from typing import Any
import yaml
import time
import random


def exec_command(command, env=None) -> str:
    if env is None:
        env = os.environ.copy()

    process = subprocess.run(command, capture_output=True, env=env)

    if process.returncode == 0:
        return process.stdout.decode()

    raise Exception('command {} failed\nstdout:\n{}\nstderr:\n{}'.format(
        command,
        process.stdout.decode(),
        process.stderr.decode(),
    ))


def generate_random_config() -> Any:
    ml_config = {
        'adam_lr': 0.01,
        'assets': ['SANHOK', 'MIRAMAR'],
        'batch_size': 1024,
        'hidden_dim': random.randint(20, 30),
        'n_batches_per_epoch': 50,
        'n_epochs': 100,
        'num_lstm_layers': 4,
        'droupout_percent': random.randint(20, 40),
        'sample_top_features': {
            'n_top_features': random.randint(60, 80),
            'feature_importance_path': "feature_importance.csv",
        },
        'scheduler': {'type': 'linear'},
    }
    return {
        'ml': {
            'config': ml_config,
            'path': 'model/ml_regressor',
        }
    }


def main():
    begin = time.time()
    with open('config.yml', 'w') as file:
        yaml.safe_dump(generate_random_config(), file)
    exec_command(['python3', 'train_scripts/train_lstm_regressor.py'])
    print(f'finished in {int(time.time() - begin)}s')


if __name__ == '__main__':
    main()
