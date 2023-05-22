from utils import read_config, setup_logging

from hft_ml_regressor import HFTMlRegressor
from train_scripts.train_utils import train_regressor


def main():
    log = setup_logging()
    config = read_config()

    regressor = HFTMlRegressor(
        config=config['ml']['config'],
        training_save_path=config['ml']['path'],
        last_epoch_save_path=config['ml']['last_epoch_path'],
        log=log,
        use_wandb=True
    )
    train_regressor(regressor, save_path=config['ml']['path'], log=log)


if __name__ == '__main__':
    main()
