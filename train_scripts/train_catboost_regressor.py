from utils import read_config, setup_logging

from hft_catboost_regressor import HFTCatboostRegressor
from train_scripts.train_utils import train_regressor


def main():
    log = setup_logging()
    config = read_config()

    regressor = HFTCatboostRegressor(
        config=config['catboost']['config'],
        log=log
    )
    train_regressor(regressor, save_path=config['catboost']['path'], log=log)


if __name__ == '__main__':
    main()
