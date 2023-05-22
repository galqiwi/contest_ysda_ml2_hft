import pandas as pd

from constants import TARGET_NAME
from utils import read_config, setup_logging, read_data, bootstrap_r2_score

from hft_catboost_regressor import HFTCatboostRegressor
from hft_ml_regressor import HFTMlRegressor
from train_scripts.train_utils import train_regressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    log = setup_logging()
    config = read_config()

    ml_regressor = HFTMlRegressor(
        config=config['ml']['config'],
        training_save_path=config['ml']['path'],
        log=log,
        use_wandb=True
    )
    ml_regressor.load(config['ml']['path'])

    cb_regressor = HFTCatboostRegressor(
        config=config['catboost']['config'],
        log=log
    )
    cb_regressor.load(config['catboost']['path'])

    base_train, test = train_test_split(
        read_data(percentage=None, sample_from_start=True),
        test_size=0.1,
        shuffle=False
    )
    base_val, test = train_test_split(test, test_size=0.5, shuffle=False)
    train, test = train_test_split(test, test_size=0.5, shuffle=False)
    val, test = train_test_split(test, test_size=0.5, shuffle=False)

    X = pd.DataFrame({
        'ml_regressor': ml_regressor.predict(train.drop(columns=[TARGET_NAME])),
        'cb_regressor': cb_regressor.predict(train.drop(columns=[TARGET_NAME])),
    }, index=train.index)
    y = train[TARGET_NAME]
    model = LinearRegression()
    model.fit(X, y)

    for df_name, df in [
        ('test', test),
        ('val', val),
        ('train', train),
    ]:
        X = pd.DataFrame({
            'ml_regressor': ml_regressor.predict(df.drop(columns=[TARGET_NAME])),
            'cb_regressor': cb_regressor.predict(df.drop(columns=[TARGET_NAME])),
        }, index=df.index)

        score = bootstrap_r2_score(
            target=df[TARGET_NAME],
            # prediction=X['ml_regressor']
            prediction=pd.Series(model.predict(X), index=df.index)
        )
        log.info(f'r2_score({df_name}): {score}')


if __name__ == '__main__':
    main()
