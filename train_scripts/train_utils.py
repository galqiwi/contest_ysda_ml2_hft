from utils import read_data

from sklearn.model_selection import train_test_split
from hft_model import test_model


def train_regressor(regressor, save_path, log):
    train, test = train_test_split(
        read_data(percentage=None, sample_from_start=True),
        test_size=0.05,
        shuffle=False
    )
    val, test = train_test_split(test, test_size=0.5, shuffle=False)

    regressor.fit(train, val)
    regressor.save(save_path)

    for df_name, df in [
        ('test', test),
        ('val', val),
    ]:
        log.info(f'r2_score({df_name}): {test_model(regressor, df)}')
