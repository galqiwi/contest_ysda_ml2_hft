# My ydsa ml2 hft contest solution

This repo got me 3rd place.

[contest rules](https://github.com/SpectralTechnologies/contest_ysda/blob/main/trading.ipynb)

## Repo structure:
This code tries to be as SOLID as possible. Everything is modular and interchengable for easier experiment setup.
- [feature generator](feature_generator.py)
- [catboost model](hft_catboost_regressor.py)
- [lstm model](hft_ml_regressor.py)
- [universal model interface](hft_model.py)
