personalization
_________________
An end-to-end demo machine learning pipeline to provide an artifact for a real-time inference service

Requirements: we want to create a machine learning pipeline which satisfies the following properties that given
data can train the model and save it to artifact. 
Our implementation of the package 'personalization'
We choose to use Polars to read data, it is roughly 2-3 times faster than Pandas and supports nice API for 
aggregations and features creation.
For the model part, we decided to take lightGBM  due to ts speed, small size (model artifact size up to 50 Mb on 300 million rows of search data) and explainability. The user should choose lightGBM parameters carefully.
We tested an example lightgbm params in notebooks/train.ipynb.
The offline evaluation has been done in notebooks/train.ipynb, we can see significant increase in NDCG levels across venues with our model against the baseline.
The code is tested in Github Actions, current coverage is around 80 percent.
The inference service code can be found here https://github.com/ra312/model-server
# How to run

1. obttain sessions.csv and venues.csv and move them to the root folder
2. Check python --verrsion > 3.8.1 
3. Install personalization
```console
    python -m pip instal personalization
```
1. Train pipeline and get artifact, copy this into bash
```console
python3 -m personalization \
    --sessions-bucket-path sessions.csv \
    --venues-bucket-path venues.csv \
    --objective lambdarank \
    --num_leaves 100 \
    --min_sum_hessian_in_leaf 10 \
    --metric ndcg --ndcg_eval_at 10 20 \
    --learning_rate 0.8 \
    --force_row_wise True \
    --num_iterations 10 \
    --trained-model-path trained_model.joblib
```
#TODO:
For demo purposes, we choose to ingest sessiona and venues data locally and save model file locally. Given more time and infrastructure, I would add more things
#TODO:
1. Scalability: add Flyte workflow (reusing the code here)
2. Data: add support to ingest sessions and venues data from a database
3. Versioning: add MLFlow integration

[![PyPI version](https://badge.fury.io/py/personalization.svg)](http://badge.fury.io/py/personalization)
[![Test Status](https://github.com/ra312/personalization/workflows/Test/badge.svg?branch=develop)](https://github.com/ra312/personalization/actions?query=workflow%3ATest)
[![Lint Status](https://github.com/ra312/personalization/workflows/Lint/badge.svg?branch=develop)](https://github.com/ra312/personalization/actions?query=workflow%3ALint)
[![codecov](https://codecov.io/gh/ra312/personalization/branch/main/graph/badge.svg)](https://codecov.io/gh/ra312/personalization)
[![Join the chat at https://gitter.im/ra312/personalization](https://badges.gitter.im/ra312/personalization.svg)](https://gitter.im/ra312/personalization?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.python.org/pypi/personalization/)
[![Downloads](https://pepy.tech/badge/personalization)](https://pepy.tech/project/personalization)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![CI](https://github.com/ra312/personalization/actions/workflows/action.yml/badge.svg)](https://github.com/ra312/personalization/actions/workflows/action.yml)
_________________

[Read Latest Documentation](https://ra312.github.io/personalization/) - [Browse GitHub Code Repository](https://github.com/ra312/personalization/)
_________________

**personalization** An endpoint service to provide real-time personalization
