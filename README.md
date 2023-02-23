personalization
_________________
An end-to-end demo machine learning pipeline to provide an artifact for a real-time inference service
# Aim
We want to create a machine learning training code which satisfies the following properties that given
data can train the model and save it to artifact
# Solution
Our implementation of the package 'personalization'
We choose to use Polars to read data, it is roughly 2-3 times faster than Pandas and supports nice API for 
aggregations and features creation.
For the model part, we decided to take lightGBM  due to ts speed, small size (model artifact size up to 50 Mb on 300 million rows of search data) and explainability. The user should choose lightGBM parameters carefully.
We tested an example lightgbm params in notebooks/train.ipynb.
# Offline evaluation
The offline evaluation has been done in notebooks/train.ipynb, we can see significant increase in NDCG levels across venues with our model against the baseline.
# CICD: code style and PyPI
The code is checked with pre-commit configs, tested and published in Github Actions, current coverage is around 80 percent.

The inference service code can be found here https://github.com/ra312/model-server
# How to run
1. Obtain sessions.csv and venues.csv and move them to the root folder
2. Install personalization
```console
    python -m pip instal personalization
```
3. Run the following command in shell to train pipeline and get artifact:
   
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

# TODO
Next steps:
1. Scalability(e.g. use Flyte)
2. Data: add support to ingest sessions and venues data from a database
3. Versioning: add MLFlow integration

[![PyPI version](https://badge.fury.io/py/personalization.svg)](http://badge.fury.io/py/personalization)
[![Test Status](https://github.com/ra312/personalization/workflows/Test/badge.svg?branch=develop)](https://github.com/ra312/personalization/actions?query=workflow%3ATest)
[![CI Status](https://github.com/ra312/personalization/workflows/Lint/badge.svg?branch=develop)](https://github.com/ra312/personalization/actions?query=workflow%3ALint)
[![codecov](https://codecov.io/gh/ra312/personalization/branch/main/graph/badge.svg)](https://codecov.io/gh/ra312/personalization)
[![Join the chat at https://gitter.im/ra312/personalization](https://badges.gitter.im/ra312/personalization.svg)](https://gitter.im/ra312/personalization?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://pypi.python.org/pypi/personalization/)
[![Downloads](https://pepy.tech/badge/personalization)](https://pepy.tech/project/personalization)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://timothycrosley.github.io/isort/)
[![CI](https://github.com/ra312/personalization/actions/workflows/action.yml/badge.svg)](https://github.com/ra312/personalization/actions/workflows/action.yml)
_________________

## Development

* Clone this repository
* Requirements:
  * [Poetry](https://python-poetry.org/)
  * Python 3.8.1+
* Create a virtual environment and install the dependencies

```sh
poetry install
```

* Activate the virtual environment

```sh
poetry shell
```

### Testing

```sh
pytest
```

### Pre-commit

Pre-commit hooks run all the auto-formatters (e.g. `black`, `isort`), linters (e.g. `mypy`, `flake8`), and other quality
 checks to make sure the changeset is in good shape before a commit/push happens.

You can install the hooks with (runs for each commit):

```sh
pre-commit install
```

Or if you want them to run only for each push:

```sh
pre-commit install -t pre-push
```

Or if you want e.g. want to run all checks manually for all files:

```sh
pre-commit run --all-files
```

---

This project was generated using the [wolt-python-package-cookiecutter](https://github.com/woltapp/wolt-python-package-cookiecutter) template.
