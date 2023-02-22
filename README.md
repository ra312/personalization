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

1. obtain sessions.csv and venues.csv and move them to the root folder
2. Check python --verrsion > 3.8.1 
3. Install personalization
```console
    python -m pip instal personalization
```
1. Train pipeline and get artifact,
 
   please run the following command in shell
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
For demo purposes, we choose to ingest sessiona and venues data locally and save model file locally. Given more time and infrastructure, I would add more things

1. Scalability: add Flyte workflow (reusing the code here)
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

[Read Latest Documentation](https://ra312.github.io/personalization/) - [Browse GitHub Code Repository](https://github.com/ra312/personalization/)
_________________
=======
# model server
```mermaid
---
title: REST-inference service
---
classDiagram
    note "100 requests per second"

    class VenueRating{
    """
    Represents the predicted ranking of a venue.

    Attributes:
    -----------
    venue_id : int The ID of the venue being rated.
    q80_predicted_rank : float
        The predicted ranking of the venue,
        as a 80-quantile of predicted rating
        for venue across available sessions
    """
    venue_id: int
    q80_predicted_rank: float
    }
    class TrainingPipeline{
      str pre-trained-model-file: stored with mlflow in gcs bucket
    }

    class InferenceFeatures{
    venue_id: int
    conversions_per_impression: float
    price_range: int
    rating: float
    popularity: float
    retention_rate: float
    session_id_hashed: int
    position_in_list: int
    is_from_order_again: int
    is_recommended: int
    }
    class FastAPIEndpoint{
      def predict_ratings(): Callabe
    }

    class Model_Instance{
        joblib.load(model_artifact_bucket)
        str model_artifact_bucket - variable
        str rank_column - fixed for the model
        str group_column - fixed for the model
    }
    TrainingPipeline --|> Model_Instance
    InferenceFeatures --|> FastAPIEndpoint
    Model_Instance --|> FastAPIEndpoint
    FastAPIEndpoint --|> VenueRating

```

[![PyPI](https://img.shields.io/pypi/v/model-server?style=flat-square)](https://pypi.python.org/pypi/model-server/)

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/model-server?style=flat-square)](https://pypi.python.org/pypi/model-server/)

[![PyPI - License](https://img.shields.io/pypi/l/model-server?style=flat-square)](https://pypi.python.org/pypi/model-server/)

[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)


---

**Documentation**: [https://ra312.github.io/model-server](https://ra312.github.io/model-server)
**Training Source Code**: [https://github.com/ra312/personalization](https://github.com/ra312/personalization)
**Source Code**: [https://github.com/ra312/model-server](https://github.com/ra312/model-server)
**PyPI**: [https://pypi.org/project/model-server/](https://pypi.org/project/model-server/)

---

A model server  for almost realtime inference

## Installation

```sh
pip install model-server
```

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

### Documentation

The documentation is automatically generated from the content of the [docs directory](./docs) and from the docstrings
 of the public signatures of the source code. The documentation is updated and published as a [Github project page
 ](https://pages.github.com/) automatically as part each release.

### Releasing

Trigger the [Draft release workflow](https://github.com/ra312/model-server/actions/workflows/draft_release.yml)
(press _Run workflow_). This will update the changelog & version and create a GitHub release which is in _Draft_ state.

Find the draft release from the
[GitHub releases](https://github.com/ra312/model-server/releases) and publish it. When
 a release is published, it'll trigger [release](https://github.com/ra312/model-server/blob/master/.github/workflows/release.yml) workflow which creates PyPI
 release and deploys updated documentation.

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
