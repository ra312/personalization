personalization
_________________
An end-to-end demo machine learning pipeline to provide an artifact for a real-time inference service

Requirements: we want to create a machine learning pipeline which satisfies the following properties

    1. Multiple Models Support: The code should support maintaining 
    wide range of machine learning algorithms,
    linear regression, decision trees, random forests,
    and deep learning models, to meet diverse business requirements.
    2. Configurability: The API should be highly configurable to
        allow users to customize
        the machine learning models to their specific use cases.
        This may include hyperparameter tuning, feature selection, and feature engineering.
    3. Flexibility: The API should be flexible enough to handle a wide range of data formats,
    such as CSV, JSON, and Parquet. It should also support various
    deployment environments, such as on-premises, cloud-based, and hybrid environments.
    4. Scalability: The API should be designed with scalability in mind,
    meaning it can handle large volumes of data, high request rates, and multiple concurrent users.
    This may involve incorporating distributed computing
    and parallel processing techniques to handle the workload.
    5. Support versioning with MLFlow
    6. Documentation: The API should be accompanied by comprehensive documentation,
    including user manuals, API reference guides, and developer documentation.
    This will make it easier for users to learn
    how to use the API and integrate it into their applications.
# How to run

1. obttain sessions.csv and venues.csv and move them to the root folder
2. Check python --verrsion > 3.8.1 
3. Install personalization
```console
    python -m pip instal personalization
```
4. Train pipeline and get artifact, copy this into bash
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
