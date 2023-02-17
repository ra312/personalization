"""**personalization**

An endpoint service to provide real-time personalization
Requirements: we want to create a machine learning pipeline which satisfies the following properties
    1. Multiple Models Support: The API should support a
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
    5. Documentation: The API should be accompanied by comprehensive documentation,
    including user manuals, API reference guides, and developer documentation.
    This will make it easier for users to learn
    how to use the API and integrate it into their applications.

"""
__version__ = "0.0.1"

__all__ = "BaseMachineLearningPipeline"
