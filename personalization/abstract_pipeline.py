"""
A training pipeline class
"""

from abc import (
    ABC,
    abstractmethod,
)


class BaseMachineLearningPipeline(ABC):
    """
    An abstract base class for a machine learning pipeline.

    Attributes:
        model: The machine learning model used by the pipeline.

    Methods:
        prepare_datasets(): Abstract method
        that prepares the data for training the machine learning model.
        train(): Abstract method that trains the machine learning model.
    """

    @property
    def model(self):
        """Getter method for the machine learning model."""
        return self._model

    @model.setter
    def model(self, value):
        """Setter method for the machine learning model."""
        self._model = value

    def __init__(self, **kwargs):
        """Constructor method for the machine learning pipeline."""
        self._model = None

    @abstractmethod
    def prepare_datasets(self):
        """Abstract method that prepares the data for training the machine learning model."""

    @abstractmethod
    def train(self):
        """Abstract method that trains the machine learning model."""

    @abstractmethod
    def __del__(self):
        """Destructor method for the machine learning pipeline
        that releases memory occupied by class instance."""
