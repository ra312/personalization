from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
    Optional,
)


class BaseMachineLearningPipeline(ABC):
    """
    An abstract base class for a machine learning pipeline.

    Attributes:
        model: The machine learning model used by the pipeline.

    Methods:
        __init__(): Constructor method for the machine learning pipeline.
        prepare_datasets(): prepares the data for model
        train(): Abstract method that trains the machine learning model.
        model(): Getter method for the machine learning model.
        model(): Setter method for the machine learning model.
        __del__(): release memory occupied by the pipeline
    """

    def __init__(self, **kwargs: Any) -> None:
        """Constructor method for the machine learning pipeline."""
        self._model: Optional[Any] = None

    @abstractmethod
    def prepare_datasets(self) -> None:
        """Abstract method that prepares the data for training the machine learning model."""

    @abstractmethod
    def train(self) -> None:
        """Abstract method that trains the machine learning model."""

    @property
    def model(self) -> Optional[Any]:
        """Getter method for the machine learning model."""
        return self._model

    @model.setter
    def model(self, value: Any) -> None:
        """Setter method for the machine learning model."""
        self._model = value

    def __del__(self) -> None:  # noqa: B027
        """Destructor method for the machine learning pipeline
        that releases memory occupied by class instance."""
