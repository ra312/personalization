import pytest

from personalization.abstract_pipeline import (
    BaseMachineLearningPipeline,
)


class DummyPipeline(BaseMachineLearningPipeline):
    """a test class to check exceptions are raised
        if methods are ont overridden in the child class

    Args:
        BaseMachineLearningPipeline (_type_): _description_
    """

    def test_method(self):
        """
        do not override the required methods
        """


def test_pipeline_abstract_methods():
    with pytest.raises(TypeError):
        DummyPipeline()
