import pytest

from personalization.abstract_pipeline import BaseMachineLearningPipeline


class DummyPipeline(BaseMachineLearningPipeline):
    """A test class to check exceptions are not raised
    if methods are overridden in the child class.
    """

    def prepare_datasets(self):
        """Override the required prepare_datasets method."""

    def train(self):
        """Override the required train method."""

    def __del__(self):
        """Override the required __del__ method."""


class TestDummyPipeline:
    """Test cases for the DummyPipeline class."""

    def test_prepare_datasets(self):
        pipeline = DummyPipeline()
        assert pipeline.prepare_datasets() is None


class TestBaseMachineLearningPipeline:
    """Test cases for the BaseMachineLearningPipeline class."""

    def test_init(self):
        with pytest.raises(TypeError):
            BaseMachineLearningPipeline()

    def test_set_model(self):
        pipeline = DummyPipeline()
        pipeline.model = "test_model"
        assert pipeline.model == "test_model"
