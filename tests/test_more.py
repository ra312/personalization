import pytest

from personalization import BaseMachineLearningPipeline


class DummyPipeline(BaseMachineLearningPipeline):
    """A test class to check exceptions are raised
    if methods are not overridden in the child class.
    """

    def prepare_datasets(self):
        """Override the required prepare_datasets method."""

    def train(self):
        """Override the required train method."""


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
        class ConcretePipeline(BaseMachineLearningPipeline):
            """
            A concrete implementation of a machine learning pipeline.
            """

            def prepare_datasets(self):
                """
                Prepare the data for training the machine learning model.
                """
                # TODO: implement prepare_datasets method
                return "Datasets prepared"

            def train(self):
                """
                Train the machine learning model.
                """
                # TODO: implement train method
                return "Model trained"

        pipeline = ConcretePipeline()
        pipeline.model = "test_model"
        assert pipeline.model == "test_model"
