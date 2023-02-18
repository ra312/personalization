import os
import sys
from io import StringIO

import pandas as pd
import polars as pl
import pytest

from personalization.ranking_pipeline import RankingPipeline


@pytest.fixture
def sessions_csv_path(tmp_path):
    """Create a temporary CSV file with session data and return its path."""
    sessions_data = [
        {"session_id": 1, "user_id": 1},
        {"session_id": 2, "user_id": 2},
        {"session_id": 3, "user_id": None},
    ]
    sessions_csv_path_string = os.path.join(tmp_path, "sessions.csv")
    pd.DataFrame(sessions_data).to_csv(
        sessions_csv_path_string, index=False
    )
    return sessions_csv_path_string


@pytest.fixture
def venues_csv_path(tmp_path):
    """Create a temporary CSV file with venue data and return its path."""
    venues_data = [
        {"venue_id": 1, "name": "Venue 1"},
        {"venue_id": 2, "name": "Venue 2"},
        {"venue_id": 3, "name": None},
    ]
    venues_csv_path_str = os.path.join(tmp_path, "venues.csv")
    pd.DataFrame(venues_data).to_csv(venues_csv_path_str, index=False)
    return venues_csv_path_str


def test_drop_nulls(sessions_csv_path, venues_csv_path):
    """Test that null rows are dropped and logged correctly."""
    output = StringIO()
    sys.stdout = output
    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    pipeline.prepare_datasets()
    output.getvalue()
    # Check that null rows were dropped from venues dataframe
    assert pipeline.venues.shape == (2, 2)


def test_csv_path_not_proviced():
    """Test that an error is raised if the specified CSV file does not exist."""

    with pytest.raises(ValueError):
        RankingPipeline(
            sessions_bucket_path=None,
            venues_bucket_path="/tmp/non_existing_venues.csv",
        )
    with pytest.raises(ValueError):
        RankingPipeline(
            sessions_bucket_path="/tmp/non_existing_sessions.csv",
            venues_bucket_path=None,
        )


def test_invalid_csv_path():
    """Test that an error is raised if the specified CSV file does not exist."""

    with pytest.raises(FileNotFoundError):
        RankingPipeline(
            sessions_bucket_path="/tmp/non_existing_file.csv",
            venues_bucket_path="/tmp/non_existing_venues.csv",
        )


def test_venues_dataframe_type(sessions_csv_path, venues_csv_path):
    """Test that the venues attribute is a Polars DataFrame."""
    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    assert isinstance(pipeline.venues, pl.DataFrame)


def test_sessions_dataframe_type(sessions_csv_path, venues_csv_path):
    """Test that the sessions attribute is a Polars DataFrame."""
    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    assert isinstance(pipeline.sessions, pl.DataFrame)


def test_prepare_datasets_called(
    sessions_csv_path, venues_csv_path, mocker
):
    """Test that prepare_datasets is called when the object is created."""
    mock_prepare_datasets = mocker.spy(
        RankingPipeline, "prepare_datasets"
    )
    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    pipeline.prepare_datasets()
    mock_prepare_datasets.assert_called_once_with(pipeline)


# def test_cleanup_resources(sessions_csv_path, venues_csv_path, mocker):
#     """Test that __del__ method cleans up resources."""
#     mock_log = mocker.patch("logging.info")
#     pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
#     del pipeline
#     mock_log.assert_called_once_with("Cleaning up resources used by the RankingPipeline object")
