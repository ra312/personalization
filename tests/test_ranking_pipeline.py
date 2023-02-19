import os

import pandas as pd
import polars as pl
import pytest

from personalization.ranking_pipeline import RankingPipeline


from .utils import (
    generate_sessions_dataframe,
    generate_venues_dataframe,
)


@pytest.fixture
def sessions_csv_path(tmp_path):
    """Create a temporary CSV file with session data and return its path."""

    sessions_csv_path_string = os.path.join(tmp_path, "sessions.csv")
    generate_sessions_dataframe().write_csv(sessions_csv_path_string)
    return sessions_csv_path_string


@pytest.fixture
def venues_csv_path(tmp_path):
    """Create a temporary CSV file with venue data and return its path."""
    # venues_data = {"venue_id": [1, 2, 3], "name": ["Venue 1", "Venue 2", None]}
    venues_csv_path_str = os.path.join(tmp_path, "venues.csv")
    generate_venues_dataframe().write_csv(venues_csv_path_str)
    return venues_csv_path_str


def test_drop_nulls(sessions_csv_path, venues_csv_path):
    """Test that null rows are dropped and logged correctly."""

    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    pipeline.__drop__nulls__()
    # pipeline.prepare_datasets()

    # Check that null rows were dropped from venues dataframe
    assert pipeline.venues.shape == (9, 6)
    assert pipeline.sessions.shape == (9, 8)


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


def test_validate_columns(sessions_csv_path, venues_csv_path):
    """Test that an error is raised if a required column is missing."""

    with pytest.raises(
        ValueError,
        match="Column 'venue_id' is not found in sessions file",
    ):
        sessions_data = pd.read_csv(sessions_csv_path)
        # Remove 'venue_id' column from sessions data
        sessions_data = sessions_data.drop(columns=["venue_id"])
        sessions_data.to_csv(sessions_csv_path, index=False)
        RankingPipeline(sessions_csv_path, venues_csv_path)

    with pytest.raises(
        ValueError,
        match="Column 'venue_id' is not found in venues file",
    ):
        venues_data = pd.read_csv(venues_csv_path)
        venues_data = venues_data.drop(columns=["venue_id"])
        venues_data.to_csv(venues_csv_path, index=False)
        RankingPipeline(sessions_csv_path, venues_csv_path)


def test_join_sessions_and_venues_no_data_loss(
    sessions_csv_path, venues_csv_path
):
    """Test that no rows are lost after joining sessions and venues data."""
    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    # we extract row count from class attribute since we drop null columns
    sessions_count = pipeline.sessions.shape[0]
    venues_count = pipeline.venues.shape[0]
    pipeline.__join__sessions__and__venues__()
    # Check that the number of rows
    #  is the same as
    # the sum of the number of rows
    # in the two input files
    expected_num_rows = min(sessions_count, venues_count)
    assert pipeline.ranking_data.shape[0] == expected_num_rows


def test_join_sessions_and_venues_no_duplicate_columns(
    sessions_csv_path, venues_csv_path
):
    """Test that no duplicate columns are created after joining sessions and venues data."""
    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    pipeline.__join__sessions__and__venues__()
    # Check that no column names appear more than once in the output dataframe
    assert set(pipeline.ranking_data.columns) == set(
        pipeline.sessions.columns + pipeline.venues.columns
    )
