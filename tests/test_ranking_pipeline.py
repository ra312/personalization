import os

import pandas as pd
import polars as pl
import pytest

from personalization.ranking_pipeline import RankingPipeline

from .utils import (
    generate_sessions_dataframe,
    generate_venues_dataframe,
)
from personalization import __DEFAULT__LGB__PARAMS__


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


def test_save_datasets(sessions_csv_path, venues_csv_path):
    # Set up the test
    train_data_path = "/tmp/train_data.binary"
    val_data_path = "/tmp/val_data.binary"
    pipeline = RankingPipeline(
        sessions_csv_path,
        venues_csv_path,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
    )
    pipeline.prepare_datasets()
    # Call the method being tested
    pipeline.__save__datasets__()

    # Check that the binary files were created
    assert os.path.exists(train_data_path)
    assert os.path.exists(val_data_path)


def test_train_fails_if_invalid(
    sessions_csv_path, venues_csv_path
) -> None:
    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    with pytest.raises(ValueError):
        # case when train_set is empty
        lgb_params = {
            "objective": "lambdarank",
            "num_leaves": 100,
            "min_sum_hessian_in_leaf": 10,
            "metric": "ndcg",
            "ndcg_eval_at": [10, 20, 40],
            "learning_rate": 0.8,
            "force_row_wise": True,
            "num_iterations": 2,
        }
        pipeline.train(params=lgb_params)
    with pytest.raises(ValueError):
        # case when train_set is empty
        pipeline.train({})


def test_train_succeeds(sessions_csv_path, venues_csv_path) -> None:
    pipeline = RankingPipeline(sessions_csv_path, venues_csv_path)
    pipeline.prepare_datasets()
    lgb_params = {
        "objective": "lambdarank",
        "num_leaves": 100,
        "min_sum_hessian_in_leaf": 10,
        "metric": "ndcg",
        "ndcg_eval_at": [10, 20, 40],
        "learning_rate": 0.8,
        "force_row_wise": True,
        "num_iterations": 2,
    }

    try:
        pipeline.train(params=lgb_params)
    except Exception as e:
        pytest.fail(f"Failed with unexpected error: {e}")
