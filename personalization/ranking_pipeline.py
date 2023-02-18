"""
This module defines a Pipeline for ranking sessions based on venue features.
"""
import logging
import os

import polars as pl

from .abstract_pipeline import BaseMachineLearningPipeline


class RankingPipeline(BaseMachineLearningPipeline):
    """
    Pipeline for ranking sessions based on venue features.

    Attributes
    ----------
    venues : pl.DataFrame
        DataFrame with information about venues.
    sessions : pl.DataFrame
        DataFrame with information about sessions.

    Parameters
    ----------
    sessions_bucket_path : str
        Path to the CSV file containing the sessions data.
    venues_bucket_path : str
        Path to the CSV file containing the venues data.
    """

    def __init__(
        self, sessions_bucket_path: str, venues_bucket_path: str
    ):
        """
        Initialize the RankingPipeline object.

        Parameters
        ----------
        sessions_bucket_path : str
            Path to the CSV file containing the sessions data.
        venues_bucket_path : str
            Path to the CSV file containing the venues data.
        """
        super().__init__()
        if not sessions_bucket_path or not venues_bucket_path:
            raise ValueError(
                "Either sessions path or venues path is not provided"
            )
        if not os.path.isfile(
            sessions_bucket_path
        ) or not os.path.isfile(venues_bucket_path):
            raise FileNotFoundError(
                f"File {venues_bucket_path} or {sessions_bucket_path} does not exist."
            )
        self.venues = pl.read_csv(venues_bucket_path)
        self.sessions = pl.read_csv(sessions_bucket_path)

    def __drop__nulls__(self):
        """
        Drop rows with missing values from the venues DataFrame.

        Returns
        -------
        None
        """
        init_rows_cnt = self.venues.shape[0]
        self.venues = self.venues.drop_nulls()
        frac_dropped = (
            (self.venues.shape[0] - init_rows_cnt) / init_rows_cnt * 100
        )
        logging.info(
            "There are %s percentage of rows with at least one null value",
            frac_dropped,
        )
        logging.info("dropping them ..")

    def prepare_datasets(self):
        self.__drop__nulls__()

    def train(self):
        pass

    def __del__(self):
        """
        Clean up any resources used by the RankingPipeline object.
        """
        # close any open file handles or connections here
        logging.info(
            "Cleaning up resources used by the RankingPipeline object"
        )
        if hasattr(self, "venues"):
            logging.info(
                "Cleaning up resources used by the RankingPipeline object"
            )
            del self.venues
