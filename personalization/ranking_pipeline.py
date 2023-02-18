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
        # EXPLAIN: we assume that we can fit datasets in memory, i.e.
        # either data volume is moderate or we are inside a high-mem instance
        self.venues = pl.read_csv(venues_bucket_path)
        self.sessions = pl.read_csv(sessions_bucket_path)
        self.ranking_data = pl.DataFrame()
        self.__validate__columns__()

    def __validate__columns__(self):
        if "venue_id" not in self.venues.columns:
            raise ValueError(
                "Column 'venue_id' is not found in venues file"
            )
        if "venue_id" not in self.sessions.columns:
            raise ValueError(
                "Column 'venue_id' is not found in sessions file"
            )

    def __convert__boolean__to__int__(self):
        if not self.ranking_data:
            return
        bool_cols = self.ranking_data.select(pl.col(pl.Boolean)).columns
        self.ranking_data = self.ranking_data.with_columns(
            [
                pl.col(column).cast(pl.Int8, strict=False).alias(column)
                for column in bool_cols
            ]
        )

    def __drop__nulls__(self):
        """
        Drop rows with missing values from the venues DataFrame.

        Returns
        -------
        None
        """
        init_rows_cnt = self.venues.shape[0]
        self.venues = self.venues.drop_nulls()
        self.sessions = self.sessions.drop_nulls()
        frac_dropped = (
            (self.venues.shape[0] - init_rows_cnt) / init_rows_cnt * 100
        )
        logging.info(
            "There are %s percentage of rows with at least one null value",
            frac_dropped,
        )
        logging.info("dropping them ..")

    def __join__sessions__and__venues__(self):

        self.ranking_data: pl.DataFrame = self.sessions.join(
            self.venues, on="venue_id"
        )
        self.__convert__boolean__to__int__()
        # hex_string = "0a21dde9-1495-417c-bb9d-9922b81f2e6a"
        self.ranking_data = self.ranking_data.with_columns(
            pl.col("session_id")
            .str.replace("-", "")
            .alias("session_id_hashed")
            .hash(seed=0)
        )

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
