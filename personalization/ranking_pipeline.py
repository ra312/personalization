"""
This module defines a Pipeline for ranking sessions based on venue features.
"""
import gc
import logging
import os

import lightgbm as lgb
import polars as pl
from sklearn.model_selection import train_test_split

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
        # if venue_id is not Int64, then enforce it, so we can join

        self.sessions = pl.read_csv(sessions_bucket_path)
        self.ranking_data = pl.DataFrame()
        self.__validate__columns__()
        self.group_column = "session_id"
        self.rank_column = "rating"
        self.label_column = "has_seen_venue_in_this_session"
        self.features = [
            "venue_id",
            "conversions_per_impression",
            #  "purchased",
            "price_range",
            "rating",
            "popularity",
            "retention_rate",
            # 'session_id_hashed',
            "position_in_list",
            #  'has_seen_venue_in_this_session',
            #  'is_new_user',
            "is_from_order_again",
            "is_recommended",
        ]
        self.train_set = None
        self.val_set = None
        self.test_set = None

    def __validate__columns__(self) -> None:
        if "venue_id" not in self.venues.columns:
            raise ValueError(
                "Column 'venue_id' is not found in venues file"
            )
        if "venue_id" not in self.sessions.columns:
            raise ValueError(
                "Column 'venue_id' is not found in sessions file"
            )

    def __convert__boolean__to__int__(self) -> None:
        if self.ranking_data.is_empty():
            return
        bool_cols = self.ranking_data.select(pl.col(pl.Boolean)).columns
        if len(bool_cols) == 0:
            # no boolean columns to cast
            return
        self.ranking_data = self.ranking_data.with_columns(
            [
                pl.col(column).cast(pl.Int8, strict=False).alias(column)
                for column in bool_cols
            ]
        )

    def __drop__nulls__(self) -> None:
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

    def __join__sessions__and__venues__(self) -> None:
        self.ranking_data = self.sessions.join(
            self.venues, on="venue_id"
        )
        self.__convert__boolean__to__int__()
        # hex_string = "0a21dde9-1495-417c-bb9d-9922b81f2e6a"
        # self.ranking_data = self.ranking_data.with_columns(
        #     pl.col("session_id")
        #     .str.replace("-", "")
        #     .alias("session_id_hashed")
        #     .hash(seed=0)
        # )

    def prepare_datasets(self) -> None:
        self.__drop__nulls__()
        self.__join__sessions__and__venues__()
        self.ranking_data = pl.concat([self.ranking_data] * 100)
        train_set, unseen_set = train_test_split(
            self.ranking_data, train_size=0.2, test_size=0.8
        )

        val_set, test_set = train_test_split(
            unseen_set, train_size=0.2, test_size=0.8
        )
        group_column = self.group_column
        rank_column = self.rank_column
        label_column = self.label_column
        features = self.features

        train_set = train_set.sort(
            by=[group_column, rank_column], reverse=False
        )
        train_set_group_sizes = (
            train_set.groupby(group_column)
            .agg(pl.col(group_column).count().alias("count"))
            .sort(group_column)
            .select("count")
        )

        val_set = val_set.sort(
            by=[group_column, rank_column], reverse=False
        )
        val_set_group_sizes = (
            val_set.groupby(group_column)
            .agg(pl.col(group_column).count().alias("count"))
            .sort(group_column)
            .select("count")
        )

        train_y = train_set[[label_column]]
        train_x = train_set[features]

        val_y = val_set[[label_column]]
        val_x = val_set[features]

        # test_x = test_set[features]

        lgb_train_set = lgb.Dataset(
            train_x.to_pandas(),
            label=train_y.to_pandas(),
            group=train_set_group_sizes.to_numpy(),
            free_raw_data=True,
        ).construct()

        lgb_valid_set = lgb.Dataset(
            val_x.to_pandas(),
            label=val_y.to_pandas(),
            group=val_set_group_sizes.to_numpy(),
            reference=lgb_train_set,
            free_raw_data=True,
        ).construct()

        # some memory management
        # del train_set
        # del val_set
        del train_y
        del train_x

        gc.collect()

        self.train_set, self.val_set = lgb_train_set, lgb_valid_set

    def train(self) -> None:
        pass

    def __del__(self) -> None:
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
