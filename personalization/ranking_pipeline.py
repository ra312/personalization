"""
This module defines a Pipeline for ranking sessions based on venue features.
"""
import gc
import logging
import os
import pathlib
from typing import (
    Any,
    Dict,
    Optional,
)

import lightgbm as lgb
import polars as pl
from sklearn.model_selection import train_test_split

from .abstract_pipeline import BaseMachineLearningPipeline
from .file_utils import (
    check_file_location,
    delete_file_if_exists,
    save_model_to_file,
)

__DEFAULT__LGB__PARAMS__ = {
    "objective": "lambdarank",
    "num_leaves": 100,
    "min_sum_hessian_in_leaf": 10,
    "metric": "ndcg",
    "ndcg_eval_at": [10, 20, 40],
    "learning_rate": 0.8,
    "force_row_wise": True,
    "num_iterations": 10,
}


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
        self,
        sessions_bucket_path: str,
        venues_bucket_path: str,
        **kwargs: str,
    ) -> None:
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
        self.venues: pl.DataFrame = pl.read_csv(venues_bucket_path)
        # if venue_id is not Int64, then enforce it, so we can join

        self.sessions: pl.DataFrame = pl.read_csv(sessions_bucket_path)
        self.ranking_data: pl.DataFrame = pl.DataFrame()
        self.__validate__columns__()
        self.group_column: str = "session_id"
        self.rank_column: str = "rating"
        self.label_column: str = "has_seen_venue_in_this_session"
        self.features = [
            "venue_id",
            "conversions_per_impression",
            "price_range",
            "rating",
            "popularity",
            "retention_rate",
            "position_in_list",
            "is_from_order_again",
            "is_recommended",
        ]
        self.train_set: lgb.Dataset = lgb.Dataset(data=[])  # type: ignore[no-any-unimported]
        self.val_set: lgb.Dataset = lgb.Dataset(data=[])  # type: ignore[no-any-unimported]
        # read train_data_path parameter if provided
        self.train_data_path: str = kwargs.get(
            "train_data_path", "/tmp/train_set.binary"
        )
        self.val_data_path = kwargs.get(
            "val_data_path", "/tmp/val_set.binary"
        )
        self.n_features = len(self.features)
        delete_file_if_exists(self.train_data_path)
        delete_file_if_exists(self.val_data_path)

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

    def __save__datasets__(self) -> None:
        if not hasattr(self, "train_set") or self.train_set is None:
            raise Exception("No attribute 'train_set' found")
        if isinstance(self.train_set, pl.DataFrame):
            raise ValueError("self.train_set is not Polars dataframe")
        self.train_set.save_binary(self.train_data_path)
        if not hasattr(self, "val_set") or self.val_set is None:
            raise Exception("No attribute 'val_set' found")
        if isinstance(self.val_set, pl.DataFrame):
            raise ValueError("self.val_set is not Polars dataframe")
        self.val_set.save_binary(self.val_data_path)

    def prepare_datasets(self) -> None:
        self.__drop__nulls__()
        self.__join__sessions__and__venues__()
        self.ranking_data = pl.concat([self.ranking_data] * 100)
        train_set, unseen_set = train_test_split(
            self.ranking_data, train_size=0.2, test_size=0.8
        )

        val_set, _ = train_test_split(
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

        lgb_train_set: Any = lgb.Dataset(
            train_x.to_pandas(),
            label=train_y.to_pandas(),
            group=train_set_group_sizes.to_numpy(),
            free_raw_data=True,
        ).construct()

        lgb_valid_set: Any = lgb.Dataset(
            val_x.to_pandas(),
            label=val_y.to_pandas(),
            group=val_set_group_sizes.to_numpy(),
            reference=lgb_train_set,
            free_raw_data=True,
        ).construct()

        del train_y
        del train_x

        gc.collect()

        self.train_set = lgb_train_set

        self.val_set = lgb_valid_set
        self.__save__datasets__()

    def __load__datasets__(self) -> None:
        if check_file_location(self.train_data_path) is False:
            raise ValueError(f"No train file found at {self.train}")
        with pathlib.Path(self.train_data_path) as train_data_pathlib:
            self.train_set = lgb.Dataset(train_data_pathlib)

        if check_file_location(self.val_data_path) is False:
            raise ValueError(f"No val file found at {self.train}")

        with pathlib.Path(self.val_data_path) as val_data_pathlib:
            self.val_set = lgb.Dataset(val_data_pathlib)

    def train(self, params: Optional[Any]) -> None:
        # EXPLAIN: due to mypy nagging typing from base class

        if len(params) == 0:  # type: ignore[arg-type]
            params = __DEFAULT__LGB__PARAMS__
            logging.info("Empty lightgbm params are passed")
            logging.info("#" * 10)
            logging.info(" ... using default params")
            params = params.update(__DEFAULT__LGB__PARAMS__)
            logging.info(params)
            logging.info("#" * 10)
        if not isinstance(params, dict):
            raise ValueError(
                "params parameter is expected to be of type dict"
            )
        evals_logs: Dict[Any, Any] = {}
        # check dataset exists and not empty
        if not hasattr(self, "train_set"):
            raise ValueError("no attribute train_set")
        if self.train_set is None:
            raise ValueError("train_set attribute is empty")
        if self.train_set.num_data() == 0:
            logging.info(
                "train_set not found in memory, loading from bucket"
            )
            try:
                self.__load__datasets__()
            except ValueError:
                logging.info("Failed to load file from the location")
        if self.train_set.num_feature() != self.n_features:
            raise ValueError(
                "Some of the features were lost during preprocessing"
            )
        lgb_train_set = self.train_set
        lgb_valid_set = self.val_set
        self.model = lgb.train(
            params=params,
            train_set=lgb_train_set,
            valid_sets=[lgb_valid_set, lgb_train_set],
            valid_names=["val", "train"],
            verbose_eval=25,
            evals_result=evals_logs,
            early_stopping_rounds=25,
        )

    def export_model_artifact(self, model_path: str)-> None:
        save_model_to_file(
            traine_model=self.model, model_path=model_path
        )
        # TODO: add MLFlow integration and gcs integration

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
