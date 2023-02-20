import logging
import pathlib
from typing import Any

import joblib


def delete_file_if_exists(file_path: str) -> None:
    """Delete a file if it exists.

    Args:
        file_path: The path to the file to delete.
    """
    # Convert the file path string to a Path object
    pathlib_instance = pathlib.Path(file_path)

    # Check if the file exists
    if pathlib_instance.exists():
        # If the file exists, delete it
        logging.info("File {file_path} exists, deleting it")
        pathlib_instance.unlink()


def check_file_location(file_path: str) -> bool:
    """Check if a file location exists and is a file.

    Args:
        file_path: The path to the file location to check.

    Returns:
        True if the file location exists and is a file, False otherwise.
    """
    # Convert the file path string to a Path object
    pathlib_instance = pathlib.Path(file_path)

    # Check if the file location exists and is a file

    return bool(pathlib_instance.is_file())


def load_model_from_artifact(model_artifact_bucket: str) -> Any:
    """we load it locally for demo only"""
    loaded_model = joblib.load(open(model_artifact_bucket, "rb"))
    return loaded_model
