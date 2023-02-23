"""
pipeline to train personalization model
"""
__version__ = "0.0.1"


from .file_utils import load_model_from_artifact
from .ranking_pipeline import RankingPipeline

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


__all__ = [
    "RankingPipeline",
    "load_model_from_artifact",
    "__DEFAULT__LGB__PARAMS__",
]
