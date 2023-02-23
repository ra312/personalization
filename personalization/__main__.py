import argparse

from .ranking_pipeline import RankingPipeline


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments and return an `argparse.Namespace` object.

    Returns:
        argparse.Namespace: An object containing the parsed command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Description of your program"
    )
    parser.add_argument(
        "--sessions-bucket-path",
        type=str,
        required=True,
        help="Path to sessions file",
    )
    parser.add_argument(
        "--venues-bucket-path",
        type=str,
        required=True,
        help="Path to venues file",
    )
    parser.add_argument(
        "--objective",
        type=str,
        default="lambdarank",
        help="LightGBM objective",
    )
    parser.add_argument(
        "--num_leaves",
        type=int,
        default=100,
        help="Number of leaves in LightGBM model",
    )
    parser.add_argument(
        "--min_sum_hessian_in_leaf",
        type=int,
        default=10,
        help="Minimum sum of hessian in one leaf",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ndcg",
        help="Metric for LightGBM evaluation",
    )
    parser.add_argument(
        "--ndcg_eval_at",
        type=int,
        nargs="+",
        default=[10, 20],
        help="Evaluation position for NDCG metric",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.8,
        help="Learning rate for LightGBM model",
    )
    parser.add_argument(
        "--force_row_wise",
        type=bool,
        default=True,
        help="Whether to process data row-wise",
    )
    parser.add_argument(
        "--num_iterations",
        type=int,
        default=10,
        help="Number of boosting iterations",
    )
    parser.add_argument(
        "--trained-model-path",
        type=str,
        help="path to save the trained model",
    )

    args = parser.parse_args()

    # Parse arguments
    return args


def train_and_export(parsed_args: argparse.Namespace):
    lgbm_params = {
        "objective": parsed_args.objective,
        "num_leaves": parsed_args.num_leaves,
        "min_sum_hessian_in_leaf": parsed_args.min_sum_hessian_in_leaf,
        "metric": parsed_args.metric,
        "ndcg_eval_at": parsed_args.ndcg_eval_at,
        "learning_rate": parsed_args.learning_rate,
        "force_row_wise": parsed_args.force_row_wise,
        "num_iterations": parsed_args.num_iterations,
    }
    print(f"parsed_args={parsed_args}")
    pipeline = RankingPipeline(
        sessions_bucket_path=parsed_args.sessions_bucket_path,
        venues_bucket_path=parsed_args.venues_bucket_path,
    )
    pipeline.prepare_datasets()

    pipeline.train(params=lgbm_params)
    pipeline.export_model_artifact(
        model_path=parsed_args.trained_model_path
    )


if __name__ == "__main__":
    parsed_args = parse_arguments()
    train_and_export(parsed_args)
