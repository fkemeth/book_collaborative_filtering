import pandas as pd

import multiprocessing as mp

from sklearn.model_selection import KFold, train_test_split

from .collaborative_filter import CollaborativeFilter


class Evaluator:
    def __init__(self, ratings: pd.DataFrame, params: dict) -> None:
        self.params = params

        self.ratings = ratings
        self.user_ids = ratings.user_id.unique()
        self.num_users = len(self.user_ids)

        self.kf = KFold(n_splits=self.num_users, shuffle=True, random_state=42)

        self.metrics = {}
        self.metrics["coverage"] = []
        self.metrics["mae"] = []

    @staticmethod
    def log_results(metrics: dict, coverage: float, mae: float) -> None:
        metrics["coverage"].append(coverage)
        metrics["mae"].append(mae)

    def evaluate(self, train_index, test_index) -> None:
        train_user_ids, test_user_ids = (
            self.user_ids[train_index],
            self.user_ids[test_index],
        )
        train_ratings, test_ratings = (
            self.ratings[self.ratings.user_id.isin(train_user_ids)],
            self.ratings[self.ratings.user_id.isin(test_user_ids)],
        )

        input_ratings, heldout_ratings = train_test_split(
            test_ratings, stratify=test_ratings.user_id, test_size=0.1, random_state=42
        )

        cf = CollaborativeFilter(
            train_ratings,
            user_col="user_id",
            item_col="item_id",
            neighborhood_method=self.params["neighborhood_method"],
            correlation_method=self.params["correlation_method"],
            minimal_similarity=self.params["minimal_similarity"],
            number_of_neighbors=self.params["number_of_neighbors"],
            minimum_number_of_items_rated_in_common=self.params[
                "minimum_number_of_items_rated_in_common"
            ],
            minimal_number_of_ratings=self.params["minimal_number_of_ratings"],
            deviation_from_mean=self.params["deviation_from_mean"],
        )

        similarities = cf.get_similarities(input_ratings)
        predicted_scores = cf.get_scores(similarities, input_ratings)

        predictions = heldout_ratings.merge(
            predicted_scores.rename("scores"), on="item_id", how="left"
        )
        coverage = 1 - predictions.scores.isna().sum() / len(predictions)
        mae = (predictions.rating - predictions.scores).abs().mean()
        Evaluator.log_results(self.metrics, coverage, mae)
        return coverage, mae

    def run(self, number_of_runs: int = 1) -> None:
        for i, (train_index, test_index) in enumerate(self.kf.split(self.user_ids)):
            self.evaluate(train_index=train_index, test_index=test_index)

            if i == number_of_runs - 1:
                return True

    def run_parallel(self, number_of_runs: int = 1) -> None:
        metrics = {"coverage": [], "mae": []}
        pool = mp.Pool(mp.cpu_count() - 1)
        for train_index, test_index in list(self.kf.split(self.user_ids))[
            :number_of_runs
        ]:
            pool.apply_async(
                self.evaluate,
                args=(train_index, test_index),
                callback=lambda x: Evaluator.log_results(metrics, *x),
            )
        pool.close()
        pool.join()
        self.metrics = metrics
