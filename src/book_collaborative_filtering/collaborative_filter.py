import numpy as np
import pandas as pd
from pandas.core import nanops


class CollaborativeFilter:
    def __init__(
        self,
        ratings: pd.DataFrame,
        user_col: str = "user_id",
        item_col: str = "book_id",
        neighborhood_method: str = "threshold",
        correlation_method: str = "pearson",
        minimal_similarity: float = 0.7,
        number_of_neighbors: int = 50,
        minimum_number_of_books_rated_in_common: int = 10,
        minimal_number_of_ratings: int = 5,
        deviation_from_mean: bool = True,
    ) -> None:
        self.ratings = ratings
        self.user_col = user_col
        self.item_col = item_col

        self.user_means = (
            ratings[[self.user_col, "rating"]].groupby(self.user_col).agg("mean")
        )

        self.neighborhood_method = neighborhood_method
        self.correlation_method = correlation_method
        self.minimal_similarity = minimal_similarity
        self.number_of_neighbors = number_of_neighbors
        self.minimum_number_of_books_rated_in_common = (
            minimum_number_of_books_rated_in_common
        )
        self.minimal_number_of_ratings = minimal_number_of_ratings
        self.deviation_from_mean = deviation_from_mean

    @staticmethod
    def calculate_raw_similarities(
        ratings_data: pd.DataFrame,
        input_ratings: pd.DataFrame,
        minimum_number_of_books_rated_in_common: int,
        correlation_method: str = "pearson",
        user_col: str = "user_id",
        item_col: str = "book_id",
        remove_self_similarity: bool=True,
        remove_nan_similarities: bool=True
    ) -> pd.Series:
        assert (
            len(input_ratings[user_col].unique()) == 1
        ), "Input ratings should belong to one user only"
        assert correlation_method in [
            "pearson",
            "spearman",
        ], "Only pearson and spearman correlation methods supported"
        user_id = input_ratings[user_col].unique()[0]
        ratings = pd.concat([ratings_data, input_ratings])

        # Restrict ratings to user-rated items only
        relevant_ratings = pd.merge(
            ratings, input_ratings[item_col], on=[item_col], how="inner"
        )
        uii_matrix = relevant_ratings.pivot_table(
            index=[user_col], columns=[item_col], values="rating"
        )

        similarities = uii_matrix.corrwith(
            uii_matrix.loc[user_id],
            axis=1,
            method=lambda x, y: nanops.nancorr(
                x,
                y,
                min_periods=minimum_number_of_books_rated_in_common,
                method=correlation_method,
            ),
        )
        similarities.name = "similarities"

        # Remove self similarity
        if remove_self_similarity:
            similarities[user_id] = np.nan
        # Remove nans
        if remove_nan_similarities:
            similarities = similarities.dropna()
        return similarities

    @staticmethod
    def select_neihgborhood(
        similarities: pd.Series,
        neighborhood_method: str,
        minimal_similarity: float = 0.5,
        number_of_neighbors: int = 50,
    ) -> pd.Series:
        # Consider those users with at least a similarity of minimal_similarity
        assert neighborhood_method in [
            "threshold",
            "number",
            None
        ], "Only 'threshold', 'number' or None neighborhood methods supported"

        if neighborhood_method == "threshold":
            return similarities[similarities > minimal_similarity]
        elif neighborhood_method == "number":
            return similarities.nlargest(n=number_of_neighbors)

    def get_similarities(self, input_ratings: pd.DataFrame) -> pd.DataFrame:
        # Calculate similarities between input ratings and ratings data
        similarities = CollaborativeFilter.calculate_raw_similarities(
            self.ratings,
            input_ratings,
            self.minimum_number_of_books_rated_in_common,
            self.correlation_method,
            self.user_col,
            self.item_col,
            remove_self_similarity=True,
            remove_nan_similarities=True
        )

        # Select neighborhood
        similarities = CollaborativeFilter.select_neihgborhood(
            similarities,
            self.neighborhood_method,
            self.minimal_similarity,
            self.number_of_neighbors,
        )
        return similarities

    @staticmethod
    def calculate_score(
        self,
        column,
        similarities: pd.Series,
        input_mean: float = 0.0,
        user_means: pd.Series = None,
        minimal_number_of_ratings: int = 1,
        deviation_from_mean: bool = False,
    ) -> float:
        similarities = similarities[column.notna()]
        column = column[column.notna()]
        # If book has been rated less than minimal_number_of_ratings, set its score to nan
        if len(column) < minimal_number_of_ratings:
            return np.nan

        # Calculate weighted mean of ratings as scores
        denominator = similarities.sum()
        if denominator == 0:
            return np.nan

        if deviation_from_mean:
            user_means = pd.merge(
                user_means, similarities, left_index=True, right_index=True, how="inner"
            )
            numerator = (
                np.sum(column * similarities)
                - (user_means["rating"] * user_means["similarities"]).sum()
            )
            return input_mean + numerator / denominator
        else:
            numerator = (column * similarities).sum()
            return numerator / denominator

    def get_scores(
        self, similarities: pd.Series, input_ratings: pd.DataFrame
    ) -> pd.DataFrame:
        relevant_ratings = pd.merge(
            self.ratings,
            similarities,
            left_on=[self.user_col],
            how="inner",
            right_index=True,
        )
        uii_matrix = relevant_ratings.pivot_table(
            index=[self.user_col], columns=[self.item_col], values="rating"
        )

        input_mean = input_ratings["rating"].mean()
        predicted_scores = uii_matrix.apply(
            lambda x: self.calculate_score(
                x,
                similarities,
                input_mean=input_mean,
                user_means=self.user_means,
                minimal_number_of_ratings=self.minimal_number_of_ratings,
                deviation_from_mean=self.deviation_from_mean,
            )
        )
        return predicted_scores
