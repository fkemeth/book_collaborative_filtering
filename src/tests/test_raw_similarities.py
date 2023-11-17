import pytest

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from book_collaborative_filtering.collaborative_filter import CollaborativeFilter


class TestCollaborativeFilterRawSimilarities:
    ratings = pd.DataFrame(
        data={
            "user_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
            "item_id": ["a", "b", "c", "d", "a", "b", "c", "d", "a", "c", "d", "e", "a", "b", "c", "d"],
            "rating": [1, 2, 2, 3, 2, 3, 4, 5, 5, 4, 3, 2, 5, 5, 5, 5],
        }
    )
    input_ratings = pd.DataFrame(
        data={"user_id": [99, 99, 99], "item_id": ["a", "b", "c"], "rating": [1, 2, 2]}
    )

    def test_calculate_raw_similarities_wo_filters(self):
        similarities = CollaborativeFilter.calculate_raw_similarities(
            self.ratings,
            self.input_ratings,
            minimum_number_of_books_rated_in_common=2,
            correlation_method="pearson",
            user_col="user_id",
            item_col="item_id",
            remove_self_similarity=False,
            remove_nan_similarities=False,
            remove_constant_ratings=False
        )
        assert len(similarities) == 5
        assert similarities[1] == pytest.approx(1., rel=1e-6)
        x = np.array([2, 3, 4])
        y = np.array([1, 2, 2])
        p_coeff = np.mean((x-x.mean())*(y-y.mean()))/(np.std(x)*np.std(y))
        assert similarities[2] == pytest.approx(p_coeff, rel=1e-6)
        assert similarities[3] == pytest.approx(-1., rel=1e-6)
        assert np.isnan(similarities[4])
        assert similarities[99] == pytest.approx(1., rel=1e-6)

    def test_calculate_raw_similarities_with_filters(self):
        similarities = CollaborativeFilter.calculate_raw_similarities(
            self.ratings,
            self.input_ratings,
            minimum_number_of_books_rated_in_common=2,
            correlation_method="pearson",
            user_col="user_id",
            item_col="item_id",
            remove_self_similarity=True,
            remove_nan_similarities=True
        )
        assert len(similarities) == 3
        assert similarities[1] == pytest.approx(1., rel=1e-6)
        x = np.array([2, 3, 4])
        y = np.array([1, 2, 2])
        p_coeff = np.mean((x-x.mean())*(y-y.mean()))/(np.std(x)*np.std(y))
        assert similarities[2] == pytest.approx(p_coeff, rel=1e-6)
        assert similarities[3] == pytest.approx(-1., rel=1e-6)

    def test_calculate_raw_similarities_spearman(self):
        similarities = CollaborativeFilter.calculate_raw_similarities(
            self.ratings,
            self.input_ratings,
            minimum_number_of_books_rated_in_common=2,
            correlation_method="spearman",
            user_col="user_id",
            item_col="item_id",
        )
        assert len(similarities) == 3
        assert similarities[1] == pytest.approx(1., rel=1e-6)
        x = np.array([2, 3, 4])
        y = np.array([1, 2, 2])
        p_coeff = spearmanr(x, y).correlation
        assert similarities[2] == pytest.approx(p_coeff, rel=1e-6)
        assert similarities[3] == pytest.approx(-1., rel=1e-6)
    
    def test_calculate_raw_similarities_min_rated(self):
        similarities = CollaborativeFilter.calculate_raw_similarities(
            self.ratings,
            self.input_ratings,
            minimum_number_of_books_rated_in_common=1,
            correlation_method="spearman",
            user_col="user_id",
            item_col="item_id",
        )
        assert len(similarities) == 3
        similarities = CollaborativeFilter.calculate_raw_similarities(
            self.ratings,
            self.input_ratings,
            minimum_number_of_books_rated_in_common=2,
            correlation_method="spearman",
            user_col="user_id",
            item_col="item_id",
        )
        assert len(similarities) == 3
        similarities = CollaborativeFilter.calculate_raw_similarities(
            self.ratings,
            self.input_ratings,
            minimum_number_of_books_rated_in_common=3,
            correlation_method="spearman",
            user_col="user_id",
            item_col="item_id",
        )
        assert len(similarities) == 2
        similarities = CollaborativeFilter.calculate_raw_similarities(
            self.ratings,
            self.input_ratings,
            minimum_number_of_books_rated_in_common=4,
            correlation_method="spearman",
            user_col="user_id",
            item_col="item_id",
        )
        assert len(similarities) == 0

