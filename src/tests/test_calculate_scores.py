import pytest

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from book_collaborative_filtering.collaborative_filter import CollaborativeFilter


class TestCollaborativeFilterCalculateScore:
    column = pd.Series([1, np.nan, 5, 5, 1])

    def test_calculate_score(self) -> None:
        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([0, 1, 1, 1, 0]),
            input_mean=0.0,
            user_means=None,
            minimal_number_of_ratings=1,
            deviation_from_mean=False,
        )
        assert score == pytest.approx(5, rel=1e-6)

        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([0, 1, 1, 1, 0]),
            input_mean=0.0,
            user_means=None,
            minimal_number_of_ratings=4,
            deviation_from_mean=False,
        )
        assert score == pytest.approx(5, rel=1e-6)

        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([0, 1, 1, 1, 0]),
            input_mean=0.0,
            user_means=None,
            minimal_number_of_ratings=5,
            deviation_from_mean=False,
        )
        assert np.isnan(score)

        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([-1, 1, 1, 1, -1]),
            input_mean=0.0,
            user_means=None,
            minimal_number_of_ratings=1,
            deviation_from_mean=False,
        )
        assert np.isnan(score)

        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([1, 0, 0, 0, 1]),
            input_mean=0.0,
            user_means=None,
            minimal_number_of_ratings=4,
            deviation_from_mean=False,
        )
        assert score == pytest.approx(1, rel=1e-6)

        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([0.5, 0.5, 0.5, 0.5, 0.5]),
            input_mean=0.0,
            user_means=None,
            minimal_number_of_ratings=4,
            deviation_from_mean=False,
        )
        assert score == pytest.approx(3, rel=1e-6)

        similarities = np.random.random(5)
        similarities[1] = 0.
        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series(similarities),
            input_mean=0.0,
            user_means=None,
            minimal_number_of_ratings=4,
            deviation_from_mean=False,
        )
        assert score == pytest.approx(np.nansum(self.column*similarities)/np.sum(similarities), rel=1e-6)

    def test_calculate_score_mean_centered(self) -> None:
        # Check with user means all 3 -> score=5
        column = pd.Series([1, np.nan, 5, 5, 1])
        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([0, 1, 1, 1, 0], name="similarities"),
            input_mean=3.0,
            user_means=pd.Series([3, 3, 3, 3, 3], name="rating"),
            minimal_number_of_ratings=1,
            deviation_from_mean=True,
        )
        assert score == pytest.approx(5, rel=1e-6)

        # Check if input mean is smaller -> score=4 for input_mean=2
        column = pd.Series([1, np.nan, 5, 5, 1])
        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([0, 1, 1, 1, 0], name="similarities"),
            input_mean=2.0,
            user_means=pd.Series([3, 3, 3, 3, 3], name="rating"),
            minimal_number_of_ratings=1,
            deviation_from_mean=True,
        )
        assert score == pytest.approx(4, rel=1e-6)

        # Check if user means are different -> score=3
        column = pd.Series([1, np.nan, 5, 5, 1])
        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([0, 1, 1, 1, 0], name="similarities"),
            input_mean=3.0,
            user_means=pd.Series([3, 3, 5, 5, 3], name="rating"),
            minimal_number_of_ratings=1,
            deviation_from_mean=True,
        )
        assert score == pytest.approx(3, rel=1e-6)

        # Check if user means are different -> score=2
        column = pd.Series([1, np.nan, 5, 5, 1])
        score = CollaborativeFilter.calculate_score(
            self.column,
            pd.Series([1, 1, 1, 1, 1], name="similarities"),
            input_mean=3.0,
            user_means=pd.Series([3, 3, 5, 5, 3], name="rating"),
            minimal_number_of_ratings=1,
            deviation_from_mean=True,
        )
        assert score == pytest.approx(2, rel=1e-6)

