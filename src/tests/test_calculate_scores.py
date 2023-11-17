import pytest

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from book_collaborative_filtering.collaborative_filter import CollaborativeFilter


class TestCollaborativeFilterCalculateScore:
    column = pd.Series([1, np.nan, 5, 5, 1])

    def test_calculate_score(self):
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


