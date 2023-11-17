import pandas as pd
from book_collaborative_filtering.collaborative_filter import CollaborativeFilter

class TestCollaborativeFilterNeighborhoods:
    similarities = pd.Series(data = [1, 0, -1, 0.5], index=[1, 2, 3, 4], name="similarities")

    def test_select_neighborhoods_none(self):
        similarities = CollaborativeFilter.select_neihgborhood(
            self.similarities,
            neighborhood_method=None,
        )
        assert similarities.equals(self.similarities)

    def test_select_neighborhoods_threshold(self):
        similarities = CollaborativeFilter.select_neihgborhood(
            self.similarities,
            neighborhood_method="threshold",
            minimal_similarity=-1
        )
        assert similarities.equals(pd.Series(data = [1, 0, 0.5], index=[1, 2, 4], name="similarities"))

        similarities = CollaborativeFilter.select_neihgborhood(
            self.similarities,
            neighborhood_method="threshold",
            minimal_similarity=0.2
        )
        assert similarities.equals(pd.Series(data = [1, 0.5], index=[1, 4], name="similarities"))

    def test_select_neighborhoods_number(self):
        similarities = CollaborativeFilter.select_neihgborhood(
            self.similarities,
            neighborhood_method="number",
            number_of_neighbors=3
        )
        assert similarities.equals(pd.Series(data = [1, 0.5, 0], index=[1, 4, 2], name="similarities"))

        similarities = CollaborativeFilter.select_neihgborhood(
            self.similarities,
            neighborhood_method="number",
            number_of_neighbors=2
        )
        assert similarities.equals(pd.Series(data = [1, 0.5], index=[1, 4], name="similarities"))