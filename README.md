# Evaluating Design Choices of User-User Collaborative Filtering Across Datasets

This branch provides code to evaluate different user-based recommender systems on two public datasets. See also ![this post](https://medium.com/@felixkemeth/evaluating-design-choices-of-user-user-collaborative-filtering-across-datasets-f09995547267) for a detailed explanation.

+ goodbooks-10k: Data taken from ![this repo](https://github.com/zygmuntz/goodbooks-10k)
+ Movielens-100k: Data taken from ![Kaggle](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset/)

You may also pull the data from this repo using ![git lfs](https://git-lfs.com/).

To evaluate design choices of the user-user recommender system on the goodbooks-10k dataset, run

`evaluate_book_cf.ipynb`

To evaluate design choices of the user-user recommender system on the Movielens-100k dataset, run

`evaluate_movie_cf.ipynb`

To create plots of the resulting metrics, run

`evaluation_plots.ipynb`
