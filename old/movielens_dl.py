#!/usr/bin/env python

###########################################################################
# movielens_dl.py: Module for converting Movielens data
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2018-01-15
###########################################################################

import numpy as np
import pandas as pd
import sklearn.model_selection
import scipy.sparse

def get_user_data():
    """Reads the user information from the movielens database
    (https://grouplens.org/datasets/movielens/100k/).
    
    Returns:
    data -- movielens user data as a Pandas DataFrame with columns:
            (user_id, movie_id, rating, time)
    """
    df = pd.read_csv("ml-100k/u.data", sep="\t", header=None,
                     names=("user_id", "movie_id", "rating", "time"))
    # Convert Unix seconds to a Pandas timestamp:
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df

def get_movie_data():
    """Reads the movie information from the movielens database.  This
    gives the genres that the movie with the given ID fits into by
    whether each respective column is 0 or 1. Note that this may be
    multiple genres at once.

    Returns:
    data -- movielens movie information as a Pandas DataFrame with columns:
            movie_id, Action, Adventure, Animation, Childrens,
            Comedy, Crime, Documentary, Drama, Fantasy, Film-Noir,
            Horror, Musical, Mystery, Romance, Sci-Fi, Thriller, War,
            Western

    """
    names = ("movie_id", "Action", "Adventure", "Animation",
             "Childrens", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
             "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
             "Thriller", "War", "Western")
    # Note from ml-100k/README: We need the first column, and last 19.
    df = pd.read_csv("ml-100k/u.item", sep="|", header=None,
                     names=names, usecols=[0] + list(range(6,24)))
    return df

def train_test_split(user_data, frac):
    """Splits the user information from the movielens database into
    training & testing datasets.

    Parameters:
    user_data -- User information from movielens database.
    frac -- Fraction of ratings for test matrix (e.g. 0.15 if 15% of
            data should be in testing data, and the rest in training)

    Returns:
    train_mat -- Training user item matrix (as a NumPy matrix)
    test_mat -- Testing user item matrix (as a NumPy matrix)
    """
    return sklearn.model_selection.train_test_split(
        user_data, test_size=frac)

def user_to_utility(user_data):
    """Convert movielens user data from a dataframe (with one row per
    movie rating) to a utility matrix (with one row per user, and one
    column per movie, and each element gives the user's rating on that
    movie from 1 to 5).

    Note that in this matrix, a value of 0 is considered to be a
    missing value - not a rating of zero.

    Parameters:
    user_data -- Dataframe of the format returned by get_user_data()

    Returns:
    matrix -- Utility matrix in the form of a NumPy array

    """
    shape = (int(user_data["user_id"].max() + 1),
             int(user_data["movie_id"].max() + 1))
    movies  = user_data["movie_id"]
    users   = user_data["user_id"]
    ratings = user_data["rating"]
    m = scipy.sparse.coo_matrix((ratings, (users, movies)),
                                shape=shape, dtype=np.int8)
    m = m.toarray()
    return m
