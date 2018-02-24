#!/usr/bin/env python

###########################################################################
# content_based.py: Module for content-based filtering algorithm
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2018-01-15
###########################################################################

def compute_similarity(a, b):
    """Compute similarity between two vectors.

    Parameters:
    a --
    b --

    Returns:
    sim -- Similarity between a and b
    """
    pass

def sim_to_rating(sim, avg_rating):
    """Converts between similarities and ratings.

    Parameters:
    sim --
    avg_rating --

    Returns:
    pred_rating -- Predicted ranking fo this user
    """
    pass

def get_ratings(train, test):
    """Predict users' ratings of movies.

    Parameters:
    train --
    test --

    Returns:
    estimated_ratings_collab --
    rmse -- RMS error between actual and predicted ratings
    """
    pass
