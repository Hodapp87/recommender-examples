#!/usr/bin/env python

###########################################################################
# collaborative_filtering.py: Module for SVD-based recommender system
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2018-01-15
###########################################################################

import numpy as np
import numpy.linalg

def SVD(X):
    """Returns the singular value decomposition of the input matrix.

    Parameters:
    X -- Input matrix of shape (m,n), as a NumPy matrix or array

    Returns:
    U -- Left singular vectors as NumPy array of shape (m, m)
    S -- Singular values as 1D NumPy array of shape (m,)
    V -- Right singular vectors as NumPy array of shape (n, n)

    """
    U, S, V = numpy.linalg.svd(X)
    return U, S, V

def get_ratings(train, test):
    """Compute the ratings and error (???)

    Parameters:
    train --
    test --

    Returns:
    estimated_ratings_collab --
    rmse -- RMS error between actual user ratings, and predicted ratings
    """
    pass
