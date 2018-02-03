###########################################################################
# slope_one.py: Slope One Predictors implementation for recommender system
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2018-02-02
###########################################################################

"""Module implementing Slope One Predictors over NumPy arrays.

Slope One Predictors come from the paper arXiv:cs/0702144v2, "Slope
One Predictors for Online Rating-Based Collaborative Filtering,"
by Daniel Lemire and Anna Maclachlan.
"""

import numpy as np

def deviation_from_df(data):
    """Computes the average deviation described in the "Slope One
    Predictors" paper, and returns this in matrix form, along with a
    matrix giving the number of users who have rated each pair of
    items.

    For the returned 'dev' matrix, element [j,i] gives the average
    deviation of item i with respect to item j.  Note that dev[i,j] =
    -dev[j,i].
    For 'counts', element [i,j] gives the number of users who have
    rated both item i and item j.

    Parameters:
    data -- Pandas dataframe; needs columns "user_id", "movie_id", "rating"

    Returns:
    dev -- NumPy array of shape (n,n), if M.shape = (m,n).
    counts -- NumPy array of shape (n,n)
    """
    n = data["movie_id"].max() + 1
    m = data["user_id"].max() + 1
    movie2idx = [np.where(data["movie_id"] == i)[0] for i in range(n)]
    user2idx = [np.where(data["user_id"] == i)[0] for i in range(m)]    
    diff = np.zeros((n,n))
    dev = np.zeros((n,n))
    counts = np.zeros((n,n))    
    for i in range(n):
        # 'i' = the index of one movie
        for row in data.iloc[movie2idx[i]].itertuples():
            # u = user who rated movie i
            u = row.user_id
            # u_i = user u's rating on movie i
            u_i = row.rating
            # row.movie_id = i (due to how movie2idx is made)
            others = data.iloc[user2idx[u]]
            # others = dataframe of ratings, all of which are by user
            # u.  Then u_js is a series of ratings for those movies,
            # and js is the movie indices corresponding to those:
            u_js, js = others["rating"], others["movie_id"]
            # As user u also rated movie i, all other ratings here are
            # (combined with u_i): a pair of ratings by the same user.
            # We can exploit some symmetry too:
            #mask = js > i
            #u_js, js = u_js[mask], js[mask]
            # So, we may add up u_j - u_i below (and for the flipped
            # indices it's simply negated):
            d = u_js - u_i
            diff[i, js] -= d
            #diff[js, i] += d
            # and count the ratings, so we may compute an average:
            counts[i, js] += 1
            #counts[js, i] += 1
    # This is still much slower than the version that starts from a
    # matrix.
    diff -= diff.T
    counts += counts.T
    dev = diff / np.maximum(1, counts)
    return dev, counts

def deviation(M, mask):
    """Computes the average deviation described in the "Slope One
    Predictors" paper, and returns this in matrix form, along with a
    matrix giving the number of users who have rated each pair of
    items.

    For the returned 'dev' matrix, element [j,i] gives the average
    deviation of item i with respect to item j.  Note that dev[i,j] =
    -dev[j,i].
    For 'counts', element [i,j] gives the number of users who have
    rated both item i and item j.

    Parameters:
    M -- NumPy array for utility matrix (each row is a user, each column
         an item, and each element a rating)
    mask -- Boolean array (same shape as M) which is True for elements of
            :M: that are observed

    Returns:
    dev -- NumPy array of shape (n,n), if M.shape = (m,n).
    counts -- NumPy array of shape (n,n)
    """
    m,n = M.shape
    # In order to get *average* deviation, we need the number of users
    # who rated each pair of items.  If we take columns i and j from
    # 'mask', element-wise logical AND them, and sum them, this gives
    # that number for items i and j.
    #
    # Note that this is just the dot product of columns i and j - and
    # that matrix multiplication A*B is just the dot products of every
    # combination of A's rows and B's columns.  Thus, if A=mask.T and
    # B=mask, that's every pair of columns of mask, dotted:
    m2 = mask.astype(np.int)
    counts = m2.T @ m2
    # Then counts[i,j] = number of users who rated both item i and j.
    #
    # Now, what we really want is average deviation.  However, the way
    # the summation is written, this is just total deviation divided
    # by number of ratings.  We already computed number of ratings
    # above.  So, all we need is total deviation.
    #
    # But also, total deviation is the sum of (u_j - u_i) for every
    # pair of i and j rated by the same user.  We can split that into
    # the difference of the sum of u_j and the sum of u_i, provided
    # that we use the exact same summation for each.
    #
    # Note that if we take the dot product between column j of the
    # mask, and column i of M, what we get is: The sum of only the
    # ratings for item i done by users who also rated item j.
    # Following the same reasoning for 'counts', this means that if S
    # is the matrix product of mask.T and M, then S[j,i] is that same
    # sum of ratings for any i and j.
    S = m2.T @ M
    # If S[j,i] is the sum of only ratings for item i by users who
    # also rated j, clearly S[i,j] is the sum of only ratings for item
    # j by users who also rated i.  This has some symmetry to it: Both
    # S[j,i] and S[i,j] summed over the same ratings (always those by
    # users who rated both item i and item j).
    #
    # Note that this is basically the definition of u_j and u_i.  The
    # only thing left is that we need to do S[i,j]-S[j,i] for every i
    # and j, and this gives the above u_j - u_i for all i & j:
    diffs = S.T - S
    # and normalizing this by the counts turns it to an average:
    dev = diffs / np.maximum(1, counts)
    # By convention, if counts=0 then diffs=0, so we just say
    # deviation=0 to avoid division by 0 - hence np.maximum(1, counts)
    return dev, counts

def predict(M, mask, dev, counts, u, j, weighted = False):
    """Predicts, based on the "Slope One" scheme, how user :u: would rate
    item :j:.

    Parameters:
    M -- NumPy array with utility matrix (see :deviation: function)
    mask -- Boolean array giving observed elements (see :deviation:)
    dev -- Deviation matrix computed over same set of items
    counts -- Matrix of counts for each pair; counts[i,j] is the number of
              users who have rated both item i and item j
    u -- User index (i.e. row in :M:)
    j -- Item index (i.e. column in :M:)
    weighted -- If True, use Weighted Slope One rather than just Slope One.
                (default False)

    Returns:
    p -- Predicted rating (as a float) for user u and item j
    """
    m,n = M.shape
    # S_u is a mask over M's columns for items user 'u' rated:
    S_u = mask[u, :]
    if weighted:
        # In 'Weighted Slope One', we sum over everything user 'u' rated,
        # regardless of whether other users rated both this and item j:
        S_u[j] = False
        c_j = counts[j, S_u]
        devs = dev[j, S_u]
        u = M[u, S_u]
        return ((devs + u) * c_j).sum() / max(1.0, c_j.sum())
    else:
        # In the 'Slope One' formula we are summing over R_j, which is:
        # Every item 'i' (i != j), such that: user 'u' rated item 'i', and
        # at least one other user rated both item 'i' and item 'j'.
        # Below we compute this likewise as a mask over M's columns:
        R_j = S_u * (counts[u, :] > 0)
        R_j[j] = False
        u = M[u, R_j].sum()
        devs = dev[j, R_j].sum()
        card = max(1.0, R_j.sum())
        return (u + devs) / card
