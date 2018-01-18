#!/usr/bin/env python

###########################################################################
# collaborative_filtering.py: Module for SVD-based recommender system
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2018-01-15
###########################################################################

import movielens_dl

import numpy as np
#import numpy.linalg
import scipy.sparse
from scipy.sparse.linalg import svds, norm

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

def SVT(M, delta, eps, tau, l, kmax):
    """Run the Singular Value Thresholding algorithm described in
    arXiv:0810.03286v1 (A Singular Value Thresholding Algorithm for
    Matrix Computation) to perform matrix completion on the input
    matrix.  Values of 0 in the input matrix are considered to be
    missing data.

    Parameters:
    M -- Input matrix
    delta -- Step size (delta) in algorithm
    eps -- Tolerance in algorithm
    tau -- Tau parameter in algorithm (sets 'cutoff' in singular values)
    l -- Increment value in algorithm
    kmax -- Maximum number of iterations

    Returns:
    ???
    (TODO)
    """
    mask = M > 0
    #k0 = tau / (delta * numpy.linalg.norm(M))
    #Y = k0 * delta * M
    #print("k0={}".format(k0))
    Y = np.zeros(shape=M.shape)
    r = 0
    for k in range(kmax):
        print("Iteration {}/{}...".format(k + 1, kmax))
        s = r + 1
        sv = np.inf
        while sv > tau: # or (s-l) <= 1:
            if s >= min(Y.shape):
                break
            U,S,Vt = svds(Y, k=s)
            # SciPy is weird:
            U, S, Vt = U[:, ::-1], S[::-1], Vt[::-1, :]
            sv = S[-1]
            print("s={}, sv={}".format(s, sv))
            s = s + l
        r_ = np.where(S > tau)[0]
        if r_.size:
            r = r_[-1] + 1
            #print("S={}, U={}, Vt={}, r={}".format(S.shape, U.shape, Vt.shape, r))
            #print("S'={}, U'={}, Vt'={}".format(S[:r].shape, U[:,:r].shape, Vt[:r,:].shape))
            X = ((S[:r] - tau) * U[:,:r]) @ Vt[:r,:]
            #print("X={0.shape}".format(X))
        else:
            X = np.zeros(shape=M.shape)
        # Ending condition:
        num = np.linalg.norm((X - M) * mask)
        den = np.linalg.norm(M)
        print("num={}, den={}, num/den={}".format(num, den, num/den))
        if (num / den) < eps:
            break
        Y = (Y + delta * (M - X))*mask
    return X

# SVT test params:
n1 = 500
n2 = 400
mask = np.random.random((n1,n2)) < 0.1
m = mask.sum()
mat = (np.random.randint(5, size=(n1,n2)) + 1) * mask
p = m / (n1 * n2)
delta = 1.2 / p
eps = 1e-4
tau = 5 * min(n1,n2)
l = 5
kmax = 500

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
