#!/usr/bin/env python

###########################################################################
# collaborative_filtering.py: Module for SVD-based recommender system
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2018-01-15
###########################################################################

import movielens_dl

import numpy as np
from numpy.linalg import norm

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
    k0 = tau / (delta * np.linalg.norm(M))
    Y = k0 * delta * M
    #print("k0={}".format(k0))
    #Y = np.zeros(shape=M.shape)
    r = 0
    for k in range(kmax):
        print("Iteration {}/{}...".format(k + 1, kmax))
        s = r + 1
        sv = np.inf
        while sv > tau: # or (s-l) <= 1:
            if s >= min(Y.shape):
                break
            U,S,Vt = svds(Y, k=s)
            # N.B. 'svds' returns in *ascending* order of singular
            # value, not descending (like most packages seem to)
            sv = S[0]
            print("s={}, sv={}".format(s, sv))
            s = s + l
        #import pdb; pdb.set_trace()
        #r_ = np.where(S > tau)[0]
        if True or r_.size:
            #r = r_[0] + 1
            #print("S={}, U={}, Vt={}, r={}".format(S.shape, U.shape, Vt.shape, r))
            #print("S'={}, U'={}, Vt'={}".format(S[:r].shape, U[:,:r].shape, Vt[:r,:].shape))
            #X = ((S[-r:] - tau) * U[:,-r:]) @ Vt[-r:,:]
            X = ((S[S > tau] - tau) * U[:, S > tau]) @ Vt[S > tau, :]
            #print("X={0.shape}".format(X))
        else:
            X = np.zeros(shape=M.shape)
        # Ending condition:
        num = np.linalg.norm((X - M) * mask)
        den = np.linalg.norm(M)
        print("num={}, den={}, num/den={}".format(num, den, num/den))
        if (num / den) < eps:
            break
        Y[:,:] = (Y + delta * (M - X))*mask
    return X

def UV(M, mask, d, lr=0.5):
    n,m = M.shape
    U = np.zeros((n,d))
    V = np.zeros((d,m))
    # Initialize elements:
    avg = M[mask].mean()
    U[:,:] = np.sqrt(avg / d)
    V[:,:] = np.sqrt(avg / d)
    # TODO: Perturb randomly?
    delta = np.inf
    rmse = []
    rmse_avg = None
    i = 0
    while delta > 0.001:
        i += 1
        # TODO: Check for change in RMSE
        # Pick an element in U to optimize:
        r = np.random.randint(n)
        s = np.random.randint(d)
        num = (V[s,:] * (M[r,:] - (U[r,:] @ V) + U[r,s]*V[s,:]) * mask[r,:]).sum()
        den = np.square(V[s,:] * mask[r,:]).sum()
        if (den > 0):
            U[r,s] = (1-lr)*U[r,s] + lr * num / den
        # Pick an element in V to optimize:
        r = np.random.randint(d)
        s = np.random.randint(m)
        num = (U[:,r] * (M[:,s] - (U @ V[:,s]) + U[:,r]*V[r,s]) * mask[:,s]).sum()
        den = np.square(U[:,r] * mask[:,s]).sum()
        if (den > 0):
            V[r,s] = (1-lr)*V[r,s] + lr * num / den
        rmse.append(norm((U @ V - M)*mask))
        if len(rmse) >= 100:
            old_avg = rmse_avg
            rmse_avg = sum(rmse) / 100
            rmse = []
            if old_avg is not None:
                if old_avg > 0:
                    delta = (old_avg - rmse_avg) / old_avg
                else:
                    delta = 0
            print("{}: RMSE={} change={:2f}%".format(i, rmse_avg, delta * 100))
    return U, V

user = movielens_dl.get_user_data()
user_mat = movielens_dl.user_to_utility(user).astype(np.float32)
mask = user_mat > 0
means = user_mat.sum(axis=1) / np.maximum((user_mat > 0).sum(axis=1), 1)
bias = np.ones(user_mat.shape) * means[:,np.newaxis] * (user_mat > 0)
user_mat[:,:] -= bias

# SVT test params:
if False:
    n1, n2 = user_mat.shape
    mask = user_mat > 0
    m = mask.sum()
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
