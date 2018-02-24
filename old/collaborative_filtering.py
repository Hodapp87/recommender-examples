#!/usr/bin/env python

###########################################################################
# collaborative_filtering.py: Module for SVD-based recommender system
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2018-01-15
###########################################################################

import movielens_dl

import numpy as np
from numpy.linalg import norm
import scipy.optimize

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

def AltMinComplete(M, mask, T, k):
    """Runs the AltMinComplete algorithm, which performs matrix
    completion.  This returns matrices :U: and :V: such that (U @ V.T)
    is a low-rank approximation of the observed elements of :M: (those
    for which the corresponding value in :mask: is True).  That
    low-rank approximation also contains estimates for the
    "unobserved" elements of M.

    AltMinComplete is defined in algorithm 2 of arXiv 1212.0467,
    "Low-rank Matrix Completion using Alternating Minimization".

    Parameters:
    M -- Input matrix of shape (m,n)
    mask -- Binary matrix (same shape as :M:) in which True elements are
            considered as observed, and otherwise unobserved
    T -- Number of partitions/iterations to use
    k -- Rank of matrix

    Returns:
    U -- Matrix of shape (m,k)
    V -- Matrix of shape (k,n)
    """
    m,n = M.shape
    # TODO: Clipping (what is mu?)
    # Partition into 2*T + 1 subsets:
    i,j = np.where(mask)
    idxs = np.random.choice(i.size, i.size)
    T_ = 2*T+1
    omega_i = [[] for _ in range(T_)]
    omega_j = [[] for _ in range(T_)]
    for a in range(i.size):
        omega_i[a % T_].append(i[a])
        omega_j[a % T_].append(j[a])
    omega_i = [np.array(l) for l in omega_i]
    omega_j = [np.array(l) for l in omega_j]
    # Initialize:
    p = mask.sum() / mask.size
    mask2 = np.zeros(mask.shape)
    mask2[omega_i[0], omega_j[0]] = 1
    U,_,_ = scipy.sparse.linalg.svds(M * mask2 / p, k=k)
    print(U.shape)
    V = np.zeros((n,k))
    import sys
    # Iteratively alternate:
    for t in range(T):
        print(t)
        i0, j0 = omega_i[t + 1],     omega_j[t + 1]
        i1, j1 = omega_i[T + t + 1], omega_j[T + t + 1]
        def resid_v(V2):
            V2 = np.reshape(V2, (n,k))
            return norm((U @ V2.T - M)[i0, j0])
        V_flat = V.flatten()
        opt = scipy.optimize.minimize(resid_v, V_flat)
        V = np.reshape(opt.x, (n,k))
        def resid_u(U2):
            U2 = np.reshape(U2, (m,k))
            return norm((U2 @ V.T - M)[i1, j1])
        U_flat = U.flatten()
        opt = scipy.optimize.minimize(resid_u, U_flat)
        U = np.reshape(opt.x, (m,k))
    return U, V

user = movielens_dl.get_user_data()
user_mat = movielens_dl.user_to_utility(user).astype(np.float32)
mask = user_mat > 0
#means = user_mat.sum(axis=1) / np.maximum((user_mat > 0).sum(axis=1), 1)
#bias = np.ones(user_mat.shape) * means[:,np.newaxis] * (user_mat > 0)
#user_mat[:,:] -= bias

def test_slope_one():
    print("Loading data...")
    user = movielens_dl.get_user_data()
    train, test = movielens_dl.train_test_split(user, 0.25)
    train_mat = movielens_dl.user_to_utility(train).astype(np.float32)
    test_mat = movielens_dl.user_to_utility(test).astype(np.float32)
    cols = max(train_mat.shape[1], test_mat.shape[1])
    pad_extra = lambda arr,n: np.pad(arr, ((0,0),(0,n)), 'constant')
    train_mat = pad_extra(train_mat, cols - train_mat.shape[1])
    test_mat  = pad_extra(test_mat,  cols - test_mat.shape[1])
    train_mask = train_mat > 0
    test_mask = test_mat > 0
    print("Computing deviation matrix...")
    dev, counts = deviation(train_mat, mask)
    print("Predicting on training...")
    err = 0.0
    for row in train.itertuples():
        p = slope_one(train_mat, train_mask, dev, counts, row.user_id, row.movie_id)
        err += np.abs(p - row.rating)
    train_err = (err / len(train))
    print("Predicting on testing...")
    err = 0.0
    for row in test.itertuples():
        p = slope_one(test_mat, test_mask, dev, counts, row.user_id, row.movie_id)
        err += np.abs(p - row.rating)
    test_err = (err / len(test))
    print("Slope One -- MAE, training: {}; testing: {}".format(train_err, test_err))
    print("Predicting on training...")
    err = 0.0
    for row in train.itertuples():
        p = weighted_slope_one(train_mat, train_mask, dev, counts, row.user_id, row.movie_id)
        err += np.abs(p - row.rating)
    train_err = (err / len(train))
    print("Predicting on testing...")
    err = 0.0
    for row in test.itertuples():
        p = weighted_slope_one(test_mat, test_mask, dev, counts, row.user_id, row.movie_id)
        err += np.abs(p - row.rating)
    test_err = (err / len(test))
    print("Weighted Slope One -- MAE, training: {}; testing: {}".format(train_err, test_err))

if False:
    # Yes, this is ugly, but it's just here to test whether this can even
    # work
    import sys
    sys.path.append("./matrix-completion-whirlwind")
    import mc_util
    import mc_solve

    omega_mask = np.ones(mask.shape)
    omega_mask[~mask] = np.nan
    m_omega = mc_util.masked(user_mat, omega_mask)

    U_ls, V_ls = mc_solve.altMinSense(M_Omega=m_omega,
                                      Omega_mask=omega_mask,
                                      r=2, method='lsq')
    # CVX doesn't seem to work.  mcFrobSolveRightFactor_cvx tries to
    # access V_T.value.T, but V_T.value is None and I don't know if this
    # is some sort of error condition.

#np.save("AltMinComplete_cvx_2_U", U_ls)
#np.save("AltMinComplete_cvx_2_V", V_ls)

#U,V = AltMinComplete(user_mat, mask, 10, 100)
#print(U)
#print(V)
#numpy.save("AltMinComplete_U", U)
#numpy.save("AltMinComplete_V", V)

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

