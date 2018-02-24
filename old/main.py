#!/usr/bin/env python

###########################################################################
# main.py: 
# Author: Chris Hodapp (hodapp87@gmail.com)
# Date: 2018-02-01
###########################################################################

import movielens_dl
import slope_one
import numpy as np
import time

def test_slope_one():
    print("Loading data...")
    user = movielens_dl.get_user_data()
    train, test = movielens_dl.train_test_split(user, 0.25)
    print()
    print("Computing deviation matrix (via DataFrame -> matrix -> deviation matrix...")
    t0 = time.time()
    train_mat = movielens_dl.user_to_utility(train).astype(np.float32)
    test_mat = movielens_dl.user_to_utility(test).astype(np.float32)
    cols = max(train_mat.shape[1], test_mat.shape[1])
    pad_extra = lambda arr,n: np.pad(arr, ((0,0),(0,n)), 'constant')
    train_mat = pad_extra(train_mat, cols - train_mat.shape[1])
    test_mat  = pad_extra(test_mat,  cols - test_mat.shape[1])
    train_mask = train_mat > 0
    test_mask = test_mat > 0
    t0 = time.time() - t0
    print("{:.3f} sec for DataFrame -> matrix".format(t0))
    t1 = time.time()
    dev, counts = slope_one.deviation(train_mat, train_mask)
    t1 = time.time() - t1
    print("{:.3f} sec for computing deviation matrix".format(t1))
    print("{:.3f} sec for both".format(t0 + t1))
    print()
    print("Computing deviation matrix (directly from DataFrame)...")
    t2 = time.time()
    dev2, counts2 = slope_one.deviation_from_df(train)
    dev2 = np.pad(dev2, ((0,dev.shape[0] - dev2.shape[0]),), 'constant')
    t2 = time.time() - t2
    print("{:.3f} sec for computing deviation matrix".format(t2))
    print()
    print("Matrices match: {}".format(np.allclose(dev, dev2)))
    print()
    print("Predicting on training...")
    err = 0.0
    for row in train.itertuples():
        p = slope_one.predict(train_mat, train_mask, dev, counts, row.user_id, row.movie_id, weighted=False)
        err += np.abs(p - row.rating)
    train_err = (err / len(train))
    print("Predicting on testing...")
    err = 0.0
    for row in test.itertuples():
        p = slope_one.predict(test_mat, test_mask, dev, counts, row.user_id, row.movie_id, weighted=False)
        err += np.abs(p - row.rating)
    test_err = (err / len(test))
    print("Slope One -- MAE, training: {}; testing: {}".format(train_err, test_err))
    print("Predicting on training...")
    err = 0.0
    for row in train.itertuples():
        p = slope_one.predict(train_mat, train_mask, dev, counts, row.user_id, row.movie_id, weighted=True)
        err += np.abs(p - row.rating)
    train_err = (err / len(train))
    print("Predicting on testing...")
    err = 0.0
    for row in test.itertuples():
        p = slope_one.predict(test_mat, test_mask, dev, counts, row.user_id, row.movie_id, weighted=True)
        err += np.abs(p - row.rating)
    test_err = (err / len(test))
    print("Weighted Slope One -- MAE, training: {}; testing: {}".format(train_err, test_err))

def main():
    test_slope_one()

if __name__ == "__main__":
    main()
