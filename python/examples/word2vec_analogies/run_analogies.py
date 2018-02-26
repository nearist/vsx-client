# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:23:41 2017

@author: Chris
"""

import sys
sys.path.append('../../src/')

from Client import *
import h5py
import time


###########################################################################
#    Load the dataset into CPU memory
###########################################################################

print 'Loading dataset into memory...'
sys.stdout.flush()

h5f = h5py.File('./data/Google_word2vec_analogies_uint8_8std.h5', 'r')

# Load the dataset completely into memory--the slice operator at the end tells
# h5py how much of the matrix to load into memory, [:] loads the whole thing.
query_vecs = h5f['query_vecs'][:]
word_vecs = h5f['word_vecs'][:]
abc_i = h5f['abc_i'][:]
d_i = h5f['d_i'][:]

###########################################################################
#    Connect to the hardware
###########################################################################

print 'Connecting to Nearist board...'
sys.stdout.flush()

c = Client()
c.open("000.000.0.00", 5555)

c.reset()

c.set_distance_mode(DistanceMode.L1)
c.set_query_mode(QueryMode.KNN_A)

# For the analogies experiment, we want the top four matches.
c.set_read_count(4)

###########################################################################
#    Load the dataset into the Nearist board
###########################################################################

print 'Loading dataset into Nearist hardware...'
sys.stdout.flush()

# Time this step.
t0 = time.time()

# Load the word vectors.
c.ds_load(word_vecs.tolist())

# Measure elapsed time.
elapsed = time.time() - t0

print '   Done. Loading took %.0f seconds.' % elapsed

###########################################################################
#    Perform Analogies Test Queries
##########################################################################

num_right = 0

num_queries = query_vecs.shape[0]

# NearestNeighbors is more efficient when given multiple queries to work
# on simultaneously. This batch size is somewhat arbitrary--we haven't
# experimented with this parameter.
batch_size = 128

print 'Performing all', num_queries, ' queries...'
sys.stdout.flush()

# Record the start time.
t0 = time.time()

# For each batch of query vectors...
for i in range(0, num_queries, batch_size):

    # Calculate the index of the last vector in this query batch.
    end_i = min(i + batch_size, num_queries)

    # Progress update.
    if not i == 0:
        # Estimate how much time is left to complete the test.
        queries_per_sec = ((time.time() - t0)  / i)
        time_est = queries_per_sec * (num_queries - i) / 60.0


        print 'Query %5d / %5d (%.0f%%) Time Remaining:~%.0f min....' % (i, num_queries, float(i) / num_queries * 100.0, time_est)
        sys.stdout.flush()

    # Find the nearest neighbors for all queries in this batch.
    batch_results = c.query(query_vecs[i:end_i, :].tolist())

    # Loop over the batch results.
    for j in range(0, len(batch_results)):
        # Get the results for query number (i + j)
        results = batch_results[j]

        assert(len(results) == 4)

        # Look through the four results...
        for r in results:
           # If we encounter the correct index, we got it right.
           if r['ds_id'] == d_i[i + j]:
               num_right += 1
               break
           # If the result is either a, b, or c (the input words used
           # to form the query), then ignore it.
           elif r['ds_id'] in abc_i[i + j, :]:
               continue
           # If the result isn't a, b, c, or d, then it's wrong.
           else:
               break


elapsed = time.time() - t0

print 'Done, %.0f seconds' % elapsed

accuracy = float(num_right) / float(query_vecs.shape[0]) * 100.0

print 'Final accuracy %.2f%% (%d / %d)' % (accuracy, num_right, query_vecs.shape[0])
