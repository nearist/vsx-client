# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:23:41 2017

@author: Chris
"""

import sys
from Client import *
import h5py
import time

# NOTE - These values should be updated with the ones you received.
api_key = ""
nearist_port = 0
nearist_ip = "0.0.0.0"

###########################################################################
#    Load the dataset into CPU memory
###########################################################################

print 'Loading dataset into memory...'
sys.stdout.flush()

h5f = h5py.File('./data/Google_word2vec_analogies_uint8_8std.h5', 'r')

# Load the dataset completely into memory--the slice operator at the end tells
# h5py how much of the matrix to load into memory, [:] loads the whole thing.
query_vecs = h5f['query_vecs'][:]
abc_i = h5f['abc_i'][:]
d_i = h5f['d_i'][:]

###########################################################################
#    Connect to the hardware
###########################################################################

print 'Connecting to Nearist board...'
sys.stdout.flush()

# Establish a connection to the appliance.  
c = Client()
c.open(nearist_ip, nearist_port, api_key)

# We're using L1 distance with a k-NN search.
c.set_distance_mode(Common.DistanceMode.L1)
c.set_query_mode(Common.QueryMode.KNN_A)

# For the analogies experiment, we want the top four matches.
c.set_read_count(4)

###########################################################################
#    Load the dataset into the Nearist board
###########################################################################

print("Loading dataset vectors from appliance harddisk...")
sys.stdout.flush()

# Time this step.
t0 = time.time()
    
# Load the training vectors *remotely* from the appliance harddisk.
c.load_dataset_file(file_name='/nearist/Google_word2vec_analogies/Google_word2vec_analogies_uint8_8std.h5',
                    dataset_name='word_vecs')


# Measure elapsed time.
elapsed = time.time() - t0

print '   Done. Loading took %.0f seconds.' % elapsed
sys.stdout.flush()

###########################################################################
#    Perform Analogies Test Queries
###########################################################################

num_right = 0

print 'Performing all', query_vecs.shape[0], ' queries...'
sys.stdout.flush()

# Reset the hardware timer.
c.reset_timer()

# Record the start time.
t0 = time.time()

# Find the nearest neighbors for all queries in this batch.
# The query call will break the queries into smaller batches and report
# progress.
batch_results = c.query(query_vecs.tolist(), batch_size=500, verbose=True)

# Get the total elapsed time (including internet overhead) in ms.
wall_time = (time.time() - t0) * 1000.0

# Get only the time spent on the appliance. Convert from nanoseconds to ms.
hw_time = c.get_timer_value() / 1E6

print '%22s %.0f ms' % ('Observed time:', wall_time)
print '%20s %.0f ms for %d queries' % ('Hardware time:', hw_time, query_vecs.shape[0])
print '%20s %.0f ms' % ('Average hw latency:', hw_time / query_vecs.shape[0])
print '%20s %.0f%%' % ('Internet Overhead:', (wall_time - hw_time) * 100.0 / float(hw_time))

###########################################################################
#    Score the Results
###########################################################################

# Score the results.
for j in range(0, len(batch_results)):
    
    # Get the results for query number 'j'.
    results = batch_results[j]

    assert(len(results) == 4)

    # Check the results. The correct answer is word 'd', but it's possible that
    # words 'a', 'b', and 'c' will appear in these results--we ignore these and
    # they don't count against accuracy. However, if any word other than 'a',
    # 'b', or 'c' is ranked higher than 'd', then the analogy has failed.
    for r in results:
       # If we encounter the correct index, we got it right.
       if r['ds_id'] == d_i[j]:
           num_right += 1
           break
       # If the result is either a, b, or c (the input words used
       # to form the query), then ignore it.
       elif r['ds_id'] in abc_i[j, :]:
           continue
       # If the result isn't a, b, c, or d, then it's wrong.
       else:
           break

# Calculate our accuracy.
accuracy = float(num_right) / float(query_vecs.shape[0]) * 100.0

print 'Final accuracy %.2f%% (%d / %d)' % (accuracy, num_right, query_vecs.shape[0])
