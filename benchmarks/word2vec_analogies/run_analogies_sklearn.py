# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:49:49 2017

@author: Chris
"""

import h5py
from sklearn.neighbors import NearestNeighbors
import sys
import time
import psutil
import os


# Read in the path to the dataset file from the command line.
script_name, dataset_file = sys.argv


def mem_usage_gb():
    """
    Returns the current memory usage in gigabytes (2^30 bytes).
    """
    # Retrieve the current memory usage in bytes.
    mem_bytes = psutil.Process(os.getpid()).memory_info()[0]     
    
    # Convert to gigabytes and return.
    return mem_bytes / (2 ** 30)

# This little check has to do with the multiprocess module, which I believe is
# used for the n_jobs parameter of the NearestNeighbors model.
if __name__ == '__main__':

    ###########################################################################
    #    Load the dataset
    ###########################################################################
            
    startMem = mem_usage_gb()
    
    print 'Memory usage at start %.2fGB' % startMem
    print 'Loading the dataset file...'
    sys.stdout.flush()    
    
    h5f = h5py.File(dataset_file, 'r')
    
    # Load the dataset completely into memory--the slice operator at the end tells
    # h5py how much of the matrix to load into memory, [:] loads the whole thing.
    query_vecs = h5f['query_vecs'][:]
    word_vecs = h5f['word_vecs'][:]
    abc_i = h5f['abc_i'][:]        
    d_i = h5f['d_i'][:]        
    
    print 'Loading dataset took %.2fGB extra' % (mem_usage_gb() - startMem)
    
    ###########################################################################
    #    Create Nearest Neighbor Model
    ###########################################################################
    
    startMem = mem_usage_gb()
    
    # Create the NearestNeighbors object to perform the search using Cosine similarity
    # Using more than 1 job is probablamatic, I believe because of memory consumption.
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='brute', metric='cosine', n_jobs=1)
    
    # Give it the word vectors to be searched.
    nbrs.fit(word_vecs)
    
    print 'Creating Neighbors model took %.2fGB extra' % (mem_usage_gb() - startMem)
        
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
        batch_results = nbrs.kneighbors(X=query_vecs[i:end_i, :], return_distance=False)
        
        # Loop over the batch results.     
        for j in range(0, len(batch_results)):
            # Get the results for query number (i + j)
            results = batch_results[j]        
    
            assert(len(results) == 4)
    
            # Look through the four results...
            for r in results:
               # If we encounter the correct index, we got it right.       
               if r == d_i[i + j]:
                   num_right += 1
                   break
               # If the result is either a, b, or c (the input words used
               # to form the query), then ignore it.
               elif r in abc_i[i + j, :]:
                   continue
               # If the result isn't a, b, c, or d, then it's wrong.
               else:
                   break
            
    
    elapsed = time.time() - t0
    
    print 'Done, %.0f seconds' % elapsed 
    
    accuracy = float(num_right) / float(query_vecs.shape[0]) * 100.0
    
    print 'Final accuracy %.2f%% (%d / %d)' % (accuracy, num_right, query_vecs.shape[0])         
        
    