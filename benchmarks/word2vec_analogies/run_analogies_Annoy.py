# -*- coding: utf-8 -*-
"""
@author: levgas
"""

import h5py
import sys
import time
import psutil
import os
import csv

from annoy import AnnoyIndex

# Read in the path to the dataset file from the command line.
script_name, dataset_file, result_csv_file = sys.argv

# Number of trees in annoy
num_trees_vals = [10, 100, 500]

# Number of searching neighbors
K = 4

def mem_usage_gb():
    """
    Returns the current memory usage in gigabytes (2^30 bytes).
    """
    # Retrieve the current memory usage in bytes.
    mem_bytes = psutil.Process(os.getpid()).memory_info()[0]

    # Convert to gigabytes and return.
    return mem_bytes / (2 ** 30)

# Print iterations progress
def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    len_filled = int(length * iteration // total)
    bar = fill * len_filled + '-' * (length - len_filled)
    sys.stdout.write('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    sys.stdout.flush()

    # Print New Line on Complete
    if iteration >= total:
        print()

###########################################################################
#    Load the dataset
###########################################################################

startMem = mem_usage_gb()

print('Memory usage at start %.2fGB' % startMem)
print('Loading the dataset file...')
sys.stdout.flush()

h5f = h5py.File(dataset_file, 'r')

# Load the dataset completely into memory--the slice operator at the end tells
# h5py how much of the matrix to load into memory, [:] loads the whole thing.
query_vecs = h5f['query_vecs'][:]
word_vecs = h5f['word_vecs'][:]
abc_i = h5f['abc_i'][:]
d_i = h5f['d_i'][:]

print('Loading dataset took %.2fGB extra' % (mem_usage_gb() - startMem))
sys.stdout.flush()

###########################################################################
#    Create Annoy object
###########################################################################

print('\nCreating Annoy')
sys.stdout.flush()

memStart = mem_usage_gb()
tmStart = time.time()

# Create the annoy object to perform the search using Cosine similarity
annoy = AnnoyIndex(len(word_vecs[0]))

# Give it the word vectors to be searched.
for i in range(len(word_vecs)):
    if i % 200000 == 0:
        print_progress_bar(i, len(word_vecs), prefix='Annoy indexing progress', suffix='Complete')

    # Add item to the index.
    annoy.add_item(i, word_vecs[i])

# Print completed status bar
print_progress_bar(len(word_vecs), len(word_vecs), prefix='Annoy indexing progress', suffix='Complete')

tmIndexing = time.time() - tmStart
memIndexing = mem_usage_gb() - memStart

print('Annoy indexing took %.2fGB extra and %.2fsec time' % (memIndexing, tmIndexing))
sys.stdout.flush()

###########################################################################
#    Run accuracy tests for different number of trees
###########################################################################

# query vectors count
num_queries = query_vecs.shape[0]

# create report file
with open(result_csv_file, 'w') as csvOut:

    writer = csv.writer(csvOut)
    # write header
    writer.writerow(['Number of trees',
                     'Indexing + Build time(sec)', 'Indexing + Build memory(GB)',
                     'Test time(sec)', 'Correct(Percent)'])

    for num_trees in num_trees_vals:

        annoy.unbuild()

        print('Building annoy with %d tree(s)...' % num_trees)
        sys.stdout.flush()

        # Record used memory at the start time
        memStart = mem_usage_gb()
        # Record the start time
        tmStart = time.time()

        # Build annoy tree
        annoy.build(num_trees)

        # Calculate build time
        tmBuild = time.time() - tmStart
        # Calculate used memory for build
        memBuild = mem_usage_gb() - memStart

        print('Building annoy took %.2fGB extra and %.2fsec time' % (memBuild, tmBuild))
        sys.stdout.flush()

        ###########################################################################
        #    Perform Analogies Test Queries
        ##########################################################################

        num_right = 0

        print('Performing all %d queries...' % num_queries)
        sys.stdout.flush()

        # Record the start time.
        tmStart = time.time()

        # For each batch of query vectors...
        for i in range(num_queries):

            # Find the nearest neighbors for the query
            result = annoy.get_nns_by_vector(query_vecs[i], K, include_distances=False)
            assert(len(result) == K)

            # Look through the k results...
            for r in result:
                # If we encounter the correct index, we got it right.
                if r == d_i[i]:
                    num_right += 1
                    break
                # If the result is either a, b, or c (the input words used
                # to form the query), then ignore it.
                elif r in abc_i[i, :]:
                    continue
                # If the result isn't a, b, c, or d, then it's wrong.
                else:
                    break

        # Accumulate elapsed time
        tmElapsed = time.time() - tmStart
        print('Done, %.0f seconds' % tmElapsed)

        # Calculate accuracy
        accuracy = float(num_right) / float(num_queries) * 100.0
        print('Final accuracy %.2f%% (%d / %d)' % (accuracy, num_right, num_queries))

        # write test result
        writer.writerow([num_trees, tmIndexing + tmBuild, memIndexing + memBuild, tmElapsed, accuracy])
        csvOut.flush()

