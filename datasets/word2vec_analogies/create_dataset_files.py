"""
This script extracts the word vectors from Google's trained model of 3M words.

It creates a dataset for the analogies test also created by Google.

Basics for using the gensim word2vec model:
  - model.index2word is a list of all the word strings.
  - model.syn0 is the matrix of word vectors [3M x 300]
     - model.syn0norm is the word vectors normalized. You must call init_sims
       first, though.
  - Look up a word vector with just model['hello']
  - Look up a word index with model.vocab['hello'].index

"""


import h5py
import gensim
import numpy as np
import sys

from sys import argv

##############################################################################
#  Load the model
##############################################################################

init = False
if init:   
    # Get the paths to the original Google model file and to the test analogies
    # from the command line.
    script_name, model_path, analogies_path = argv
    
    print ('Loading word2vec model...')
    sys.stdout.flush()
    
    # Load Google's pre-trained Word2Vec model.
    model = gensim.models.Word2Vec.load_word2vec_format(model_path, binary=True)  
    
    print ('Normalizing all word vectors...')
    sys.stdout.flush()
    
    # Normalize all word vectors. Replace the original vectors to save memory.
    model.init_sims(replace=True)

##############################################################################
#  Parse Analogies
##############################################################################

print ('Reading in analogies...')
sys.stdout.flush()

# Count the number of analogies in the file.
num_analogies = 0

with open(analogies_path, 'rb') as f:
    for line in f:
    
        # Skip over the section headers, which start with ": "
        if line[0] == ':':
            continue
        
        num_analogies += 1

print ('    There are', num_analogies, 'analogies.')

# Read in the analogies.

# Create a matrix to hold the word indeces for all of the analogies.    
analogies = np.zeros((num_analogies, 4), dtype=int)

with open(analogies_path, 'rb') as f:

    analogy_num = 0
        
    for line in f:
    
        # Skip over the section headers, which start with ": "
        if line[0] == ':':
            continue
    
        line = line.rstrip()    
    
        # Split the analogy into it's four words
        words = line.split(' ')
        
        assert(len(words) == 4)
        
        # Look up the index of the words and store them.
        for i in range(0, 4):                    
            analogies[analogy_num, i] = model.vocab[words[i]].index

        # Increment the row number.
        analogy_num += 1            

print ('Validating indeces...')
sys.stdout.flush()

# Verify no entries are zero.        
assert(np.count_nonzero(analogies) == (analogies.shape[0] * analogies.shape[1]))

##############################################################################
#  Precompute Query Vectors
##############################################################################

print ('Computing analogy query vectors...')
sys.stdout.flush()

# Create a matrix to hold all of the query vectors.
query_vecs = np.zeros((num_analogies, model.syn0.shape[1]))

# For each of the analogies...
for i in range(0, num_analogies):

    a = model.syn0norm[analogies[i, 0], :]
    b = model.syn0norm[analogies[i, 1], :]
    c = model.syn0norm[analogies[i, 2], :]

    # Construct the query vector as the average of  b, c, and -a
    # Use the normalized version of the vectors.
    query_vec = (b + c - a) / 3.0

    # Normalize the query vector.
    query_vec = query_vec / np.linalg.norm(query_vec)

    query_vecs[i, :] = query_vec

##############################################################################
#  Save dataset to HDF5 file.
##############################################################################

print ('Writing dataset to disk...')
sys.stdout.flush()

h5f = h5py.File('./data/Google_word2vec_analogies.h5', 'w')

h5f.create_dataset(name='query_vecs', data=query_vecs)
h5f.create_dataset(name='word_vecs', data=model.syn0norm)
#h5f.create_dataset(name='words', data=model.index2word)
h5f.create_dataset(name='abc_i', data=analogies[:, 0:3])
h5f.create_dataset(name='d_i', data=analogies[:, 3])

h5f.close()