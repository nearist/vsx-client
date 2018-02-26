# -*- coding: utf-8 -*-
"""
This script maps the floating point values in the word vectors dataset to
integers.

First, it takes a sample of the dataset and plots a histogram of all the
values in the matrix. This reveals a roughly normal distribution of values.

The normal distribution motivates us to try a logistic mapping function. We
calculate the saturation point for the values as 8 x the standard deviation
of the values. This means that any point farther than 8 standard deviations
from the mean will be mapped to 0 (for negative numbers) or 255 (for positive
numbers).

Created on Wed Sep 27 20:31:06 2017

@author: Chris
"""
from __future__ import division
import matplotlib.pyplot as plt
import h5py
import sys
import numpy as np
import Transforms
import seaborn as sns
sns.set()


###########################################################################
#    Load the dataset
###########################################################################

print 'Loading the dataset file...'
sys.stdout.flush()

h5f = h5py.File('./data/Google_word2vec_analogies.h5', 'r')

# How many vectors to use for things like plotting the data distribution.
sample_size = 30000

###########################################################################
#    Apply integer mapping
###########################################################################

print 'Applying integer mapping to dataset...'
sys.stdout.flush()

# The data shows a normal distribution, so we're going to use a
# logistic mapping function.

# This time, load the entire dataset into memory.
word_vecs = h5f['word_vecs'][:]

# Calculate the mean using a sample of the vectors.
center = np.mean(word_vecs[0:sample_size].flatten())

# Verify that the mean is roughly zero.
assert(abs(center - 0.0) < 0.01)

# Learn the parameters for the logistic mapping function.
# The mapping function will saturate at 8 standard deviations, and will convert
# the values to 8-bit integers.
tf = Transforms.Transforms(np.uint8)

standard_deviations = 8
tf.learn_logistic(word_vecs, num_std=standard_deviations)

# Apply logistic mapping to the dataset vectors.
word_vecs_int = tf.apply_logistic(word_vecs)

# Load the query vectors into memory and map the values.
query_vecs = h5f['query_vecs'][:]

# Apply logistic mapping to the query vectors.
query_vecs_int = tf.apply_logistic(query_vecs)

# Take a small subset of the dataset and plot a histogram of the values.

abc_i = h5f['abc_i'][:]
d_i = h5f['d_i'][:]

# Close the floating point dataset file.
h5f.close()


###########################################################################
#   Save transformed data
###########################################################################

h5f_int = h5py.File('./data/Google_word2vec_analogies_uint8_%dstd.h5' % standard_deviations, 'w')

h5f_int.create_dataset(name='query_vecs', data=query_vecs_int, dtype='uint8')
h5f_int.create_dataset(name='word_vecs', data=word_vecs_int, dtype='uint8')
h5f_int.create_dataset(name='abc_i', data=abc_i)
h5f_int.create_dataset(name='d_i', data=d_i)

h5f_int.close()

###############################################################################
#   Plot data
###############################################################################

# Set subplots
f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8,30))
plt.subplots_adjust(hspace=.6)

# Plot distribution of float values
word_vecs = h5f['word_vecs'][0:sample_size]
sns.distplot(word_vecs.flatten(), bins=100, ax=ax1)
ax1.set_title("Histogram of float data", size=13)
ax1.set_xlabel("Float value")
ax1.set_ylabel("Density")

# Plot distribution of integer values
sns.distplot(word_vecs_int[0:sample_size].flatten(), bins=255, ax=ax2)
ax2.set_title("Histogram of integer data", size=13)
ax2.set_xlabel("Integer value")
ax2.set_ylabel("Density")

# Plot transformation values
sns.regplot(word_vecs.flatten(), word_vecs_int[0:sample_size].flatten()], scatter=True, fit_reg=False, ax=ax3)
ax3.set_title("Scatterplot of sampled mappings from float to integer", size=13)
ax3.set_xlabel("Float value")
ax3.set_ylabel("Integer value")

plt.show()
