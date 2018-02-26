# -*- coding: utf-8 -*-
"""
This example performs a classification experiment on the MNIST handwritten
digit dataset.

The digit images have already been converted to feature vector representations
using a convolutional neural network.

This script performs classification using the KNeighborsClassifier from the
scikit-learn package.

There are two versions of the dataset here which we perform classification on.
The first is a floating point version of the vectors, and the second version is
an integer dataset.

This script runs the following classification metrics on the dataset:


- L1 with floating point data
- L2 with floating point data
- L1 with integer data
- L2 with integer data
- Cosine distance with floating point data
- L1 with L2-normalized floating point data
- L1 with L2-normalized integer data

"""

import h5py
import sys
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize

# The number of neighbors to use for classification.
k = 10

# Load the labels for the training and test vectors.
# Note: the slice operator at the end tells h5py how much of the matrix to load
# into memory, so [:] loads the whole thing.
y_train = h5py.File('./data/y_train.h5', 'r')['y_train'][:]
y_test = h5py.File('./data/y_test.h5', 'r')['y_test'][:]

###############################################################################
#  Run classification of the test articles with floating point and L2
###############################################################################

print("\n=========================")
print("Precision: Floating point")
print("Distance:  Euclidean (L2)")
print("=========================")

print("\nLoading dataset...")
sys.stdout.flush()

X_train = h5py.File('./data/X_train_float.h5', 'r')['X_train'][:]
X_test = h5py.File('./data/X_test_float.h5', 'r')['X_test'][:]

print("  Training set [%5d x %d]" % (len(X_train), len(X_train[0])))
print("  Test set     [%5d x %d]" % (len(X_test), len(X_test[0])))

print("\nRunning %d-NN Classification..." % k)
sys.stdout.flush()

# Build a k-NN classifier with the following parameters
#  - Majority wins
#  - Euclidean (L2) distance
#  - Brute-force calculation of distances
knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=2)
knn.fit(X_train, y_train)

# Time this step.
t0 = time.time()

# Classify the test vectors.
p = knn.predict(X_test)

# Measure elapsed time.
elapsed = time.time() - t0

# Measure throughput.
vec_per_sec = float(len(y_test)) / elapsed

print("  Classification took %.3fsec (%.2f vectors / sec)" % (elapsed, vec_per_sec))

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

# Calculate the percentage correct.
accuracy = float(numRight) / float(len(y_test)) * 100.0

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), accuracy))

###############################################################################
#  Run classification of the test articles with floating point and L1
###############################################################################

print("\n=========================")
print("Precision: Floating point")
print("Distance:  L1")
print("=========================")

print("\nLoading dataset...")
sys.stdout.flush()

X_train = h5py.File('./data/X_train_float.h5', 'r')['X_train'][:]
X_test = h5py.File('./data/X_test_float.h5', 'r')['X_test'][:]

print("  Training set [%5d x %d]" % (len(X_train), len(X_train[0])))
print("  Test set     [%5d x %d]" % (len(X_test), len(X_test[0])))

print("\nRunning %d-NN Classification..." % k)
sys.stdout.flush()

# Build a k-NN classifier with the following parameters
#  - Majority wins
#  - Euclidean (L2) distance
#  - Brute-force calculation of distances
knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=1)
knn.fit(X_train, y_train)

# Time this step.
t0 = time.time()

# Classify the test vectors.
p = knn.predict(X_test)

# Measure elapsed time.
elapsed = time.time() - t0

# Measure throughput.
vec_per_sec = float(len(y_test)) / elapsed

print("  Classification took %.3fsec (%.2f vectors / sec)" % (elapsed, vec_per_sec))

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

# Calculate the percentage correct.
accuracy = float(numRight) / float(len(y_test)) * 100.0

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), accuracy))


###############################################################################
#  Run classification of the test articles with uint8 and L1
###############################################################################

print("\n=========================")
print("Precision: uint8")
print("Distance:  L1")
print("=========================")

print("\nLoading dataset...")
sys.stdout.flush()

X_train = h5py.File('./data/X_train_uint8.h5', 'r')['X_train'][:]
X_test = h5py.File('./data/X_test_uint8.h5', 'r')['X_test'][:]

print("  Training set [%5d x %d]" % (len(X_train), len(X_train[0])))
print("  Test set     [%5d x %d]" % (len(X_test), len(X_test[0])))

print("\nRunning %d-NN Classification..." % k)
sys.stdout.flush()

# Time this step.
t0 = time.time()

# Build a k-NN classifier with the following parameters
#  - Majority wins
#  - L1 distance (Minkowski with p = 1)
#  - Brute-force calculation of distances.
knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=1)
knn.fit(X_train, y_train)

# Classify the test vectors.
p = knn.predict(X_test)

# Measure elapsed time.
elapsed = time.time() - t0

# Measure throughput.
vec_per_sec = float(len(y_test)) / elapsed

print("  Classification took %.3fsec (%.2f vectors / sec)" % (elapsed, vec_per_sec))

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

# Calculate the percentage correct.
accuracy = float(numRight) / float(len(y_test)) * 100.0

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), accuracy))



###############################################################################
#  Run classification of the test articles with uint8 and L1
###############################################################################

print("\n=========================")
print("Precision: uint8")
print("Distance:  Euclidean (L2)")
print("=========================")

print("\nLoading dataset...")
sys.stdout.flush()

X_train = h5py.File('./data/X_train_uint8.h5', 'r')['X_train'][:]
X_test = h5py.File('./data/X_test_uint8.h5', 'r')['X_test'][:]

print("  Training set [%5d x %d]" % (len(X_train), len(X_train[0])))
print("  Test set     [%5d x %d]" % (len(X_test), len(X_test[0])))

print("\nRunning %d-NN Classification..." % k)
sys.stdout.flush()

# Time this step.
t0 = time.time()

# Build a k-NN classifier with the following parameters
#  - Majority wins
#  - L1 distance (Minkowski with p = 1)
#  - Brute-force calculation of distances.
knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=1)
knn.fit(X_train, y_train)

# Classify the test vectors.
p = knn.predict(X_test)

# Measure elapsed time.
elapsed = time.time() - t0

# Measure throughput.
vec_per_sec = float(len(y_test)) / elapsed

print("  Classification took %.3fsec (%.2f vectors / sec)" % (elapsed, vec_per_sec))

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

# Calculate the percentage correct.
accuracy = float(numRight) / float(len(y_test)) * 100.0

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), accuracy))


###############################################################################
#  Run classification of the test articles with floating point and cosine
###############################################################################

print("\n=========================")
print("Precision: Floating point")
print("Distance:  Cosine")
print("=========================")

print("\nLoading dataset...")
sys.stdout.flush()

X_train = h5py.File('./data/X_train_float.h5', 'r')['X_train'][:]
X_test = h5py.File('./data/X_test_float.h5', 'r')['X_test'][:]

print("  Training set [%5d x %d]" % (len(X_train), len(X_train[0])))
print("  Test set     [%5d x %d]" % (len(X_test), len(X_test[0])))

print("\nRunning %d-NN Classification..." % k)
sys.stdout.flush()

# Build a k-NN classifier with the following parameters
#  - Majority wins
#  - Euclidean (L2) distance
#  - Brute-force calculation of distances
knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', metric='cosine')
knn.fit(X_train, y_train)

# Time this step.
t0 = time.time()

# Classify the test vectors.
p = knn.predict(X_test)

# Measure elapsed time.
elapsed = time.time() - t0

# Measure throughput.
vec_per_sec = float(len(y_test)) / elapsed

print("  Classification took %.3fsec (%.2f vectors / sec)" % (elapsed, vec_per_sec))

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

# Calculate the percentage correct.
accuracy = float(numRight) / float(len(y_test)) * 100.0

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), accuracy))


###############################################################################
#  Run classification of the test articles with L2-normalized floating point
#  and L1
###############################################################################

print("\n=========================")
print("Precision: L2-Normalized Floating point")
print("Distance:  L1")
print("=========================")

print("\nLoading dataset...")
sys.stdout.flush()

X_train = h5py.File('./data/X_train_float.h5', 'r')['X_train'][:]
X_test = h5py.File('./data/X_test_float.h5', 'r')['X_test'][:]

# Apply L2 Normalization to vectors
X_train_norm = normalize(X_train, norm='l2')
X_test_norm = normalize(X_test, norm='l2')

print("  Training set [%5d x %d]" % (len(X_train_norm), len(X_train_norm[0])))
print("  Test set     [%5d x %d]" % (len(X_test_norm), len(X_test_norm[0])))

print("\nRunning %d-NN Classification..." % k)
sys.stdout.flush()

# Build a k-NN classifier with the following parameters
#  - Majority wins
#  - Euclidean (L2) distance
#  - Brute-force calculation of distances
knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=1)
knn.fit(X_train_norm, y_train)

# Time this step.
t0 = time.time()

# Classify the test vectors.
p = knn.predict(X_test_norm)

# Measure elapsed time.
elapsed = time.time() - t0

# Measure throughput.
vec_per_sec = float(len(y_test)) / elapsed

print("  Classification took %.3fsec (%.2f vectors / sec)" % (elapsed, vec_per_sec))

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

# Calculate the percentage correct.
accuracy = float(numRight) / float(len(y_test)) * 100.0

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), accuracy))


###############################################################################
#  Run classification of the test articles with L2-normalized integer values
#  and L1
###############################################################################

print("\n=========================")
print("Precision: L2-Normalized Integer")
print("Distance:  L1")
print("=========================")

print("\nLoading dataset...")
sys.stdout.flush()

X_train = h5py.File('./data/X_train_uint8.h5', 'r')['X_train'][:]
X_test = h5py.File('./data/X_test_uint8.h5', 'r')['X_test'][:]

# Apply L2 Normalization to vectors
X_train_norm = normalize(X_train, norm='l2')
X_test_norm = normalize(X_test, norm='l2')

print("  Training set [%5d x %d]" % (len(X_train_norm), len(X_train_norm[0])))
print("  Test set     [%5d x %d]" % (len(X_test_norm), len(X_test_norm[0])))

print("\nRunning %d-NN Classification..." % k)
sys.stdout.flush()

# Build a k-NN classifier with the following parameters
#  - Majority wins
#  - Euclidean (L2) distance
#  - Brute-force calculation of distances
knn = KNeighborsClassifier(n_neighbors=k, algorithm='brute', p=1)
knn.fit(X_train_norm, y_train)

# Time this step.
t0 = time.time()

# Classify the test vectors.
p = knn.predict(X_test_norm)

# Measure elapsed time.
elapsed = time.time() - t0

# Measure throughput.
vec_per_sec = float(len(y_test)) / elapsed

print("  Classification took %.3fsec (%.2f vectors / sec)" % (elapsed, vec_per_sec))

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

# Calculate the percentage correct.
accuracy = float(numRight) / float(len(y_test)) * 100.0

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), accuracy))
