"""
Apply logarithmic transform of float values to uint8 values, save to disk, and plot distributions
of data before and after transformation.
"""


from __future__ import division
import matplotlib
import numpy as np
import h5py
import Transforms
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


###############################################################################
# Load, transform, and save data
###############################################################################

# Load the floating point version of the train dataset.
h5f = h5py.File('./data/X_train_float.h5', 'a')
X_train_float = h5f['X_train'][...]
h5f.close()

# Load the floating point version of the test dataset.
h5f = h5py.File('./data/X_test_float.h5', 'a')
X_test_float = h5f['X_test'][...]
h5f.close()

# Apply transformation

# Instantiate Transforms class
trans = Transforms.Transforms(np.uint8)

# Learn the train set max value
trans.learn_logarithmic(X_train_float)

# Convert the train set to uint8
X_train_int = trans.apply_logarithmic(X_train_float)

# Use the same denominator value to convert the test set to uint8
X_test_int = trans.apply_logarithmic(X_test_float)


# Save the integer versions of the train and test set to HDF5 files.
h5f = h5py.File('./data/X_train_uint8.h5', 'w')
h5f.create_dataset(name='X_train', data=X_train_int)
h5f.close()

h5f = h5py.File('./data/X_test_uint8.h5', 'w')
h5f.create_dataset(name='X_test', data=X_test_int)
h5f.close()



###############################################################################
# Plot data
###############################################################################

# Sample size used in plotting
sample_size = 5000

# Get percentage of non-zero values in float dataset
float_X_flat = X_train_float.flatten()
float_non_zero_percentage = float_X_flat[np.where( float_X_flat > 0)].shape[0] / float_X_flat.shape[0]
# Get sample of non-zero float values for plotting
float_sample = np.random.choice(float_X_flat, sample_size)
float_sample_non_zero = float_sample[np.where( float_sample > 0 )]


# Get percentage of non-zero integer values in dataset
int_X_flat = X_train_int.flatten()
int_non_zero_percentage = int_X_flat[np.where( int_X_flat > 0)].shape[0] / int_X_flat.shape[0]
# Get sample of non-zero integer values for plotting
int_sample = np.random.choice(int_X_flat, sample_size)
int_sample_non_zero = int_sample[np.where( int_sample > 0 )]


# Set subplots
f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(8,30))
plt.subplots_adjust(hspace=.6)

# Plot distribution of float values
sns.distplot(float_sample_non_zero, bins=50, ax=ax1)
ax1.set_title("Histogram of sampled non-zero floats\n (%i percent of data is non-zero)" % int(float_non_zero_percentage * 100), size=13)
ax1.set_xlabel("Float value")
ax1.set_ylabel("Density")

# Plot distribution of integer values
sns.distplot(int_sample_non_zero, bins=50, ax=ax2)
ax2.set_title("Histogram of sampled non-zero integers\n (%i percent of data is non-zero)" % int(int_non_zero_percentage * 100), size=13)
ax2.set_xlabel("Integer value")
ax2.set_ylabel("Density")

# Plot transformation values
sns.regplot(X_train_float.flatten()[:sample_size], X_train_int.flatten()[:sample_size], scatter=True, fit_reg=False, ax=ax3)
ax3.set_title("Scatterplot of sampled mappings from float to integer", size=13)
ax3.set_xlabel("Float value")
ax3.set_ylabel("Integer value")

plt.show()
