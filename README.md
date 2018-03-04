

![alt text](http://nearist.sightbox.io/wp-content/uploads/2017/04/nearist.svg)

Nearist greatly accelerates big data searches through a revolutionary new hardware platform specifically engineered to handle the computationally demanding task of performing Nearest Neighbor Search on vector representations of content—enabling your search routines to deliver results several orders of magnitude faster than anything else on the market.


## Structure
This repository contains Nearist's Client API for interacting with Nearist servers remotely, along with example code demonstrating different uses and applications.

- `/python/src` contains code for connecting to Nearist servers and transforming data into a format compatible with Nearist hardware

The rest of the repository provides example code for applications of Nearist hardware:

- `/datasets` contains code for creating datasets 
- `/benchmarks` contains code for running and benchmarking Nearest Neighbor searches using traditional hardware 
- `/python/examples` contains code for running Nearest Neighbor searches using Nearist servers



## Preliminaries

You will have to obtain an API key in order to gain access to Nearist servers. 

Once you have done this, make sure to add the /python/src/ directory to your PYTHONPATH and to update the scripts in `python/examples` to include the API access key and supplied IP address.

## Example: MNIST

To show how Nearist works, we will run through the provided code to perform Nearest Neighbors classification of the MNIST dataset. 

The `/datasets/MNIST/` directory contains instructions and code for generating the data that we will run on Nearist hardware. (Alternatively, you can skip this step and download the generated datasets directly from [here](https://drive.google.com/drive/folders/1tr-q_uhg6PVuQKIwnLDRMtRsrG2oyS8C)). The README explains that we first train a convolutional neural network on the MNIST image classification task, and then use the trained network to transform raw images into 1024-dimensional vectors. We then convert these vectors of float data into vectors of integer data for use on Nearist hardware. This is accomplished by running:

`python mnist_feature_extraction.py` 

and then: 

`python map_to_integers.py`

You should now have the h5 datasets stored in your `data` folder.

The `/benchmarks/MNIST/` directory contains instructions and code for testing the accuracy and time of a Nearest Neighbors Classifier on CPU and GPU hardware under a variety of distance metrics (L1, L2, etc.), datatypes (uint8, float, etc.), and software packages (annoy, scikit-learn, etc.). For example, you can test the performance of a scikit-learn classifier under a variety of metrics by running:

`python run_classification_metrics.py`

The `/python/examples/MNIST/` directory contains instructions and code for running the Nearest Neighbors Classifier on Nearist hardware. After updating the script with the details of your API access key, run:

`python run_classification.py`

This will return the elapsed time and accuracy of the Nearest Neighbors Classifier using Nearist hardware.