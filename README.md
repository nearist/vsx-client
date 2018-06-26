

![alt text](http://nearist.sightbox.io/wp-content/uploads/2017/04/nearist.svg)

Nearist greatly accelerates big data searches through a revolutionary new hardware platform specifically engineered to handle the computationally demanding task of performing Nearest Neighbor Search on vector representations of content—enabling your search routines to deliver results several orders of magnitude faster than anything else on the market.


## Structure
This repository contains Nearist's Client API for interacting with Nearist servers remotely, along with example code demonstrating different uses and applications.

- `/python/src` contains code for connecting to Nearist servers and transforming data into a format compatible with Nearist hardware

The rest of the repository provides example code for applications of Nearist hardware:

- `/datasets` contains code for creating datasets 
- `/benchmarks` contains code for running and benchmarking Nearest Neighbor searches using traditional hardware 
- `/python/examples` contains code for running Nearest Neighbor searches using Nearist servers

Please visit the [wiki](https://github.com/nearist/nearist/wiki) for documentation, tutorials, and example applications.


## Preliminaries and Installation

You will have to obtain an API key in order to gain access to Nearist servers. 


Once you have done this, you can install the client library with pip:

`pip install nearist`

or you can clone this repository. If you choose to clone this repository make sure to add the /python/src/ directory to your PYTHONPATH and to update the scripts in `python/examples` to include the API access key and supplied IP address.

## API: At a Glance

Below is the basic usage for starting up the server, loading your data, setting parameters, and returning results.

```python
 1: # Import the Python API
 2: from nearist import Client 
 3: c = Client()
 4: 
 5: # Open the connection to the hardware (IP, port, API key)
 6: c.open("103.210.163.290", 9885, api_key)
 7:
 8: # Load your dataset into memory on Nearist servers
 9: c.load_dataset_file(file_name='dataset.h5', dataset_name='vectors')
10:
11: # Set query mode to k-NN
12: c.set_query_mode(QueryMode.KNN_A)
13:
14: # For k-NN, set 'k' neighbor count
15: c.set_read_count(1)
16:
17: # Set distance metric to L1
18: c.set_distance_mode(DistanceMode.L1)
19: 
20: # Load local vectors to be queried against the dataset
21: query_vectors = load(path='my_local_dataset.h5')[:10] 
22:
23: # Submit query vectors and store results
24: results = c.query(query_vectors)
```

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
