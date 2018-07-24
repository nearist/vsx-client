# MNIST Hand-Written Digit Classification Example

In this example, we are running one of the most classic machine learning benchmarks--the MNIST benchmark. 

## Explanation

### The MNIST Benchmark
The [MNIST benchmark](http://yann.lecun.com/exdb/mnist/) is an image classification task where the objective is to correctly identify which of the digits 0 - 9 appears in a given test image. The original application was automatic recognition of the hand-written zipcode on envelopes for the postal service. There are 60,000 training images and 10,000 test images. The original images are all 27 x 27 pixels, black and white. The digits are all roughly the same size and centered within the image.

### Feature Extraction
You can achieve reasonable accuracy on this benchmark using the raw images alone, but a more modern approach is to apply some feature extraction to the images prior to classification. We have used an example convolutional neural network provided by TensorFlow to extract a 1,024 component feature vector for each of the images. See the /datasets/MNIST/ directory for source code and more details on the feature extraction.

### Classification
We'll be using k-Nearest Neighbor classification to classify each of the 10,000 test images. 

### Accuracy
k-NN doesn't achieve quite as good of accuracy as the original convolutional neural network (97.3% for k-NN vs. 99.2% for the CNN).

k-NN provides other advantages, however, which make it more suitable in other image similarity applications. For instance:

* Image search - When trying to match an input image against a collection of images (i.e., "Image Search" as opposed to "Image Classification"), k-NN is the logical approach.
* Intuitive results - k-NN classification provides easily explainable results--you can gain insight into the prediction by looking at the training samples which most closely matched the input.
* Online learning - Because k-NN doesn't require a training step, the reference dataset can grow in real time. This means you don't need to do any offline re-training and updating of a model in order to cover new examples. 

## Running the Example
You'll be running the example using a remote connection to a server appliance in our datacenter.

You'll need the following in order to run the exmaple:

* The example dataset files (see below)
* The Nearist classes on your Python Path.
* The API access key generated for your user account.
* An active rental session and the IP address of your reserved Nearist server appliance.

_Make sure to update your copy of the example code with your API key and the provided IP address before running. Additionally, be sure to add /python/src/ to your PYTHONPATH._

### Required Files
In order to run the example `run_classification.py`, you will need the files listed below, which are available to download from [here](https://drive.google.com/drive/folders/1tr-q_uhg6PVuQKIwnLDRMtRsrG2oyS8C). 

You can also look in `/datasets/MNIST/` to see how this dataset was generated.

Note that the `X_train_int.h5` file is already stored on the appliance server (since it's one of our examples), so you don't need to download it to your laptop. Instead, the example loads it remotely for you.

<table>
  <tr>  <th>File</th>               <th>Dimensions</th>    <th>Component Size</th>  <th>Size</th> <th>Description</th>  </tr>
  <tr>  <td>y_train.h5</td>         <td>55000 x 1</td>     <td>int</td>    <td>0.2 MB</td>  <td>Labels (0 - 9) for training images.</td>  </tr>
  <tr>  <td>y_test.h5</td>          <td>10000 x 1</td>     <td>int</td>    <td>0.1 MB</td>  <td>Labels (0 - 9) for test images.</td>  </tr>
  <tr>  <td>X_train_uint8.h5</td>     <td>55000 x 1024</td>  <td>int</td>  <td>56.3 MB</td>  <td>Training vectors with integer components.</td>  </tr>
  <tr>  <td>X_test_uint8.h5</td>      <td>10000 x 1024</td>  <td>int</td>  <td>10.2 MB</td>  <td>Test vectors with integer components.</td>  </tr>
</table>

