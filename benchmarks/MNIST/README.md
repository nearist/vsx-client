# MNIST Benchmark
This directory contains implementations of the MNIST image classification benchmark using various k-NN implementations in order to compare the accuracy and speed of the implementations. 

Note that the goal here is _not_ to achieve state-of-the-art accuracy on the MNIST benchmark. If that were the goal, we wouldn't be using k-NN classification. Rather, we are using the MNIST dataset to empirically compare choices such as the accuracy of integer versus floating point components, the accuracy of L1 versus L2 distance, the search speed of a Nearist VSX card versus a GPU, etc.

## Obtaining the dataset
In order to run these benchmarks, you will need our pre-calculated feature vectors for the MNIST images. (See `/datasets/MNIST/` for how these feature vectors were created). 


<table>
  <tr>  <th>File</th>               <th>Dimensions</th>    <th>Component Size</th>  <th>Size</th> <th>Description</th>  </tr>
  <tr>  <td>X_train_float.h5</td>   <td>55000 x 1024</td>  <td>float</td>  <td>215 MB</td>  <td>Extracted feature vectors for training images.</td>  </tr>
  <tr>  <td>y_train.h5</td>         <td>55000 x 1</td>     <td>int</td>    <td>0.4 MB</td>  <td>Labels (0 - 9) for training images.</td>  </tr>
  <tr>  <td>X_test_float.h5</td>    <td>10000 x 1024</td>  <td>float</td>  <td>39 MB</td>   <td>Extracted feature vectors for test images.</td>  </tr>
  <tr>  <td>y_test.h5</td>          <td>10000 x 1</td>     <td>int</td>    <td>0.1 MB</td>  <td>Labels (0 - 9) for test images.</td>  </tr>
</table>

## Implementations

### scikit-learn
Script: `run_classification_sklearn.py`

We use the `sklearn.neighbors.KNeighborsClassifier` class from the scikit-learn package to perform brute-force classification of the dataset using a standard laptop or desktop CPU. This implementation is simply intended to provide a baseline for performance.

Example output:

```
=========================
Precision: Floating point
Distance:  Euclidean (L2)
=========================

Loading dataset...
  Training set [55000 x 1024]
  Test set     [10000 x 1024]

Running 10-NN Classification...
  Classification took 15.243sec (656.06 vectors / sec)
  (9841 / 10000) correct - 98.41%
```



<table>
  <tr>  <th>Hardware</th>               <th>Metric</th>    <th>Algorithm</th>  <th>Dataset</th> <th>Time</th> <th>Accuracy</th>  </tr>
  <tr>  <td>Google Cloud 8 vCPUs, 30 GB memory</td>   <td>Floating Point, L2</td>  <td>KNN, k=10</td>  <td>Training set [55000 x 1024]
  Test set     [10000 x 1024]</td>  <td>13.768 seconds</td>  <td>98.41%</td> </tr>
  
  <tr>  <td>Google Cloud 8 vCPUs, 30 GB memory</td>   <td>Floating Point, L1</td>  <td>KNN, k=10</td>  <td>Training set [55000 x 1024]
  Test set     [10000 x 1024]</td>  <td>835.815 seconds</td>  <td>98.46%</td> </tr>
  
  <tr>  <td>Google Cloud 8 vCPUs, 30 GB memory</td>   <td>uint8, L1</td>  <td>KNN, k=10</td>  <td>Training set [55000 x 1024]
  Test set     [10000 x 1024]</td>  <td>826.867 seconds</td>  <td>98.51%</td> </tr>
  
  <tr>  <td>Google Cloud 8 vCPUs, 30 GB memory</td>   <td>uint8, L2</td>  <td>KNN, k=10</td>  <td>Training set [55000 x 1024]
  Test set     [10000 x 1024]</td>  <td>851.544 seconds</td>  <td>98.51%</td> </tr>

  <tr>  <td>Google Cloud 8 vCPUs, 30 GB memory</td>   <td>Floating Point, Cosine</td>  <td>KNN, k=10</td>  <td>Training set [55000 x 1024]
  Test set     [10000 x 1024]</td>  <td>14.743 seconds</td>  <td>98.42%</td> </tr>    
  
  <tr>  <td>Google Cloud 8 vCPUs, 30 GB memory</td>   <td>L2-Normalized Floating point, L1</td>  <td>KNN, k=10</td>  <td>Training set [55000 x 1024]
  Test set     [10000 x 1024]</td>  <td>856.562 seconds</td>  <td>98.45%</td> </tr>   
  
  <tr>  <td>Google Cloud 8 vCPUs, 30 GB memory</td>   <td>L2-Normalized uint8, L1</td>  <td>KNN, k=10</td>  <td>Training set [55000 x 1024] Test set [10000 x 1024]</td>  <td>855.451 seconds</td>  <td>98.46%</td> </tr>   
</table>
