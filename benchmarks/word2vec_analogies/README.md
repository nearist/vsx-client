# Word2Vec Analogies Benchmark
This directory contains implementations of the word2vec analogies task using various k-NN implementations in order to compare the accuracy and speed of the implementations. 

Note that the goal here is _not_ to achieve state-of-the-art accuracy. If that were the goal, we wouldn't be using k-NN classification. Rather, we are using the dataset to empirically compare choices such as the accuracy of integer versus floating point components, the accuracy of L1 versus L2 distance, the search speed of a Nearist VSX card versus a GPU, etc.

## Obtaining the dataset
In order to run these benchmarks, you will need our pre-calculated feature vectors dataset for analogies. (See `/datasets/word2vec_analogies/` for how these feature vectors were created). 

<table>
  <tr>  <th>Dataset</th>               <th>Dimensions</th>    <th>Component Size</th>   <th>Description</th>  </tr>
 
  <tr>  <td>query_vecs</td>   <td>~20,000 x 300</td>  <td>int</td>  <td>Computed query vectors with integer components.</td>    </tr>
  
  <tr>  <td>word_vecs</td>   <td>~3M x 300</td>  <td>int</td>  <td>Word vectors from model with integer components.</td>    </tr>
  
  <tr>  <td>abc_i</td>   <td>~20,000 x 3</td>  <td>string</td>  <td>First three words of each analogy.</td>    </tr>  
  
  <tr>  <td>d_i</td>   <td>~20,000 x 1</td>  <td>string</td>  <td>Solution word for each analogy.</td>    </tr>  
</table>


## Implementations

### scikit-learn
Script: `run_classification_sklearn.py`

We use the `sklearn.neighbors.NearestNeighbors` class from the scikit-learn package to perform brute-force retrieval of the nearest neighbor to our query vector using a standard laptop or desktop CPU. This implementation is simply intended to provide a baseline for performance.

Example output:

```
Memory usage at start 0.00GB
Loading the dataset file...
Loading dataset took 0.00GB extra
Creating Neighbors model took 0.00GB extra
Performing all 19544  queries...
Query   128 / 19544 (1%) Time Remaining:~113 min....
Query   256 / 19544 (1%) Time Remaining:~105 min....
Query   384 / 19544 (2%) Time Remaining:~101 min....
...
...
Query 19200 / 19544 (98%) Time Remaining:~2 min....
Query 19328 / 19544 (99%) Time Remaining:~1 min....
Query 19456 / 19544 (100%) Time Remaining:~0 min....
Done, 5685 seconds
Final accuracy 73.66% (14396 / 19544)
```




<table>
  <tr>  <th>Hardware</th>               <th>Metric</th>    <th>Algorithm</th>  <th>Time</th> <th>Accuracy</th>  </tr>
  <tr>  <td>Google Cloud 8 vCPUs, 30 GB memory</td>   <td>uint8, Cosine</td>  <td>KNN, k=1</td>    <td>113 minutes</td>  <td>73.66%</td> </tr>
  
 
</table>
