# K-Means Clusterization "from scratch"

**_K-Means_** clustering is a method of vector quantization, originally from signal processing, that aims to partition **N** observations into **K** clusters in which each observation belongs to the cluster with the nearest mean (cluster centers or cluster centroid), serving as a prototype of the cluster [[WIKIPEDIA](https://en.wikipedia.org/wiki/K-means_clustering#Description)]

### Formula:
![k_means_formula](https://i.postimg.cc/cJc1NHyB/k-means.png)

**d** - metric, **i** - _i-th_ element, **a** - center of cluster, which have element on _j_ iteration. 

We have an array of observations (objects), each of which has certain values on a number of grounds. According to these values, the object is located in multidimensional space.

## K-Means Algorithm

1. The researcher determines the number of clusters that need to be formed.
2. **K** observations are randomly selected, which at this step are considered the centers of clusters.
3. Each observation is "attributed" to one of **N** clusters - the one to which the distance is shortest.
4. The new center of each cluster is calculated as an element, the features of which are calculated as the arithmetic mean of the features of the objects included in this cluster.
5. This number of iterations occurs (steps 3-4 are repeated) until the cluster centers become stable (that is, each iteration will have the same objects in each cluster), the variance within the cluster will be minimized, and between the clusters - maximized.


Code realization
------

I will be using [Python](https://www.python.org/) for code realization.
#### Libraries:
1. [Pandas](https://pandas.pydata.org/)
2. [NumPy](https://numpy.org/)
3. [Matplotlib](https://matplotlib.org/)

**You need to create metrics to use the algorithm. Here is few examples.**

**Euclidean distance**
```python
def euklid_dist(dot1, dot2):
    return np.array(np.sqrt(np.subtract(dot1, dot2)**2).sum(axis=1))
```

**Manhattan distance**
```python
def manhattan_dist(dot1, dot2):
    return np.array(abs(np.subtract(dot1, dot2)).sum(axis=1))
```

**Chebyshev distance**
```python
def chebishev_dist(dot1, dot2):
    return max(abs(np.subtract(dot1, dot2)))
```

**Now, load data.** 
I will be using dmd.csv dataset, which contains a lot of features. Let's pick 3 of them:
```python
dataframe = df[['h','pk', 'ck']]
```
Than we need to handle data, because we have missings as NaN.
Using functions _improve_data_ and _normalize_data_ can help us avoid missings and normalize data.

### K-Means function

Next step - initialize centroids. I initialize them randomly in _initialize_centroids_ function as pandas dataset.
Then I opening while loop and start seqrching for closest centroids for each dot in _closest_centroids_ function.
Then I'm calculating the mean coordinates for each group of dots and putting centroids on this coordinates.

This while loop repeats until previous dots equals to current dots.
```python
def k_means(df_tmp, K):
    c = initialize_centroids(df_tmp, K) # random initialization
    print(c)
    prev = [] # create array for comparision
    while True:
        closest = closest_centroids(df_tmp.iloc[:,:3], c)
        
        if np.array_equal(closest, prev): # if cluster dots on prev iteration equal to current loop
            break
            
        df_tmp['dots'] = closest
        for i in range(K):
            c.loc[i] = np.array((df_tmp.loc[df_tmp['dots']==i].iloc[:, :3]).mean()) # move centroids
        prev = np.copy(closest)
        
        dff = pd.DataFrame(data=df_tmp)
        dff['dots'] = prev
    return dff, c
```

Let's test algorithm on dataset
------

```python
dataframe = improve_data(dataframe)
dataframe = normalize_data(dataframe)

K = 3
model = k_means(dataframe, K)
build3d(model)
```
Result:

![img_3_clust](https://i.postimg.cc/W4HhFzz6/3-clust.png)
