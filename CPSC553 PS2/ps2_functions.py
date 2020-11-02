# ps2_functions.py
# Completed by Joanna Chen, Fall 2020
# Jay S. Stanley III, Yale University, Fall 2018
# CPSC 453 -- Problem Set 2
#
# This script contains functions for implementing graph clustering and signal processing.
#

import numpy as np
import codecs
import json
import scipy
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.linalg import eigh
from sklearn.cluster import KMeans

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        my_array    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


def gaussian_kernel(X, kernel_type="gaussian", sigma=3.0, k=5):
    """gaussian_kernel: Build an adjacency matrix for data using a Gaussian kernel
    Args:
        X (N x d np.ndarray): Input data
        kernel_type: "gaussian" or "adaptive". Controls bandwidth
        sigma (float): Scalar kernel bandwidth
        k (integer): nearest neighbor kernel bandwidth
    Returns:
        W (N x N np.ndarray): Weight/adjacency matrix induced from X
    """
    _g = "gaussian"
    _a = "adaptive"

    kernel_type = kernel_type.lower()
    D = squareform(pdist(X))
    if kernel_type == "gaussian":  # gaussian bandwidth checking
        print("fixed bandwidth specified")

        if not all([type(sigma) is float, sigma > 0]):  # [float, positive]
            print("invalid gaussian bandwidth, using sigma = max(min(D)) as bandwidth")
            D_find = D + np.eye(np.size(D, 1)) * 1e15
            sigma = np.max(np.min(D_find, 1))
            del D_find
        sigma = np.ones(np.size(D, 1)) * sigma
    elif kernel_type == "adaptive":  # adaptive bandwidth
        print("adaptive bandwidth specified")

        # [integer, positive, less than the total samples]
        if not all([type(k) is int, k > 0, k < np.size(D, 1)]):
            print("invalid adaptive bandwidth, using k=5 as bandwidth")
            k = 5

        knnDST = np.sort(D, axis=1)  # sorted neighbor distances
        sigma = knnDST[:, k]  # k-nn neighbor. 0 is self.
        del knnDST
    else:
        raise ValueError

    W = ((D**2) / sigma[:, np.newaxis]**2).T
    W = np.exp(-1 * (W))
    W = (W + W.T) / 2  # symmetrize
    W = W - np.eye(W.shape[0])  # remove the diagonal
    return W


# BEGIN PS2 FUNCTIONS


def sbm(N, k, pij, pii, sigma):
    """sbm: Construct a stochastic block model

    Args:
        N (integer): Graph size
        k (integer): Number of clusters
        pij (float): Probability of intercluster edges
        pii (float): probability of intracluster edges

    Returns:
        A (numpy.array): Adjacency Matrix
        gt (numpy.array): Ground truth cluster labels
        coords(numpy.array): plotting coordinates for the sbm
    """
     # create A
    A = np.zeros((N, N))
    temp_p = np.random.uniform(0, 1, int(N*(N+1)//2))
    upper_triangular_indices = np.triu_indices(N)
    A[upper_triangular_indices] = temp_p

    # create coords and initialize the values taking advantage of trigonometic properties of sin and cos
    partition = np.linspace(0, 2*np.pi * k / (k + 1), k)
    coords = np.random.normal(loc = 0, scale = sigma, size = (N, 2))
    x, y = np.sin(partition), np.cos(partition)
    x_y = np.column_stack((x, y))

    # set the cluster beginning and end points
    cluster_partition = np.linspace(0, N, k+1)
    cluster_partition = np.ceil(cluster_partition).astype(int)
    cluster_end = zip(cluster_partition[:-1], cluster_partition[1:])

    # set the values iterating over the cluster index points
    for i, (start, end) in enumerate(cluster_end):
        coords[start:end] += x_y[i]
        A[start:end, start:end] = A[start:end, start:end] < pii
        A[start:end, end:] = A[start:end, end:] < pij

    # set the lower triangular indices of A and set to int
    lower_triangular_indices = np.tril_indices(N, -1)
    A[lower_triangular_indices] = A.T[lower_triangular_indices]
    A = A.astype(int)
    
    # create the ground truth array based on the instruction; having unequal clusters
    val = 0
    gt = []
    for i in range(N):
      gt.append(val)
      val += 1
      if val == k:
        val = 0

    gt.sort()
    gt = np.array(gt)
    
    return A, gt, coords


def L(A, normalized=True):
    """L: compute a graph laplacian

    Args:
        A (N x N np.ndarray): Adjacency matrix of graph
        normalized (bool, optional): Normalized or combinatorial Laplacian

    Returns:
        L (N x N np.ndarray): graph Laplacian
    """
    D = np.diag(A.sum(axis=1)) # degree matrix of A = sum of A-th rows
    semi_D = scipy.linalg.fractional_matrix_power(D, -1/2) # power of (-1/2)
    
    if normalized == True:
        L = np.eye(A.shape[0]) - np.matmul(semi_D, np.matmul(A,semi_D))

    elif normalized == False:
        L = D - A
    
    return L


def compute_fourier_basis(L):
    """compute_fourier_basis: Laplacian Diagonalization

    Args:
        L (N x N np.ndarray): graph Laplacian

    Returns:
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    """
    e, psi = np.linalg.eigh(L)
    return e, psi


def gft(s, psi):
    """gft: Graph Fourier Transform (GFT)

    Args:
        s (N x d np.ndarray): Matrix of graph signals.  Each column is a signal.
        psi (N x N np.ndarray): graph Laplacian eigenvectors
    Returns:
        s_hat (N x d np.ndarray): GFT of the data
    """
    s_hat = np.matmul(psi.T,s)
    return s_hat


def filterbank_matrix(psi, e, h):
    """filterbank_matrix: build a filter matrix using the input filter h

    Args:
        psi (N x N np.ndarray): graph Laplacian eigenvectors
        e (N x 1 np.ndarray): graph Laplacian eigenvalues
        h (function handle): A function that takes in eigenvalues
        and returns values in the interval (0,1)

    Returns:
        H (N x N np.ndarray): Filter matrix that can be used in the form
        filtered_s = H@s
    """
    # if h == 'sigmoid':
    #     fx_h = lambda x: 1/(1+np.exp(-x)) 
    # elif h == 'sine':
    #     fx_h = lambda x: np.sin(x) 
    # elif h == 'cosine':
    #     fx_h = lambda x: np.cos(x)

    if h == 'cutoff':
        c = 0.8
        for i in np.arange(np.size(e)):
            if e[i] < c:
                e[i] = 1
            else:
                e[i] = 0
        fx_h = lambda t: t
        
    fx_h = np.vectorize(fx_h)
    H = np.matmul(psi, np.matmul(np.diagflat(fx_h(e)), psi.T))
    return H


def kmeans(X, k, nrep=5, itermax=300):
    """kmeans: cluster data into k partitions

    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition
        nrep (int): Number of repetitions to average for final clustering 
        itermax (int): Number of iterations to perform before terminating
    Returns:
        labels (n x 1 np.ndarray): Cluster labels assigned by kmeans
    """
    init = kmeans_plusplus(X, k)  # find your initial centroids
    # perform kmeans
    return labels


def kmeans_plusplus(X, k):
    """kmeans_plusplus: initialization algorithm for kmeans
    Args:
        X (n x d np.ndarray): input data, rows = points, cols = dimensions
        k (int): Number of clusters to partition

    Returns:
        centroids (k x d np.ndarray): centroids for initializing k-means
    """
    return centroids


def SC(L, k, psi=None, nrep=5, itermax=300, sklearn=False):
    """SC: Perform spectral clustering 
            via the Ng method
    Args:
        L (np.ndarray): Normalized graph Laplacian
        k (integer): number of clusters to compute
        nrep (int): Number of repetitions to average for final clustering
        itermax (int): Number of iterations to perform before terminating
        sklearn (boolean): Flag to use sklearn kmeans to test your algorithm
    Returns:
        labels (N x 1 np.array): Learned cluster labels
    """
    if psi is None:
        # compute the first k elements of the Fourier basis
        # use scipy.linalg.eigh
        pass
    else:  # just grab the first k eigenvectors
        psi_k = psi[:, :k]

    # normalize your eigenvector rows

    if sklearn:
        labels = KMeans(n_clusters=k, n_init=nrep,
                        max_iter=itermax).fit_predict(psi_norm)
    else:
        pass
        # your algorithm here

    return labels

