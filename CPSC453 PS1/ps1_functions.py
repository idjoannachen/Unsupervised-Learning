# ps1_functions.py
# Skeleton file by Chris Harshaw, Yale University, Fall 2017
# Adapted by Jay Stanley, Yale University, Fall 2018
# Adapted by Scott Gigante, Yale University, Fall 2019
# CPSC 553 -- Problem Set 1
#
# This script contains uncompleted functions for implementing diffusion maps.
#
# NOTE: please keep the variable names that I have put here, as it makes grading easier.

# import required libraries
import numpy as np
import codecs, json
import scipy
from scipy import linalg

##############################
# Predefined functions
##############################

def load_json_files(file_path):
    '''
    Loads data from a json file

    Inputs:
        file_path   the path of the .json file that you want to read in

    Outputs:
        json_data    this is a numpy array if data is numeric, it's a list if it's a string

    '''

    #  load data from json file
    with codecs.open(file_path, 'r', encoding='utf-8') as handle:
        json_data = json.loads(handle.read())

    # if a string, then returns list of strings
    if not isinstance(json_data[0], str):
        # otherwise, it's assumed to be numeric and returns numpy array
        json_data = np.array(json_data)

    return json_data


##############################
# Skeleton code (fill these in)
##############################


def compute_distances(X):
    '''
    Constructs a distance matrix from data set, assumes Euclidean distance

    Inputs:
        X       a numpy array of size n x p holding the data set (n observations, p features)

    Outputs:
        D       a numpy array of size n x n containing the euclidean distances between points
    '''
    n = len(X)
    D = np.zeros([n, n])

    for i in range(n):
        for j in range(n):
            D[i, j] = np.sqrt(sum((X[i] - X[j])**2)) # Euclidean distance

    # return distance matrix
    return D


def compute_affinity_matrix(D, kernel_type, sigma=None, k=None):
    '''
    Construct an affinity matrix from a distance matrix via gaussian kernel.

    Inputs:
        D               a numpy array of size n x n containing the distances between points
        kernel_type     a string, either "gaussian" or "adaptive".
                            If kernel_type = "gaussian", then sigma must be a positive number
                            If kernel_type = "adaptive", then k must be a positive integer
        sigma           the non-adaptive gaussian kernel parameter
        k               the adaptive kernel parameter

    Outputs:
        W       a numpy array of size n x n that is the affinity matrix

    Examples:
        W = compute_affinity_matrix(D, "adaptive", None, 5)
        W = compute_affinity_matrix(D, "gaussian", 3, None)
    '''
    if kernel_type=='gaussian':
      if sigma <= 0:
        raise ValueError('sigma must be a positive number')
      else:
        W = np.exp(-D*D/(sigma**2))
             
    if kernel_type=='adaptive':
        if k < 0 or type(k) != int:
          raise ValueError('k must be a positive integer')
        else:
          W = np.zeros(shape = np.shape(D))
          for i in range(len(D)):
            sigma_x_i = np.sort(D[i])[k] # we use k instead of k-1 because self doesn't count as self's neighbor
            
            for j in range(len(D)):
              # acquire distance from x_i to x_j, k nearest neighbor for the ith row and jth row
              dist = D[i, j]
              
              sigma_x_j = np.sort(D[j])[k]

              # compute adaptive knn kernel based on the formula
              first_exp_term = np.exp((-1 * (dist**2)) / (sigma_x_i ** 2))
              second_exp_term = np.exp((-1 * (dist**2)) / (sigma_x_j ** 2))
              W[i, j] = 0.5 * (first_exp_term + second_exp_term)

    # return the affinity matrix
    return W




def diff_map_info(W):
    '''
    Construct the information necessary to easily construct diffusion map for any t

    Inputs:
        W           a numpy array of size n x n containing the affinities between points

    Outputs:
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
        We assume the convention that the coordinates in the diffusion vectors are in descending order
        according to eigenvalues.
    '''

    # get symmetric matrix M_s
    D = np.diag(W.sum(axis=1)) # degree matrix of W = sum of W-th rows

    semi_D = scipy.linalg.fractional_matrix_power(D, -1/2) # power of (-1/2)
    M_s = semi_D @ W @ semi_D
    eig_val, eig_vector = np.linalg.eigh(M_s) # eigenpairs of M_s

    eig_vector_normalized = semi_D.dot(eig_vector) / np.linalg.norm(semi_D.dot(eig_vector), axis = 0) # normalzied eigenvector

    # discard trivial eigenvalue and eigenvector
    nontrivial_eig_val = eig_val[:-1]
    nontrivial_eig_vector = eig_vector_normalized[:,:-1] # discard last column

    # ascending order
    diff_eig = nontrivial_eig_val[::-1]
    diff_vec = np.flip(nontrivial_eig_vector, axis = 1)

    #return the info for diffusion maps
    return diff_vec, diff_eig 


def get_diff_map(diff_vec, diff_eig, t):
    '''
    Construct a diffusion map at t from eigenvalues and eigenvectors of Markov matrix
    Inputs:
        diff_vec    a numpy array of size n x n-1 containing the n-1 nontrivial eigenvectors of Markov matrix as columns
        diff_eig    a numpy array of size n-1 containing the n-1 nontrivial eigenvalues of Markov matrix
        t           diffusion time parameter t
    Outputs:
        diff_map    a numpy array of size n x n-1, the diffusion map defined for t
    '''

    diff_eig_power_t = diff_eig**t # t(step-size)-th power eigenvalues
    diff_map = diff_eig_power_t * diff_vec

    return diff_map
