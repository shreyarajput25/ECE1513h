import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import helper as hlp
import math as m
# Loading data
#data = np.load('data100D.npy')
data = np.load('data2D.npy')
[num_pts, dim] = np.shape(data)
variance = np.var(data)
is_valid = True
# For Validation set
if is_valid:
  valid_batch = int(num_pts / 3.0)
  np.random.seed(45689)
  rnd_idx = np.arange(num_pts)
  np.random.shuffle(rnd_idx)
  val_data = data[rnd_idx[:valid_batch]]
  data = data[rnd_idx[valid_batch:]] 
    
# Distance function for GMM
def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    return tf.reduce_sum(tf.square(tf.expand_dims(X,1)-tf.expand_dims(MU,0)),2)
    

def log_GaussPDF(X, mu, sigma):
    # Inputs
    # X: N X D
    # mu: K X D
    # sigma: K X 1
    pi = tf.constant(m.pi)
    var = tf.square(sigma)
    distance = distanceFunc(X, mu)
    NormDis = tf.exp((-1/2)*(distance)/(var))
    return (tf.math.sqrt(1/(2*var*pi)))*NormDis
    # Outputs:
    # log Gaussian PDF N X K

    # TODO

#def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K

    # TODO
def buildGraph(learning_rate =0.01, dim=2, k=3):
    input_x = tf.placeholder(tf.float, [None, dim], name='input_x')
    k_centers = tf.Variable(tf.random_normal([k, dim], stddev=0.5))
    data_size = tf.placeholder(tf.float32)
    sigma = tf.fill([k, 1], variance)
    loss = log_GaussPDF(data, k_centers, sigma)
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beat2=0.99, epsilone=1e-5).minimize(loss=loss)
    return input_x, k_centers, loss, data_size

def part2(k, epochs = 1000):
    input_x, k_centers, loss, data_size = buildGraph(learning_rate =0.01, dim=dim, k=k)
    init = tf.Initializer.global_variables()
    TrainLoss = []
    ValidLoss = []
    
   