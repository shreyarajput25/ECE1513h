import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import helper as hlp
import math as m
from helper import reduce_logsumexp
from helper import logsoftmax

def loadData2D(is_valid=True):
    # Loading data
    data = np.load('data2D.npy')
    val_data = 0
    #data = np.load('data100D.npy')
    [num_pts, dim] = np.shape(data)

    # For Validation set
    if is_valid:
        valid_batch = int(num_pts / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        val_data = data[rnd_idx[:valid_batch]]
        data = data[rnd_idx[valid_batch:]]

    return data, dim, val_data
        
def loadData100D(is_valid=True):
    # Loading data
    data = np.load('data100D.npy')
    #data = np.load('data100D.npy')
    [num_pts, dim] = np.shape(data)
    val_data = 0
    # For Validation set
    if is_valid:
        valid_batch = int(num_pts / 3.0)
        np.random.seed(45689)
        rnd_idx = np.arange(num_pts)
        np.random.shuffle(rnd_idx)
        val_data = data[rnd_idx[:valid_batch]]
        data = data[rnd_idx[valid_batch:]]
    return data, dim, val_data

def distanceFunc(X, MU):
    # Inputs
    # X: is an NxD matrix (N observations and D dimensions)
    # MU: is an KxD matrix (K means and D dimensions)
    # Outputs
    # pair_dist: is the pairwise distance matrix (NxK)
    # TODO
    return tf.reduce_sum(tf.square(tf.expand_dims(X,1)-tf.expand_dims(MU,0)),2)

def log_GaussPDF(X, MU, variance):
    D = distanceFunc(X, MU) #NxK
    var = tf.transpose(variance)
    logProb1= dim*0.5*tf.log(2*np.pi*var)
    logProb2 = 0.5*tf.multiply(D, 1/var) #(1/2*var * (x-u)^2)
    return -logProb1 -logProb2
# Distance function for GMM
#param learning_rate: optimizer's learning rate

#def log_posterior(log_PDF, log_pi):
    # Input
    # log_PDF: log Gaussian PDF N X K
    # log_pi: K X 1

    # Outputs
    # log_post: N X K
    
def log_posterior(logpi, X, MU, variance):
    logpdf = log_GaussPDF(X, MU, variance)
    logpdf_Num= tf.log(tf.transpose(logpi)) + logpdf
    log_pdf = reduce_logsumexp(logpdf_Num, 1, True)
    #log_pdf = tf.reshape(reduce_logsumexp(logpdf_Num, 1), [-1, 1])
    return logpdf_Num - log_pdf
    
def loss(logpi, X, MU, variance):
    logpdf = log_GaussPDF(X, MU, variance)
    logpdf_Num= tf.log(tf.transpose(logpi)) + logpdf
    log_pdf = -reduce_logsumexp(logpdf_Num, 1, True)
    #log_pdf = -tf.reshape(reduce_logsumexp(logpdf_Num, 1), [-1, 1])
    return tf.reduce_mean(log_pdf, 0)
    
def buildGraph(learning_rate, dim, k):
    # Variable creation
    input_x = tf.placeholder(tf.float32, [None, dim], name='input_x')
    k_centers = tf.Variable(tf.random_normal([k, dim], stddev=0.5))
    phi = tf.Variable(tf.random_normal([k, 1], stddev=0.5))
    var = tf.exp(phi)
    sai = tf.Variable(tf.random_normal([k, 1], stddev=0.5))
    logpi = tf.exp(logsoftmax(sai))
    log_posterior1 = log_posterior(logpi, input_x, k_centers, var)
    #test = tf.constant(1, tf.float32, shape=[3, 2])
    #test1, test2, test3 = reduce_logsumexp(test, reduction_indices=1, keep_dims=False)
    Ein= loss(logpi, input_x, k_centers, var)
    predicted = tf.argmax(log_posterior1, 1)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99,epsilon=1e-5).minimize(loss=Ein)
    
    return input_x, k_centers, optimizer, logpi, predicted, var, Ein

def train_Gmm(epoch, k, is_valid, dim, data, val_data):
    input_x, mean, optimizer, logpi, predicted, var, Ein= buildGraph(0.01, dim, k)
    init = tf.global_variables_initializer()
    loss_train =[]
    Accuracy_train = []
    loss_valid = []
    Accuracy_valid = []
    val_predicted = 0
    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoch):
            a, loss_trai, train_prediction = sess.run([optimizer, Ein, predicted], feed_dict={input_x: data})
            loss_train.append(float(loss_trai))

            if is_valid:
                val_loss, val_predicted = sess.run([Ein, predicted], feed_dict={input_x: val_data})
                loss_valid.append(float(val_loss))
        mean, logpi, var = sess.run([mean, logpi, var])
        
    return loss_train, train_prediction, mean, logpi, var, loss_valid, val_predicted

def plot(fig, title, label, loss1, loss2, is_valid):
    plt.figure(fig)
    plt.xlabel("Iterations")
    plt.title(title)
    plt.ylabel(label)
    
    if is_valid:
        plt.plot(range(len(loss2)), loss2, '-r', label=r"Validation Data")
        plt.plot(range(len(loss1)), loss1, '-b', label=r"Training Data")
    else:
        plt.plot(range(len(loss1)), loss1, '-b', label=r"Data")
    plt.legend(loc='upper left')
    #plt.axis([0,len(loss1),0,1.2*max(max(loss2),max(loss3))])
    plt.show()
    plt.savefig(title)

def plot_scatter(k, traindata, label=None, centers= None):
    color_list = ["b","g","r","y","c"]
    clabel = []
    for i in range(len(label)):
        clabel.append(color_list[label[i]])
    plt.figure()
    plt.title("prediction data points with {} clusters".format(k))
    plt.xlim([-5,5])
    plt.ylim([-5,5])
    plt.scatter(traindata[:,0], traindata[:,1], c=clabel, marker='.')
    plt.scatter(centers[:, 0], centers[:, 1], c='k', marker='x')
    
is_valid =True
#data, dim, val_data = loadData2D(is_valid)
data, dim, val_data = loadData100D(is_valid)

print("with 1/3rd data as the valud")
cluster = [5, 10, 15, 20, 30]
for k in range(5, 6):
    loss_train, train_pred, mean, logpi, var, val_loss, val_predicted= train_Gmm(1000, k, is_valid, dim, data, val_data)
    
    print('K = {}'.format(k))
    print('Final Training loss: ', loss_train[-1])
    #plot_scatter(k, data, train_pred, mean)
    if is_valid:
        print('Final Validation loss:', val_loss[-1])
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)
        #plot_scatter(k, val_data, val_predicted, mean)
    else:
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)

for k in range(10, 11):
    loss_train, train_pred, mean, logpi, var, val_loss, val_predicted= train_Gmm(1000, k, is_valid, dim, data, val_data)
    
    print('K = {}'.format(k))
    print('Final Training loss: ', loss_train[-1])
    #plot_scatter(k, data, train_pred, mean)
    if is_valid:
        print('Final Validation loss:', val_loss[-1])
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)
        #plot_scatter(k, val_data, val_predicted, mean)
    else:
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)
    
for k in range(15, 16):
    loss_train, train_pred, mean, logpi, var, val_loss, val_predicted= train_Gmm(1000, k, is_valid, dim, data, val_data)
    
    print('K = {}'.format(k))
    print('Final Training loss: ', loss_train[-1])
    #plot_scatter(k, data, train_pred, mean)
    if is_valid:
        print('Final Validation loss:', val_loss[-1])
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)
        #plot_scatter(k, val_data, val_predicted, mean)
    else:
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)
        
for k in range(20, 21):
    loss_train, train_pred, mean, logpi, var, val_loss, val_predicted= train_Gmm(1000, k, is_valid, dim, data, val_data)
    
    print('K = {}'.format(k))
    print('Final Training loss: ', loss_train[-1])
    #plot_scatter(k, data, train_pred, mean)
    if is_valid:
        print('Final Validation loss:', val_loss[-1])
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)
        #plot_scatter(k, val_data, val_predicted, mean)
    else:
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)
        
for k in range(30, 31):
    loss_train, train_pred, mean, logpi, var, val_loss, val_predicted= train_Gmm(1000, k, is_valid, dim, data, val_data)
    
    print('K = {}'.format(k))
    print('Final Training loss: ', loss_train[-1])
    #plot_scatter(k, data, train_pred, mean)
    if is_valid:
        print('Final Validation loss:', val_loss[-1])
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)
        #plot_scatter(k, val_data, val_predicted, mean)
    else:
        plot(k, "Loss vs iteration", "loss", loss_train, val_loss, is_valid)