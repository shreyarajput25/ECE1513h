import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def loadData():
    file = np.genfromtxt ('bank_final_last.csv', delimiter=";", dtype=float)
    Target = file[:,[27]] #Labels in last column, 1 = loan approved , 0 =not approved
    Data = file[:,0:27]#inputs from column 1-27
    
    posClass = 1
    negClass = 0
    data_pos = np.where(Target==posClass)
    data_neg = np.where(Target==negClass)
    pos_set = data_pos[0]
    neg_set = data_neg[0]
    leng = len(pos_set)
    neg_index = neg_set[:leng]
    np.random.seed(521)
    
    data = np.zeros((2*leng, 27), dtype=float)
    target = np.zeros((2*leng, 1), dtype=int)

    data[:leng], target[:leng] = Data[pos_set], Target[pos_set]
    data[leng:], target[leng:] = Data[neg_index], Target[neg_index]
#
    return data, target
    
data, target = loadData()
trainData, testData, trainTarget, testTarget = train_test_split(data, target, test_size = 0.20, random_state=109)


scaling = MinMaxScaler(feature_range=(-1,1)).fit(trainData)
trainData = scaling.transform(trainData)
testData = scaling.transform(testData)
    
def buildgraph(learning_rate, shape):
    tf.set_random_seed(521)
    W = tf.Variable(tf.truncated_normal(shape=[shape,1], stddev=0.5), name='weight')
    b = tf.Variable(0.0, name='bias')
    X = tf.placeholder(tf.float32, [None, shape], name='inputx')
    Y = tf.placeholder(tf.float32, [None,1], name='y_label')
    Lambda = tf.placeholder(tf.float32, name='lambda')
    Y_hat = tf.matmul(X, W) + b
    #Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(X, W), b))
    regterm = tf.nn.l2_loss(W)*Lambda
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=Y, logits=Y_hat)
    loss = loss + regterm
    
    train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.99,epsilon=1e-2).minimize(loss=loss)
    return W, b, X, Y, Y_hat, loss, train, Lambda 
    
def trainmodel(epochs, learning_rate):#:stochasticGradient
    trainshape = trainData.shape[1]
    Weight, b, X, Y, Y_hat, loss, train, Lambda = buildgraph(learning_rate, trainshape)
    init = tf.global_variables_initializer()
    Lamba_w = 0.01
    trainError = []
    trainAccuracy = []
    testError = []
    testAccuracy = []
    Weig = []
    patience = 300
    last_acc = 100
    j=0
    index =0
    tol_check = 10
    with tf.Session() as sess:
        sess.run(init)
        for currentepoch in range(0, epochs):
            
            print(currentepoch)
            Ein, predicted, a = sess.run([loss, Y_hat, train], feed_dict={X:trainData, Y:trainTarget, Lambda:Lamba_w})
            acc = np.mean((predicted > 0.5) == trainTarget)
            trainError.append(Ein)
            trainAccuracy.append(acc)
            
            error2, Y_hat2 = sess.run([loss, Y_hat], feed_dict={X:testData, Y:testTarget, Lambda:Lamba_w})
            acc2 = np.mean((Y_hat2 > 0.5) == testTarget)
            testError.append(error2)
            testAccuracy.append(acc2)
            
            weight, bia, Y_t, los =sess.run([Weight, b, Y_hat, loss], feed_dict={X:trainData, Y:trainTarget, Lambda:Lamba_w})
            Weig.append(weight)
            
            if((currentepoch>patience)and(last_acc>acc2)):
                if(j<tol_check):
                    j=j+1
                else:
                    last_acc=acc2
                    index = currentepoch
                    j=0
    
    return Weig, trainError, trainAccuracy, testError, testAccuracy, index, last_acc

def plot(fig, title, label, loss1, loss3):
    plt.figure(fig)
    plt.xlabel("Iterations")
    plt.title(title)
    plt.ylabel(label)
    plt.plot(range(len(loss1)), loss1, '-b', label=r"Training Data")
   # plt.plot(range(len(loss2)), loss2, '-r', label=r"Validation Data")
    plt.plot(range(len(loss3)), loss3, '-g', label=r"Valid Data")
    plt.legend(loc='upper left')
    #plt.axis([0,len(loss1),0,1.2*max(max(loss2),max(loss3))])
    plt.show()
    plt.savefig(title)
 #, bias, X, Y, Y_hat, loss, train, Lambda
Weig, trainError, trainAccuracy, testError, testAccuracy, earlystop, last_acc= trainmodel(500, 0.1)
print("Final Train loss + accuracy %.3f, %.3f, Final test loss + accuracy %.3f, %.3f"%(trainError[-1],trainAccuracy[-1],testError[-1],testAccuracy[-1]))
plot(1, "Error vs iteration", "Error", trainError, testError)
plot(2, "Accuracy vs iteration ", "Accuracy", trainAccuracy, testAccuracy)
print("early stop")
print(earlystop)
print(last_acc)
