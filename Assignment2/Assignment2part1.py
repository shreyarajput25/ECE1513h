import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load("notMNIST.npz") as data:
        Data, Target = data["images"], data["labels"]
        np.random.seed(521)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data = Data[randIndx] / 255.0
        Target = Target[randIndx]
        trainData, trainTarget = Data[:10000], Target[:10000]
        validData, validTarget = Data[10000:16000], Target[10000:16000]
        testData, testTarget = Data[16000:], Target[16000:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
def convertOneHot(trainTarget, validTarget, testTarget):
    newtrain = np.zeros((trainTarget.shape[0], 10))
    newvalid = np.zeros((validTarget.shape[0], 10))
    newtest = np.zeros((testTarget.shape[0], 10))

    for item in range(0, trainTarget.shape[0]):
        newtrain[item][trainTarget[item]] = 1
    for item in range(0, validTarget.shape[0]):
        newvalid[item][validTarget[item]] = 1
    for item in range(0, testTarget.shape[0]):
        newtest[item][testTarget[item]] = 1
    return newtrain, newvalid, newtest

def Relu(x):
    x = np.maximum(0,x)
    return x

def Reludiff(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def softmax(z):
    expo = np.exp(z - np.max(z))
    sum = np.sum(expo, axis = 1)
    sum = sum.reshape(sum.shape[0], 1)
    return np.divide(expo, sum)

def inputreshape(x):
    x_0 = x.shape[0]
    x_size = x.shape[1]*x.shape[2]
    return x.reshape(x_0, x_size)
    
def compute(Weight, Input, Bias):
    return np.matmul(Input, Weight) + Bias

def averageCE(label, prediction):
    N = prediction.shape[0]
    avg_CE =  np.sum(np.multiply(label, np.log(prediction)))
    return -avg_CE/N
    
def gradCE(label, prediction):
    gradCE = np.divide(label, prediction)
    return -np.sum(gradCE)

def average_Accuracy(label,prediction):
    temp1=np.argmax(prediction,axis=1)
    temp2=np.argmax(label,axis=1)
    temp3=np.equal(temp1,temp2)
    return ((1/temp3.shape[0])*np.sum(temp3))*100
    
def backwardMessageOut(label, prediction):
    return np.subtract(prediction, label)

def forwardpropogation(Input, W_hidden, W_out, Bias_hidden, Bias_out):
    S_hidden = compute(W_hidden, Input, Bias_hidden)
    Out_h = Relu(S_hidden)
    relu_diff = Reludiff(S_hidden)
    S_out = compute(W_out, Out_h, Bias_out)
    Out_l = softmax(S_out)
    return Out_h, Out_l, relu_diff

def backwardpropogation(Input, trainingLabels, W_hidden, W_out, Bias_hidden, Bias_out, Output_h, output_o, relu_diff):
    b_msg = backwardMessageOut(trainingLabels, output_o)
    row = np.multiply((np.matmul(b_msg, np.transpose(W_out))), relu_diff)
    dL_bo = (1/Output_h.shape[0])*np.sum(b_msg, axis = 0)
    dL_wo = (1/Output_h.shape[0])*np.matmul(np.transpose(Output_h), b_msg)
    dL_bh = (1/Input.shape[0])*np.sum(row, axis = 0)
    dL_wh = (1/Input.shape[0])*np.matmul(np.transpose(Input), row)
    return dL_bo, dL_wo, dL_bh, dL_wh
       
def trainneuralnet(alpha, iterations, k):
    trainData, validData, testData, trainTarget1, validTarget1 , testTarget1 = loadData()
    Train_label, Valid_label, Test_label = convertOneHot(trainTarget1, validTarget1, testTarget1)
    
    Train_D = inputreshape(trainData)
    Vaid_D = inputreshape(validData)
    Test_D = inputreshape(testData)
    
    mu = 0
    input_h = Train_D.shape[1]
    output = Train_label.shape[1]
    Var_h = np.sqrt(2/(input_h + k))
    weight_h = np.random.normal(mu,Var_h , (input_h, k))
    weight_o = np.random.normal(mu, np.sqrt(2/(k + output)), (k, output))
    Bias_hidden = np.random.normal(mu, np.sqrt(2/(1 + k)), (1, k))
    Bias_out = np.random.normal(mu, np.sqrt(2/(1+output)), (1, output))
    v_hidd = np.multiply((np.ones((input_h, k))), 0.00001)
    v_b_hidd = np.multiply((np.ones((1, k))), 0.00001)
    v_output = np.multiply((np.ones((k, output))), 0.00001)
    v_b_out = np.multiply((np.ones((1, output))), 0.00001)
    gamma=0.95
    Ein_train_data = []
    Ein_test = []
    Ein_valid = []
    Accuracy_Train = []
    Accuracy_valid = []
    Accuracy_Test = []

    for i in range(0, iterations):
        print(i)
        out_hidden, out_last, reludiff = forwardpropogation(Train_D, weight_h, weight_o, Bias_hidden, Bias_out)
        B_o, W_o, B_h, W_h = backwardpropogation(Train_D, Train_label, weight_h, weight_o, Bias_hidden, Bias_out, out_hidden, out_last, reludiff)
        v_hidd = np.multiply(gamma, v_hidd) + np.multiply(alpha, W_h)
        v_b_hidd = np.multiply(gamma, v_b_hidd)+ np.multiply(alpha, B_h)
        v_output = np.multiply(gamma, v_output)+ np.multiply(alpha, W_o)
        v_b_out = np.multiply(gamma, v_b_out)+ np.multiply(alpha, B_o)
        weight_h = weight_h-v_hidd
        weight_o = weight_o-v_output
        Bias_hidden = Bias_hidden-v_b_hidd
        Bias_out = Bias_out-v_b_out
        #training error loss
        Eintrain = averageCE(Train_label, out_last)
        accuracy1 = average_Accuracy(Train_label, out_last)
        Ein_train_data.append(float(Eintrain))
        Accuracy_Train.append(float(accuracy1))
       
        o_h, pred_valid, reludif = forwardpropogation(Vaid_D, weight_h, weight_o, Bias_hidden, Bias_out)
        outhidden, pred_test, relu_diff = forwardpropogation(Test_D, weight_h, weight_o, Bias_hidden, Bias_out)
        #valid data loss
        Einvalid = averageCE(Valid_label, pred_valid)
        accuracy2 = average_Accuracy(Valid_label, pred_valid)
        Ein_valid.append(float(Einvalid))
        Accuracy_valid.append(float(accuracy2))
      
        
        #test data loss
        Eintest = averageCE(Test_label, pred_test)
        accuracy3 = average_Accuracy(Test_label, pred_test)
        Ein_test.append(float(Eintest))
        Accuracy_Test.append(float(accuracy3))
       
        
    return weight_o, weight_h, Ein_train_data, Ein_test, Ein_valid, Accuracy_Train, Accuracy_valid, Accuracy_Test    

def plot(fig, title, label, loss1, loss2, loss3):
    plt.figure(fig)
    plt.xlabel("Iterations")
    plt.title(title)
    plt.ylabel(label)
    plt.plot(range(len(loss1)), loss1, '-b', label=r"Training Data")
    plt.plot(range(len(loss2)), loss2, '-r', label=r"Validation Data")
    plt.plot(range(len(loss3)), loss3, '-g', label=r"Test Data")
    plt.legend(loc='upper left')
    #plt.axis([0,len(loss1),0,1.2*max(max(loss2),max(loss3))])
    plt.show()
    plt.savefig(title)
    
iterations = 200
units = 1000
Final_weight_out, final_weight_h, Loss_train, Loss_testd, Loss_validd, acc1, acc2, acc3 = trainneuralnet(0.07, iterations, units)
print("Training loss for 1000 units= ", Loss_train[-1])
print("valid Loss for 1000 units= = ", Loss_validd[-1])
print("test loss for 1000 units= ", Loss_testd[-1])
print("Training accuracy for 1000 units= ", acc1[-1])
print("Valid accuracy for 1000 units= ", acc2[-1])
print("Test accuracy for 1000 units= ", acc3[-1])
plot(1, "Error vs iteration for hidden units = 100", "Error", Loss_train, Loss_validd, Loss_testd)
plot(2, "Accuracy vs iteration for hidden units = 100", "Accuracy", acc1, acc2, acc3)

Final_weight_out1, final_weight_h1, Loss_train1, Loss_testd1, Loss_validd1, acc_1, acc_2, acc_3 = trainneuralnet(0.07, iterations, 100)
plot(3, "Error vs iteration for hidden units = 100", "Error", Loss_train1, Loss_validd1, Loss_testd1)
plot(4, "Accuracy vs iteration for hidden units = 100", "Accuracy", acc_1, acc_2, acc_3)
print("Training loss for 100 units= ", Loss_train1[-1])
print("valid Loss for 100 units= = ", Loss_validd1[-1])
print("test loss for 100 units= ", Loss_testd1[-1])
print("Training accuracy for 100 units= ", acc_1[-1])
print("Valid accuracy for 100 units= ", acc_2[-1])
print("Test accuracy for 100 units= ", acc_3[-1])

Final_weight_out1, final_weight_h1, Losstrain1, Losstestd1, Lossvalidd1, accu1, accu2, accu3 = trainneuralnet(0.07, iterations, 200)
plot(5, "Error vs iteration for hidden units = 200", "Error", Losstrain1, Lossvalidd1, Losstestd1)
plot(6, "Accuracy vs iteration for hidden units = 200", "Accuracy", accu1, acc_2, accu3)
print("Training loss for 200 units= ", Losstrain1[-1])
print("valid Loss for 200 units= = ", Lossvalidd1[-1])
print("test loss for 200 units= ", Losstestd1[-1])
print("Training accuracy for 200 units= ", accu1[-1])
print("Valid accuracy for 200 units= ", accu2[-1])
print("Test accuracy for 200 units= ", accu3[-1])

Final_weight_out1, final_weight_h1, Losstrain1_, Losstestd1_, Lossvalidd1_, accu_1, accu_2, accu_3 = trainneuralnet(0.07, iterations, 2000)
plot(7, "Error vs iteration for hidden units = 2000", "Error", Losstrain1_, Lossvalidd1_, Losstestd1_)
plot(8, "Accuracy vs iteration for hidden units = 2000", "Accuracy", accu_1, accu_2, accu_3)
print("Training loss for 2000 units= ", Losstrain1_[-1])
print("valid Loss for 2000 units= = ", Lossvalidd1_[-1])
print("test loss for 2000 units= ", Losstestd1_[-1])
print("Training accuracy for 2000 units= ", accu_1[-1])
print("Valid accuracy for 2000 units= ", accu_2[-1])
print("Test accuracy for 2000 units= ", accu_3[-1])

#b = compute(W, trainData, 1)

#v_hidd = np.multiply((np.ones(100, 10)), 0.00001)
#b = np.ones((15000, 10))
#a = softmax1(b)
#a = backwardMessageOut(x1, x2)
#print(b.shape)
#print(v_hidden)
#print(v_out)