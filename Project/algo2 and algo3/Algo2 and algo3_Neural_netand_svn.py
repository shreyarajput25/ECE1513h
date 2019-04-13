from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import keras
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve
from sklearn.model_selection import train_test_split  
from sklearn import svm
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split  
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import auc
from sklearn.metrics import roc_curve


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
    np.random.seed(40)
    
    data = np.zeros((2*leng, 27), dtype=float)
    target = np.zeros((2*leng, 1), dtype=int)

    data[:leng], target[:leng] = Data[pos_set], Target[pos_set]
    data[leng:], target[leng:] = Data[neg_index], Target[neg_index]
#
    return data, target


class PlotLearning(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1
        f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        
        clear_output(wait=True)
        
        ax1.set_yscale('log')
        ax1.plot(self.x, self.losses, label="Training loss")
        ax1.plot(self.x, self.val_losses, label="Validation loss")
        ax1.legend()
        
        ax2.plot(self.x, self.acc, label="Training accuracy")
        ax2.plot(self.x, self.val_acc, label="validation accuracy")
        ax2.legend()
        
        plt.show();

def neural_net():
    data, target = loadData()
    trainData, testData, trainTarget, testTarget = train_test_split(data, target, test_size = 0.20, random_state=109)
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(trainData)
    trainData = scaling.transform(trainData)
    testData = scaling.transform(testData)
    earlystop = EarlyStopping(monitor='val_acc', patience=100, verbose=1, mode='auto', min_delta=0.001)
    plot = PlotLearning()
    model = Sequential()
    model.add(Dense(20, input_dim=trainData.shape[1], activation='tanh', kernel_regularizer=regularizers.l2(0.001), activity_regularizer=regularizers.l2(0.001)))
    
    model.add(Dense(1, activation='sigmoid'))
    #opt = SGD(lr=0.005, momentum=0, decay=0.01)
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    run = model.fit(trainData,trainTarget , epochs=400,validation_data=(testData, testTarget),
              callbacks=[plot], batch_size=trainData.shape[0]//20)
    test_loss, test_acc = model.evaluate(testData, testTarget)
    
    y_pred_keras = model.predict(testData).ravel()
    fpr_keras, tpr_keras, thresholds_keras = roc_curve(testTarget, y_pred_keras)
    print('Test accuracy:', test_acc)
    return fpr_keras, tpr_keras, thresholds_keras
 



def svc_param_selection(X, y, nfolds):
    
    Cs = [0.001, 0.01, 0.1]
    gammas =[0.001, 0.01, 0.1]
    kernel =['rbf','linear','poly']
    param_grid = {'kernel' :kernel, 'C': Cs, 'gamma' : gammas}
    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds, scoring='precision')
    grid_search.fit(X, y)
    grid_search.best_params_
    print(grid_search.best_params_)
    

def svm():    
    data, target = loadData()
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.20, random_state=109)
    scaling = MinMaxScaler(feature_range=(-1,1)).fit(X_train)
    X_train = scaling.transform(X_train)
    X_test = scaling.transform(X_test)
    
    y_train = np.ravel(y_train,order='C')
    y_test = np.ravel(y_test,order='C')
    svclassifier = SVC(C=0.1, gamma=0.01, kernel='linear') 
     
    svclassifier = SVC(C=0.1, gamma=0.01, kernel='linear', probability=True) 
 
    svclassifier.fit(X_train, y_train)
    y_hat = svclassifier.predict(X_train)
    y_pred = svclassifier.predict(X_test)
    print("Accuracy Test Data:",metrics.accuracy_score(y_train, y_hat))
    print("Accuracy Test Data:",metrics.accuracy_score(y_test, y_pred))
    
    print("Precision:",metrics.precision_score(y_test, y_pred))
          
    print(confusion_matrix(y_test, y_pred))  
    print(classification_report(y_test, y_pred)) 
    
    y_pred_rf = svclassifier.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_pred_rf)
    return fpr_rf, tpr_rf, thresholds_rf

fpr_keras, tpr_keras, thresholds_keras = neural_net()
auc_neural = auc(fpr_keras, tpr_keras)
fpr_rf, tpr_rf, thresholds_rf = svm() 
auc_SVM = auc(fpr_rf, tpr_rf)

def roc_cruve():
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Neural Net (area = {:.3f})'.format(auc_neural))
    plt.plot(fpr_rf, tpr_rf, label='SVM (area = {:.3f})'.format(auc_SVM))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    # Zoom in view of the upper left corner.
    plt.figure(2)
    plt.xlim(0.5, 0.8)
    plt.ylim(0.6, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Neural Net (area = {:.3f})'.format(auc_neural))
    plt.plot(fpr_rf, tpr_rf, label='SVM (area = {:.3f})'.format(auc_SVM))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show()
    
#roc_cruve() 
#run roc curves