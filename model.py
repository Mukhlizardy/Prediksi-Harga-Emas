import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import datetime

def loadData():
    emas= pd.read_csv('D:/Prediksi/static/data.csv')
    data=pd.DataFrame(emas)
    return data
def normData(data):
    datac=pd.DataFrame(data, columns=['Open', 'High', 'Low', 'Volume', 'Close'])
    datac= datac.replace(0, np.nan)
    dataclear=datac.dropna()
    dataNorm=(dataclear-dataclear.min())/(dataclear.max()-dataclear.min())
    return dataclear,dataNorm
def splitDataHarian(dataNorm):        
    x= dataNorm.drop(columns=['Close'])
    xa= np.array(x)
    y_label=dataNorm['Close']
    y=pd.DataFrame(y_label)
    yt=np.array(y)
    Xtrain=xa
    ytrain=yt
    #Xtrain, Xtest, ytrain, ytest = train_test_split(xa, yt, test_size=0.2, random_state=2)
    return Xtrain, ytrain
def splitData(dataNorm):        
    x= dataNorm.drop(columns=['Close'])
    xa= np.array(x)
    y_label=dataNorm['Close']
    y=pd.DataFrame(y_label)
    yt=np.array(y)
    Xtrain, Xtest, ytrain, ytest = train_test_split(xa, yt, test_size=0.3, random_state=42)
    return Xtrain, Xtest, ytrain, ytest

class NeuralNet():
    '''
    A two layer neural network
    '''
        
    def __init__(self, layers, learning_rate, epoch):
        self.params = {}
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.loss = []
        self.sample_size = None
        self.layers = layers
        self.X = None
        self.y = None
                
    def init_weights(self):
        '''
        Initialize the weights from a random normal distribution
        '''
        np.random.seed(1) # Seed the random number generator
        self.params["W1"] = np.random.randn(self.layers[0], self.layers[1]) 
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1],self.layers[2]) 
        self.params['b2'] = np.random.randn(self.layers[2],)
        '''
        print(self.params['W1'])
        print(self.params['b1'])
        print(self.params['W2'])
        print(self.params['b2'])
        '''
        return self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2']

    def sigmoid(self,Z,deriv=False):
        '''
        The sigmoid function takes in real numbers in any range and 
        squashes it to a real-valued output between 0 and 1.
        '''
        if(deriv==True):
            return Z*(1-Z)
        return 1/(1+np.exp(-Z))

    def forward_propagation(self,X):
        '''
        Performs the forward propagation
        '''
        self.z= np.dot(X,self.params["W1"])+self.params['b1']
        self.z2= self.sigmoid(self.z)
        self.z3= np.dot(self.z2,self.params["W2"])+self.params['b2']
        output= self.sigmoid(self.z3)
        #print(output)
        return output
    
    def backward(self,X,y,output):
        self.output_error=y-output
        self.output_delta=self.output_error* self.sigmoid(output, deriv=True)
        
        self.z2_error=self.output_delta.dot(self.params["W2"].T)
        self.z2_delta=self.z2_error* self.sigmoid(self.z2, deriv=True)
        #print(self.z2_delta)
                  
        self.params["W1"] += X.T.dot(self.z2_delta)*self.learning_rate
        self.params["W2"] += self.z2.T.dot(self.output_delta)*self.learning_rate
        self.params["b1"] += self.z2_delta.sum()*self.learning_rate
        self.params["b2"] += self.output_delta.sum()*self.learning_rate
        '''
        print(self.params['W1'])
        print(self.params['b1'])
        print(self.params['W2'])
        print(self.params['b2'])
        '''
        
    def fit(self, X, y):
        '''
        Trains the neural network using the specified data and labels
        '''
        self.X = X
        self.y = y
        self.init_weights() #initialize weights and bias
        
        for i in range(self.epoch):
            output=self.forward_propagation(X)
            self.backward(X,y,output)
        return self.params['W1'],self.params['b1'],self.params['W2'],self.params['b2']

    def predict(self, X):
        '''
        Predicts on a test data
        '''
        self.z= np.dot(X,self.params["W1"])+self.params['b1']
        self.z2= self.sigmoid(self.z)
        self.z3= np.dot(self.z2,self.params["W2"])+self.params['b2']
        output= self.sigmoid(self.z3)
        return output
    
    def mse(self, y, output):
        '''
        Calculates the accutacy between the predicted valuea and the truth labels
        '''
        mse = sum((y - output) ** 2) / len(y)
        return mse