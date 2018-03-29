
# coding: utf-8

# In[ ]:
import numpy as np
import time


class Network:
    
    def __init__(self,layer_units,learning_rate,epochs,x_train,y_train,x_test,y_test,x_val,y_val,lrSet=None):
        self.layer_units = layer_units
        self.lrSet = lrSet
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss = 0
        self.accu = 0
        self.x_train = x_train
        self.trainSize = np.amax(x_train.shape)
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val
        self.weights = list()
        self.bias = list()
        self.wsums = list()
        self.deltas = list()
        self.activations = list()
        self.gradW = list()
        self.gradB = list()
        self.activations.append(np.zeros(layer_units[0]))
        for i in range(len(layer_units)-1):
            self.weights.append(self.init_params([layer_units[i+1],layer_units[i]]))
            self.bias.append(self.init_params([layer_units[i+1]]))
            self.wsums.append(np.zeros(layer_units[i+1]))
            self.activations.append(np.zeros(layer_units[i+1]))
            self.deltas.append(np.zeros(layer_units[i+1]))
            self.gradW.append(np.zeros([layer_units[i+1],layer_units[i]]))
            self.gradB.append(np.zeros([layer_units[i+1],1]))
            
    def init_params(self,size):
        in_dim = size[-1]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        if(len(size) == 1):
            return np.random.randn(size[0])*xavier_stddev
        elif(len(size) == 2):
            return np.random.randn(size[0],size[1])*xavier_stddev

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def softmax(self,x):
        return (np.exp(x) / np.sum(np.exp(x),axis=0) )
    
    def computeLoss(self,xdata,ylabel):
        y = self.forward_pass(xdata)
        size  = ylabel.shape[1]
        self.loss = ( - np.sum( ylabel*np.log(y) + ((1-ylabel)*np.log(1-y)) )/size )
        return self.loss
    
    def accuracy(self,xdata,ylabels):
        y = self.forward_pass(xdata)
        match = np.equal(np.argmax(y,axis=0),np.argmax(ylabels,axis=0)).astype(float)
        acc   = np.sum(match)
        size  = ylabels.shape[1]
        self.accu = 100*acc/size
        return self.accu
    
    def forward_pass(self,x):
        self.activations[0] = x
        for i in range(len(self.layer_units)-1):
            z = self.weights[i].dot(self.activations[i]).T + self.bias[i]
            self.wsums[i] = z.T
            if (i == len(self.layer_units)-1):
                self.activations[i+1] = self.softmax(z.T)
            else:
                self.activations[i+1] = self.sigmoid(z.T)
        y = self.activations[len(self.layer_units)-1]
        return y
    
    def backprop(self,y,ylabel):
        last = len(self.layer_units) - 2
        self.deltas[last] = y - ylabel
        for i in range(last,0,-1):
            self.deltas[i-1] = self.weights[i].T.dot(self.deltas[i]) * self.activations[i] * (1-self.activations[i])
        for i in range(len(self.layer_units)-1):
            self.gradW[i] = - self.learning_rate * self.deltas[i].reshape(self.layer_units[i+1],1).dot(self.activations[i].reshape(1,self.layer_units[i]))
            self.gradB[i] = - self.learning_rate * self.deltas[i]
            
    def gradientDescent(self):
        for i in range(len(self.layer_units)-1):
            self.weights[i] += self.gradW[i]
            self.bias[i] += self.gradB[i]
            
    def train(self):
        start_time = time.time()
        print("Epoch ","    Loss","    Accuracy (train,val)","    Time (Mins)")
        for i in range(self.epochs):
            if (not(self.lrSet == None)):
                self.learning_rate = self.lrSet[i]
            for j in range(self.trainSize):
                out = self.forward_pass(self.x_train[:,j])
                self.backprop(out,self.y_train[:,j])
                self.gradientDescent()
            self.computeLoss(self.x_train,self.y_train)
            ta = self.accuracy(self.x_train,self.y_train)
            va = self.accuracy(self.x_val,self.y_val)
            tm = time.time() - start_time
            print(i,"        ","{0:.3f}".format(self.loss),"  ","{0:.3f}".format(ta),",","{0:.3f}".format(va),"          ",
                  "{0:.3f}".format(tm/60))

    def val(self):
        self.computeLoss(self.x_val,self.y_val)
        self.accuracy(self.x_val,self.y_val)
        print("Test accuracy: ",self.accu," Test loss: ",self.loss)
        
    def test(self):
        self.computeLoss(self.x_test,self.y_test)
        self.accuracy(self.x_test,self.y_test)
        print("Test accuracy: ",self.accu," Test loss: ",self.loss)

