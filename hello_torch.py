#%%
# import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline

# import MNIST data
from tensorflow.examples.tutorials.mnist import input_data 
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
from sklearn.preprocessing import MinMaxScaler
#%% 
import numpy as np
import torch as t
device = t.device("cuda:0" if t.cuda.is_available() else "cpu")

trainX = np.vstack([img.reshape(-1,) for img in mnist.train.images])
# trainX = MinMaxScaler().fit_transform(trainX)
trainX = t.from_numpy(trainX).float()
trainY = t.from_numpy(mnist.train.labels).float()

testX = np.vstack([img.reshape(-1,) for img in mnist.test.images])
testX = t.from_numpy(testX).float()
testY = t.from_numpy(mnist.test.labels).float()
# trainX = MinMaxScaler().fit_transform(trainX)
#%%
#define activation function
def relu(z):
    return z * (z > 0).float().to(device)

def relu_prime(z):
    return 1. * (z > 0).float().to(device)

def sigmoid(z):
    one = t.ones(1).to(device)
    return one / (one + t.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (t.ones(1).to(device) - sigmoid(z))

def error_formula(y, y_hat):
    # multi class error formula
    loss_sum = - t.mul(y, t.log(y_hat)).sum()
    return loss_sum / y.shape[1]

def softmax(output):
    expOutput = t.exp(output)
    return t.div(expOutput, expOutput.sum(dim=0).reshape(-1, expOutput.shape[1]))
#%%
# define neural network

# Input Layer -> [1 784] 
# Hidden Layer 1 -> [1 128]
# Hidden Layer 2 -> [1 64]
# Output Layer -> [1 10]
# Equation -> y = ((trainX * weights1 + bias1) * weights2 + bias2) * weights3 + bias3
weights1 = t.rand(300, 784).float()
bias1 = t.rand(300, 1).float()
weights2 = t.ones(10, 300).float()
bias2 = t.rand(10, 1).float()
# put variables to GPU
trainX = trainX.to(device)
trainY = trainY.to(device)
weights1 = weights1.to(device)
bias1 = bias1.to(device)
weights2 = weights2.to(device) # output
bias2 = bias2.to(device)
testX = testX.to(device)
testY = testY.to(device)
# bias3 = bias3.to(device)

#%%
epoch = 50
learningRate = 0.03
batchSize = 50

iteration = trainX.shape[0]
for e in range(epoch):
    i = 0
    while i < iteration:
        # print('debug: ', i)
        # feed forward
        X = trainX[i:i + batchSize].t()
        Y = trainY[i:i + batchSize].t()

        # print('debug: ',X)
        # print('debug: ',Y)
        # Hidden Layer 1
        hidden = t.mm(weights1, X) + bias1
        activation1 = sigmoid(hidden)
        # Output Layer
        output = t.mm(weights2, activation1) + bias2
        activation2 = sigmoid(output)
        # calculate the error
        error = error_formula(Y, activation2)
        # print('debug: ',error)

        # backprop
        # calculate error for every nodes
        output_errors = Y - activation2
        hidden_errors = t.mm(weights2.t(), output_errors)

        weights2_grad = t.mm(output_errors, activation1.t()) / batchSize
        bias2_grad = (output_errors.sum(dim=1) / batchSize).reshape(output_errors.shape[0], 1)

        weights1_grad = t.mm(hidden_errors * sigmoid_prime(activation1), X.t()) / batchSize
        bias1_grad = (hidden_errors.sum(dim=1) / batchSize).reshape(hidden_errors.shape[0], 1)
        # print('debug: hidden error', hidden_errors)
        # update weights
        weights2 += learningRate * weights2_grad
        bias2 += learningRate * bias2_grad
        weights1 += learningRate * weights1_grad
        bias1 += learningRate * bias1_grad
        
        i += batchSize

        if i % (trainX.shape[0]) == 0:
            print('epoch:', e + 1, ' error:', error)
    
#%%
def predict(X):

    hidden = t.mm(weights1, X.reshape(-1, 1)) + bias1
    activation1 = sigmoid(hidden)
    # Output Layer
    output = t.mm(weights2, activation1) + bias2
    activation2 = softmax(output)
    return activation2

#%%
guess = t.zeros(1).to(device)

for i in range(testY.shape[0]):
    test = predict(testX[i]).reshape(testY[i].shape)
    if (test == test.max()).nonzero() == (testY[i] == max(testY[i])).nonzero():
        guess += 1

print('accuracy: ', guess/testY.shape[0])
#%%
