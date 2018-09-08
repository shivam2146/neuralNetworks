import random
import numpy as np
from matplotlib import pyplot as plt
import math

def training_data():
    data = np.array([[np.array([0,0]),0],
                     [np.array([0,1]),1],
                     [np.array([1,0]),1],
                     [np.array([1,1]),0],])
    return data[:,0:1][:,0],data[:,1]

def sigmoid(x):
    return (1/(1+math.exp(-x)))

alpha = 0.2
n_layers = 1
epochs = 100
errors = []

def model():
    data_x,data_y = training_data()
    weights_1_1 = np.random.rand(2)
    weights_2_1 = np.random.rand(3)
    bias_1_1 = np.array([0])
    bias_2_1 = np.array([0])
    
    for i in range(epochs):
        for index in range(4):
            result_1_1 = np.dot(data_x[index],weights_1_1)+bias_1_1
            a_1_1 = sigmoid(result_1_1)
            x_2_1 = np.append(data_x[index],a_1_1)
            result_2_1 = np.dot(x_2_1,weights_2_1)+bias_2_1
            a_2_1 = sigmoid(result_2_1)
            error = 1/2 * math.pow((a_2_1-data_y[index]),2)
            print("a_2_1 {} and y_2_1 {}  and error {}".format(a_2_1,data_y[index],error))
            errors.append(error)
            #print("w_2_1 {},{},{}     a_2_1 - y_2_1 = {}- {}".format(weights_2_1[0],weights_2_1[1],weights_2_1[2],a_2_1,data_y[index]))
            w = weights_2_1[2]
            weights_2_1 += alpha * x_2_1 * (a_2_1-data_y[index]) * a_2_1 * (1-a_2_1)
            weights_1_1 += alpha * data_x[index] * a_1_1 * (1-a_1_1) * w * (a_2_1-data_y[index]) * a_2_1 * (1-a_2_1)   
    plt.plot(errors)
    plt.show()
    #print(errors)
    for item in data_x:
        result_1_1 = np.dot(item,weights_1_1)+bias_1_1
        a_1_1 = sigmoid(result_1_1)
        x_2_1 = np.append(item,a_1_1)
        result_2_1 = np.dot(x_2_1,weights_2_1)+bias_2_1
        a_2_1 = sigmoid(result_2_1)
        print("({} {})=> {}".format(item[0],item[1],a_2_1))
            
if __name__ == "__main__":
    model()