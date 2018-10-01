import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

def get_data():
    names = [i for i in range(785)]
    data = pd.read_csv('images.csv',names=names)
    X = data.loc[:,1:]
    X = X/255
    Y = data.loc[:,0]
    enc = OneHotEncoder()
    Y = enc.fit_transform(np.array(Y).reshape(-1,1))

    #split the data into training (80%) and testing (20%)
    (X_train, X_test, Y_train, Y_test) = train_test_split(X, Y, test_size=0.20)
    return X_train.values, X_test.values, Y_train.toarray(), Y_test.toarray()


def predictor():
    X_train, X_test ,Y_train, Y_test  = get_data()
    N = X_train.shape[1]

    #Defining number of neuron units in each layer
    neurons1 = 512
    neurons2 = 256
    neurons3 = 128
    outputl = 26

    #Session creation
    session = tf.InteractiveSession()

    #Placeholders for inputs
    X = tf.placeholder(dtype = tf.float32, shape =[None,N])
    Y = tf.placeholder(dtype = tf.float32, shape = [None,26])

    #Initialisers for weights and biases
    sigma = 1
    weights_in = tf.random_normal_initializer()
    biases_in = tf.zeros_initializer()

    #hidden weights
    w_1 = tf.Variable(weights_in([N,neurons1]))
    b_1 = tf.Variable(biases_in([neurons1]))
    w_2 = tf.Variable(weights_in([neurons1,neurons2]))
    b_2 = tf.Variable(biases_in([neurons2]))
    w_3 = tf.Variable(weights_in([neurons2,neurons3]))
    b_3 = tf.Variable(weights_in([neurons3]))

    #Output weights
    out_w = tf.Variable(weights_in([neurons3,outputl]))
    out_b = tf.Variable(biases_in(outputl))


    #Saver to save weights
    saver = tf.train.Saver()


        #256*784  784*512  = 256*512
        #256*512  512*256  = 256*256
        #256*256  256*128  = 256*128
        #256*128  128*26 = 256*26

    #Defining hidden layers
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X,w_1),b_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1,w_2),b_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2,w_3),b_3))

    #Output layer
    output = tf.add(tf.matmul(hidden_3,out_w),out_b)

    #Cost function
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = tf.nn.softmax(output), labels = Y))
    #Optimizer(Stochastic gradient extension)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    #Calling Initialisers
    session.run(tf.global_variables_initializer())

    #Plotting
    plt.ion()
    fig = plt.figure(figsize = (10,12))

    #fit neural sessionwork
    batch_size = 50
    mse_train = []
    mse_test = []

    epoches = 100
    for e in range(0,epoches):

        #shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(Y_train)))
        X_train = X_train[shuffle_indices]
        Y_train = Y_train[shuffle_indices]

        #minibatch training
        for i in range(0,len(Y_train)//batch_size):
            start = i*batch_size
            batch_x = X_train[start:start+batch_size]
            batch_y = Y_train[start:start+batch_size]

            #run optimizer
            session.run(optimizer,feed_dict={X:batch_x,Y:batch_y})
            saver.save(session,'model_data/',global_step =100 ,write_meta_graph=False)
            #show progress
            if np.mod(i,50) == 0:
                #MSE train and test
                mse_train.append(session.run(cost,feed_dict={X:X_train,Y:Y_train}))
                mse_test.append(session.run(cost,feed_dict={X:X_test,Y:Y_test}))
                print('MSE Train: ', mse_train[-1])
                print('MSE Test: ', mse_test[-1])
                """
                #prediction
                pred = session.run(out,feed_dict={X:X_test})
                print(pred.shape)
                print(pred)
                """



# Define the loss function
#print(loss.eval())
    correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(output), 1), tf.argmax(Y,1))
# Operation calculating the accuracy of our predictions
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(session.run(accuracy,feed_dict={X:X_test,Y:Y_test}))
# Operation comparing prediction with true label


if __name__ == "__main__":
    predictor()
