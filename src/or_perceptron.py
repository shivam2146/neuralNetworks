import random
import numpy as np
from matplotlib import pyplot as plt

sigmoid = lambda x: 1 if(x>=0) else 0


training_data = np.array([[np.array([0,0,1]),0],
                        [np.array([0,1,1]),1],
                        [np.array([1,0,1]),1],
                        [np.array([1,1,1]),1]])
learning_rate = 0.1
iterations = 100
errors = []
weight = np.random.rand(2)
weight = np.append(weight,[0])
print("initial weights:",weight)

for i in range(iterations):
    x,y = random.choice(training_data)
    result = np.dot(x,weight)
    error = y - sigmoid(result)
    errors.append(error)
    weight+= learning_rate * error * x
print("final weights:",weight)

#plot graph
plt.plot(errors)
plt.show()

#prediction
for x,_ in training_data:
    result = np.dot(weight,x)
    print("{}:{}->{}".format(x[:2],result,sigmoid(result)))
