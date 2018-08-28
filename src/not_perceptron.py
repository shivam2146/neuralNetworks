import numpy as np
import random
from matplotlib import pyplot as plt

sigmoid = lambda x: 1 if(x>=0) else 0

training_data = np.array([[np.array([0,1]),1],
                         [np.array([1,0]),0],
                         ])
learning_rate =0.1
iterations = 100
weight = np.random.rand(1)
weight = np.append(weight,[0])
errors = []
#print initial weight
print("initial weigts",weight)

for i in range(iterations):
    x,y = random.choice(training_data)
    result = np.dot(weight,x)
    error = y - sigmoid(result)
    errors.append(error)
    weight += learning_rate * error * x

#final weights
print("final weight",weight)

#plot error graphs
plt.plot(errors)
plt.show()

#prediction
for x,_ in training_data:
    result = np.dot(weight,x)
    print("{}:{}->{}".format(x[:1],result,sigmoid(result)))
