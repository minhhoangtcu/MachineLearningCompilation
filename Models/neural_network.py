import numpy as np

# A simple neural network to predict (A and B) or C
#
# Truth table
#   A  B  C  Result
#   0  0  0       0 <
#   0  0  1       1 <
#   0  1  0       0
#   0  1  1       1 <
#   1  0  0       0 <
#   1  0  1       1
#   1  1  0       1 <
#   1  1  1       1

# normalize the input to be in the range of 0 and 1
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# derivate of the output of the sigmoid function
def sigmoid_derivative(z):
    return z * (1 - z)

# shape = (4,3)
X = np.array([[0, 0, 0],
              [0, 0, 1],
              [1, 0, 0],
              [1, 1, 0],
              [0, 1, 1]])

# shape = (4,1), make sure that we do not have a vector of (4,)
y = np.array([[0, 1, 0, 1, 1]]).T

def fit(X, y):
    ''' Train our classifier (the weights) '''
    num_of_attributes = X.shape[1]

    # this is our theta matrix (3, 1), we can use it to predict new input
    # initialize the weights with random number from -1 to 1
    weights = 2*np.random.random((num_of_attributes, 1)) - 1

    for i in range(60000):
        l0 = X # first layer
        l1 = sigmoid(l0.dot(weights)) # second layer (based on the first)

        error = y - l1

        # adjust the weights depending on the direction of the error
        # sigmoid derivative? if l1 is either close to 1 or 0, then our weights are
        # more correct. For example, if you predict something to be 0.5, if that 0
        # or 1? We want binary number, so the only answer is either 0 or 1. So,
        # if the weights result in a prediction of 0.5, it is not so correct.
        #
        # So, if l1 is close to 1 or 0, then the derivative of them is small. Also,
        # if they are close to 1 or 0, then the error is going to be either huge
        # or small (because prediction = 1, but actual is 0, then error is high)
        # So, multiplying them together forces our weights to be more correct.
        # Lastly, error gives us a direction. If actual = 0, and prediction = 1,
        # our error is going to be negative, so, it forces prediction to go down.
        delta = error*sigmoid_derivative(l1)

        # l0.T becomes (3,4), so the first attributes of each sample is going to
        # multiply with their respective delta (to adjust all together). Same for
        # the second and third attributes
        weights += l0.T.dot(delta)

    return weights

def predict(X, weights):
    ''' Predict based on data and the weights. '''
    return sigmoid(X.dot(weights))

classifier = fit(X, y)
print(predict(np.array([1, 1, 1]), classifier))
