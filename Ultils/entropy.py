import math
import numpy as np
from scipy.stats import entropy as entropy_content

def entropy(data):
    """
    Return the measure of impurity in the provided binary data
    """
    classes = np.unique(data)
    occurances = np.bincount(data)
    totalData = len(data)
    result = 0

    # scale entropy, else if we always use base 2, the max value is logk for
    # k to be the number of classes
    base = 2 if len(classes) <= 1 else len(classes)

    for i in range(len(classes)):
        occurance = occurances[classes[i]]
        p_i = occurance / totalData
        result += -(p_i * math.log(p_i, base))

    return result

print(entropy([1,1,1]))

print(entropy([0,0,0,0,0,0,0,0,0,1,1,1,1,1]))
print(entropy_content([5/14, 9/14], base=2))

print(entropy([0,1,2]))
print(entropy_content([1/3, 1/3, 1/3], base=3))
