import math
import numpy as np

def relu(x):
    return np.maximum(0, x)
    

def relu_prime(x):
    return (x > 0).astype(float)
        
        
def softmax(prediction_vector):
    exp_shifted = np.exp(prediction_vector - np.max(prediction_vector))
    probability_distribution = exp_shifted / np.sum(exp_shifted)

    return probability_distribution
    