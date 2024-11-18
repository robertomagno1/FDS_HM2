import numpy as np

def sigmoid(x):
    """
    Function to compute the sigmoid of a given input x.

    Args:
        x: it's the input data matrix.

    Returns:
        g: The sigmoid of the input x
    """
    ##############################

    #Here we compute the sigmoid of x by definition

    g = 1 / (1 + np.exp(-x))
    
    ##############################  
    #Here we return the sigmoid of x  
    return g

def softmax(y):
    """
    Function to compute associated probability for each sample and each class.

    Args:
        y: the predicted 

    Returns:
        softmax_scores: it's the matrix containing probability for each sample and each class. The shape is (N, K)
    """

    ##############################
    y_exp = np.exp(y - np.max(y, axis=1, keepdims=True))
    softmax_scores = y_exp / np.sum(y_exp, axis=1, keepdims=True)
    ##############################

    return softmax_scores