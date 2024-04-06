#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def precalculate_a(X: np.ndarray) -> np.ndarray:
    
    n,d = X.shape
    a = np.zeros((d,))
    for k in range (d):
        for i in range (n):
            a[k] = a[k] + 2*(X[i][k]**2)
    return a
    
    
    """Precalculate a vector. You should only call this function once.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.

    Returns:
        np.ndarray: An (d, ) array, which contains a corresponding `a` value for each feature.
    """
    #aise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, a: np.ndarray, _lambda: float
) -> Tuple[np.ndarray, float]:
    

    
    n,d = X.shape

    c = np.zeros(d)
    b = 1 / n * np.sum(y - np.dot(X, weight))
    
    for k in range(d):
        c[k] = 2 * np.dot(X[:, k].T, y - (b + np.dot(X, weight) - weight[k] * X[:, k]))
        if c[k] < (-1 * _lambda):
            weight[k] = (c[k] + _lambda) / a[k]
        elif c[k] > _lambda:
            weight[k] = (c[k] - _lambda) / a[k]
        else:
            weight[k] = 0
            
    return weight, b
    
    
    """Single step in coordinate gradient descent.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        a (np.ndarray): An (d,) array. Respresents precalculated value a that shows up in the algorithm.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.
            Bias should be calculated using input weight to this function (i.e. before any updates to weight happen).

    Note:
        When calculating weight[k] you should use entries in weight[0, ..., k - 1] that have already been calculated and updated.
        This has no effect on entries weight[k + 1, k + 2, ...]
    """
    #raise NotImplementedError("Your Code Goes Here")


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    
    return np.sum(np.square(np.dot(X, weight) + bias - y)) + _lambda*(np.sum(np.abs(weight)))
    
    """L-1 (Lasso) regularized MSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    #raise NotImplementedError("Your Code Goes Here")

    
@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, convergence_delta: float
) -> bool:
    
    if (old_w is None):
        return 1
    else:
        return (np.max(np.abs(weight - old_w)) < convergence_delta)
    
    
    """Function determining whether weight has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compate it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of coordinate gradient descent.
        old_w (np.ndarray): Weight from previous iteration of coordinate gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight has not converged yet. True otherwise.
    """
    #raise NotImplementedError("Your Code Goes Here")
    
    

@problem.tag("hw2-A", start_line=4)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    convergence_delta: float = 1e-2,
    start_weight: np.ndarray = None,
) -> Tuple[np.ndarray, float]:
    
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
    a = precalculate_a(X)
    old_w: Optional[np.ndarray] = None
        
    
    stop_criteria = True
    old_w = start_weight
    
    w_new=np.copy(old_w)
    stop=True
    while stop:
        (w_new, b) = step(X, y , w_new, a, _lambda)
        if (convergence_criterion(w_new, old_w, convergence_delta)):
            stop=False
        old_w = np.copy(w_new)  
        

    return (w_new, b)
    
    
    """Trains a model and returns predicted weight.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float .

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    #raise NotImplementedError("Your Code Goes Here")


def dataset (n, d, k, std):
    np.random.seed(0)
    x = np.random.normal(size = (n,d))
    e = np.random.normal(loc = 0, scale = std, size = (n,))
    weights = np.zeros((d,1))
    for j in range(k):
        weights[j] = (j + 1) / k
    y = np.reshape(np.dot(weights.T, x.T) + e.T, (n,))
    lamb = 2*np.max(np.abs(np.dot(y.T - np.mean(y), x))) - 1

    return (x, y, weights, lamb)

@problem.tag("hw2-A")
def main():
    
    n,d,k,std=500,1000,100,1
    (X, Y, W, _lambda) = dataset(n, d, k, std)
    
    
    lambda_factor = 2
    delta = 1e-4

    lambda_curr = _lambda
    lambda_values = [_lambda]

    
    W_hat = np.zeros((d,1))
    W_curr=np.zeros(d)
    stop = True
    while stop:
        (W_curr, bias) = train(X,Y,lambda_curr, delta ,W_curr)
        lambda_curr = lambda_curr / lambda_factor
        lambda_values.append(lambda_curr)
        print(lambda_curr)
        if np.count_nonzero(W_curr) >= 995:
            stop= False
        W_temp=W_curr
        W_temp= np.reshape(W_temp,(d,1))
        W_hat=np.append(W_hat, W_temp, 1)
        
        
    plt.figure(1)
    plt.xscale('log')
    plt.plot(lambda_values, np.count_nonzero(W_hat, axis=0))
    plt.xlabel('Lambda log scale')
    plt.ylabel('Nonzero Coefficients')
    plt.show()


    inc_nonzeros=[]
    
    for j in range(np.size(W_hat[1])):
        count=0
        for i in range(np.size(W)):
            if(W[i]==0 and W_hat[i,j]!=0):
                count=count+1
        inc_nonzeros.append(count)
        
    total_nz=[]
    for j in range(np.size(W_hat[1])):
        total_nz.append(np.count_nonzero(W_hat[:,j]))
        
    FDR= np.divide(inc_nonzeros,total_nz)
    
    cor_nonzeros=[]
    
    for j in range(np.size(W_hat[1])):
        count=0
        for i in range(np.size(W)):
            if(W[i]!=0 and W_hat[i,j]!=0):
                count=count+1
        cor_nonzeros.append(count)
    
    TPR=np.divide(cor_nonzeros,k)
    print(TPR)
    
    
    plt.figure(2)
    plt.plot(FDR, TPR)
    plt.xlabel('FDR')
    plt.ylabel('TDR')
    plt.show()
    
    
if __name__ == "__main__":
    main()







# In[ ]:




