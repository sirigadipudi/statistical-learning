#!/usr/bin/env python
# coding: utf-8

# In[11]:


if __name__ == "__main__":
    from coordinate_descent_algo import train  # type: ignore
else:
    from .coordinate_descent_algo import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem

def error(X, y, weights, bias):
    return mse(y, np.dot(X, weights) + bias)


def mse(a: np.ndarray, b: np.ndarray):
    return np.mean(np.square(np.subtract(a, b)))

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")
    
    #loading data from the file and converting into numpy array since coordinate descent was done using numpy
    #loaded train and test from the dataset
    Y_train=np.array(df_train['ViolentCrimesPerPop'])
    X=np.array(df_train)
    X_train=X[:,1:]
    
    
    Y_test=np.array(df_test['ViolentCrimesPerPop'])
    X=np.array(df_test)
    X_test=X[:,1:]
    
    #setting weights and initialising lambda value to max
    W=np.zeros(np.shape(X_train[1]))
    _lambda = 2 * np.max(np.abs(np.dot(Y_train.T - np.mean(Y_train), X_train))) - 1
    lambda_factor = 2
    delta = 1e-4
    
    lambdaa=_lambda
    lambda_values=[]
    train_mse,test_mse = [],[]
 
    W_hat = np.zeros((X_train.shape[1], 16))
    i=0
    
    #training the data for different lambda values until lambda <0.01
    while (lambdaa > 0.01):
        lambda_values.append(lambdaa)
        (W, bias) = train(X_train,Y_train,lambdaa, delta ,W)
        W_hat[:,i]=W
        i=i+1
        
        train_mse.append(error(X_train,Y_train, W, bias))
        test_mse.append(error(X_test, Y_test, W, bias))
        
        print(lambdaa)
        lambdaa = lambdaa / lambda_factor
    
    
    (W, bias) = train(X=X_train,y=Y_train,_lambda=30)
    print(np.argmax(W, axis=0))
    print(np.argmin(W, axis=0))
    print(df_train.iloc[:, 1:].columns[np.argmax(W, axis=0)])
    print(df_train.iloc[:, 1:].columns[np.argmin(W, axis=0)])
    print(np.count_nonzero(W_hat, axis=0))
    
    #plotting all the graphs including nonzero coefficients, regularised paths and error using matplotlib 
    plt.figure(1)
    plt.xscale('log')
    plt.plot(lambda_values, np.count_nonzero(W_hat, axis=0))
    plt.xlabel('Lambda log scale ')
    plt.ylabel('Nonzero Coefficients crime_data')
    plt.show()


    x1= df_train.columns.get_loc('agePct12t29')-1
    agepct= W_hat[x1,:]
    
    x2=df_train.columns.get_loc('pctWSocSec')-1
    pctwsocsec= W_hat[x2,:]
    
    x3=df_train.columns.get_loc('pctUrban')-1
    pctUrban= W_hat[x3,:]
        
    x4=df_train.columns.get_loc('agePct65up')-1
    agePct65up= W_hat[x4,:]
    
    x5=df_train.columns.get_loc('householdsize')-1
    householdsize= W_hat[x5,:]
    
    
    
    plt.figure(2)
    plt.xscale('log')
    plt.plot(lambda_values, agepct,color='g', label='agePct12t29' )
    plt.plot(lambda_values, pctwsocsec, color='r', label='pctWSocSec')
    plt.plot(lambda_values, pctUrban,color='y', label='pctUrban' )
    plt.plot(lambda_values, agePct65up,color='b', label='agePct65up' )
    plt.plot(lambda_values, householdsize,color='c', label='householdsize' )
    plt.xlabel('Lambda log scale')
    plt.ylabel('Nonzero Coefficients ')
    plt.legend()
    plt.show()
   
    plt.plot(lambda_values, train_mse)
    plt.plot(lambda_values, test_mse)
    plt.xlabel("lambda log scale")
    plt.ylabel("Mean squared error of training and test data")
    plt.xscale("log")
    plt.legend(['train mse', 'test mse'])
    plt.show()
    
    
    
if __name__ == "__main__":
    main()


# In[ ]:




