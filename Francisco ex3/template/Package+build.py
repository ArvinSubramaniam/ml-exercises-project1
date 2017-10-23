
# coding: utf-8

# The usual imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# General tools for handling data

def standardize(x):
    ''' Normalizes data by subtracting the mean and dividing by standard deviation.
    Works for n-dimensional arrays
    '''
    centered_data = x - np.mean(x, axis=0)
    std_data = centered_data / np.std(centered_data, axis=0)
    
    return std_data


def pairwise(p,q):
    '''Returns the pairwise euclidean distance between the elements in two arrays
    '''
    #return np.sqrt(np.sum((p[:,np.newaxis,:]-q[np.newaxis,:,:])**2, axis=2)) #tensor broadcasting:very efficient
    
    #rows, cols = np.indices((p.shape[0], q.shape[0]))
    #distances = np.sqrt(np.sum((p[rows.ravel(), :] - q[cols.ravel(), :])**2, axis=1))
    #return distances.reshape((p.shape[0], q.shape[0])) #with indices: efficient
    
    #rows, cols = np.indices((p.shape[0], q.shape[0]))
    #distances = np.sqrt(np.sum((p[rows, :] - q[cols, :])**2, axis=2))
    #return distances #with indices, similar to the previous one
    
    return cdist(p, q) #the most efficient


def compute_log_p(X, mean, sigma):
    '''Computes the log-likelihood of a set of data x_n.
    Uses a gaussian distribution.
    Read Lab 1, Task C - ML EPFL 2017'''
    dxm = X - mean
    exponent = -0.5 * np.sum(dxm * np.dot(dxm, np.linalg.inv(sigma)), axis=1)
    return exponent - np.log(2 * np.pi) * (d / 2) - 0.5 * np.log(np.linalg.det(sigma))

#Cost functions

def mse_loss(y, tx, w):
    """Calculates the MSE loss function
        y: data
        tx : x^T
        w: array of weights
        """
    e = y - np.dot(tx, w)
    return np.dot(np.transpose(e), e)/2/np.size(y)

#Grid search

def generate_w(num_intervals):
    """Generate a grid of values for w0 and w1."""
    w0 = np.linspace(-100, 200, num_intervals)
    w1 = np.linspace(-150, 150, num_intervals)
    return w0, w1


def get_best_parameters(w0, w1, losses):
    """Get the best w from the result of grid search."""
    min_row, min_col = np.unravel_index(np.argmin(losses), losses.shape)
    return losses[min_row, min_col], w0[min_row], w1[min_col]

def grid_search(y, tx, w0, w1):
    """Algorithm for grid search.
        Returns the best w's"""#Naive efficiency-wise, but it works. Ask about better solutions
    losses = np.zeros((len(w0), len(w1)))
    for i in range(len(w0)):
        for j in range(len(w1)):
            losses[i,j] = mse_loss(y, tx, np.array([w0[i], w1[j]])) #here mse could be replaced by any cost
    return get_best_parameters(w0, w1, losses)


#Gradient descent

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return -np.dot(np.transpose(tx), y - np.dot(tx, w))/np.size(y)

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = mse_loss(y, tx, w) #again another cost could be used
        w = w -gamma*grad
        ws.append(w)
        losses.append(loss)
    #print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
    #      bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    ws = np.asarray(ws)
    return losses, ws

#Stochastic Gradient Descent

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    return -np.dot(np.transpose(tx), y - np.dot(tx, w))/np.size(y)

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
        Generate a minibatch iterator for a dataset.
        Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
        Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
        Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
        Example of use :
        for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
        """
    data_size = len(y)
    
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def stochastic_gradient_descent(
                                y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_stoch_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w - gamma*grad
            ws.append(w)
            losses.append(loss)
    #print("Gradient Descent:","loss", loss,"w0", w[0], "w1",w[1],"\n")
ws = np.asarray(ws)
return losses, ws


#Least squares

def least_squares(y, tx):
    """calculate the least squares solution.
        y: data array;
        x: transposed x data;
        Returns mse, and optimal weights"""
    w = np.dot(np.linalg.inv(np.dot(tx.T, tx)), np.dot(tx.T,y))
    e = y - np.dot(tx, w)
    mse = np.dot(e.T,e)/2/np.size(y)
    return mse, w

#Build a polynomial basis Phi(x)

def build_poly(x, degree):
    '''Polynomial basis functions for input data x, for j=0 up to j=degree.
        Returns the matrix formed by applying the polynomial basis to the input data'''
    num_samples = len(x)
    ones = np.array([np.ones(num_samples)])
    pol = np.asarray([x**power for power in range(1,degree+1)])
    return np.concatenate((ones, pol), axis=0).T

#Split data to do train vs test

def split_data(x, y, ratio, seed=1):
    '''
        split the dataset based on the split ratio. If ratio is 0.8
        you will have 80% of your data set dedicated to training
        and the rest dedicated to testing
        '''
    # set seed
    np.random.seed(seed)
    # ***************************************************
    x_y = np.c_[(x,y)]
    #print(x_y)
    Ntrain = int(np.size(y)*ratio)
    np.random.shuffle(x_y)
    #print(x_y)
    return x_y.T[0][:Ntrain], x_y.T[1][:Ntrain], x_y.T[0][Ntrain:], x_y.T[1][Ntrain:]

#Ridge regression

def ridge_regression(y, tx, lambda_):
    '''y: output data
        tx: transposed input data vector
        lambda_: ridge parameter multiplying the L-2 norm
        Returns loss and weights'''
    w = np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_/2/np.size(y)*np.identity(np.size(tx[0,:]))), np.dot(tx.T,y))
    e = y - np.dot(tx, w)
    loss = np.dot(e.T,e)/2/np.size(y)
    return loss, w

#Logistic Regression

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t)/(1+np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    return sum(np.log(1+np.exp(np.dot(tx,w))) - y*np.dot(tx,w))

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    return np.dot(tx.T, sigmoid(np.dot(tx,w))-y)

def learning_by_gradient_descent(y, tx, w, gamma):
    """
        Do one step of gradient descen using logistic regression.
        Return the loss and the updated w.
        """
    
    loss = calculate_loss(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w)
    # compute the gradient
    
    # update w
    w = w-gamma*grad
    
    return loss, w

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    diagS = (sigmoid(np.dot(tx, w))*(1- sigmoid(np.dot(tx, w))))
    S = np.diagflat([diagS])
    return np.dot(np.dot(tx.T, S), tx)

def learning_by_newton_method(y, tx, w, gamma):
    """
        Do one step on Newton's method.
        return the loss and updated w.
        """
    # ***************************************************
    # return loss, gradient, hessian
    loss,grad,hess = logistic_regression(y, tx, w)
    
    # update w
    w = w - gamma*np.dot(np.linalg.inv(hess), grad)
    
    return loss, w

def logistic_reg_GD(y, tx, w):
    """Logistic regression by gradient descent"""
    
    loss = calculate_loss(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w)
    # compute the gradient
    
    hess = calculate_hessian(y, tx, w)
    # compute the hessian
    
    max_iter = 100
    gamma = 1
    threshold = 1e-8
    losses = []
    
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    visualization(y, x, mean_x, std_x, w,"General function")
return loss, w

def logistic_reg_newton(y, tx, w, gamma):
    """Logistic Regression by Newton Method"""
    
    loss = calculate_loss(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w)
    # compute the gradient
    
    hess = calculate_hessian(y, tx, w)
    # compute the hessian
    
    max_iter = 100
    threshold = 1e-8
    losses = []
    
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    visualization(y, x, mean_x, std_x, w,"General function")
return loss, w


def logistic_reg_SGD(y, tx, w, batch_size):
    """return the loss, gradient, and hessian."""
    
    loss = calculate_loss(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w)
    # compute the gradient
    
    hess = calculate_hessian(y, tx, w)
    # compute the hessian
    
    max_iter = 10000
    gamma = 0.1
    threshold = 1e-8
    losses = []
    
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_stochastic_gradient_descent(y, tx, w, gamma, batch_size)
        
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    visualization(y, x, mean_x, std_x, w,"General function")
return loss, w

def logistic_reg_penalized(y, tx, w, lambda_):
    """Logistic regression by gradient descent"""
    
    loss = calculate_loss(y, tx, w)
    # compute the cost
    
    grad = calculate_gradient(y, tx, w)
    # compute the gradient
    
    hess = calculate_hessian(y, tx, w)
    # compute the hessian
    
    max_iter = 100
    gamma = 1
    threshold = 1e-8
    losses = []
    
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    visualization(y, x, mean_x, std_x, w,"General function")
return loss, w




