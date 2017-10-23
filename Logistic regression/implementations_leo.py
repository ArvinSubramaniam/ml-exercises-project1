import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *

'''support functions'''

def cost_function(y, x, w):
    loss = 0
    for n, y_n in enumerate(y):
        z = x[n,:].T.dot(w)
        loss += np.log(1 + np.exp(z)) - y_n * z
        
    return loss

def predict_y(x,w):
    y_pred = sigmoid(x.dot(w)) >= 0.5
    y_pred = [-1 if t == 0 else t for t in y_pred]
    return y_pred

def sigmoid(t):
    expt = np.exp(t)
    expt[np.isnan(expt)] = 1e50
    return expt / (1+expt)

def compute_gradient(y, x, w):
    return x.T.dot(sigmoid(x.dot(w)) - y)

def GD_step(y, x, w, gamma):
    w -= gamma * compute_gradient(y, x, w)
    return w

def SGD_step(y, x, w, gamma, batch_size):
    y_batch, x_batch, _, __ = batch_data(y, x, batch_size)
    print(w)
    w -= gamma * compute_gradient(y_batch, x_batch, w)
    return w


def batch_data(y, x, batch_size, seed=1):
    np.random.seed(seed)
    ids = list(range(len(y)))
    np.random.shuffle(ids)
    return y[ids[:batch_size]], x[ids[:batch_size],:], y[ids[batch_size:]], x[ids[batch_size:],:]

def build_poly(x, degree, normalise=False):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    normalise = True first subtract the mean and divide for the sigma each column"""
    phi = np.ones((x.shape[0],x.shape[1] * degree + 1))
    
    # normalise values in x
    
    for col in range(x.shape[1]):
            for deg in range(degree):
                phi[:,x.shape[1] *deg + col] = x[:,col]**(deg+1)
    if normalise:
        for col in range(phi.shape[1]):
            phi[:,col] = (phi[:,col]-np.mean(phi[:,col]))/np.linalg.norm(phi[:,col])*len(phi[:,col])
    
    return phi
               
def create_submission(x_sub, w, degree, filename="predictions.csv"):
    phi_sub = build_poly(x_sub, degree) 
    y_sub = predict_y(phi_sub, w)
    create_csv_submission(ids_submission, y_predicted, filename)