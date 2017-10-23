import numpy as np
import matplotlib.pyplot as plt

'''Project functions'''

def mean_squares_GD(y, tx, w, max_iters, gamma):
    '''perform max_iters of Gradient descent.
    gamma is the step size'''
    
    for n_iter in range(max_iters):
        w -= gamma * compute_gradient(y,tx,w)
        
    loss = cost_function(y,tx,w)

    return loss, w

def mean_squares_SGD(y, tx, w, max_iters, gamma):
    '''perform max_iters of Stochastic GD'''
    
    for n_iter in range(max_iters):
        
        index = np.random.randint(len(y))
        y_batch, x_batch = y[index], tx[index,:]
        
        w -= gamma * compute_gradient(y_batch,x_batch,w)
        
    loss = cost_function(y,tx,w)

    return loss, w

def least_squares(y, tx):
    '''perform least squares solving the linear system
    it handles the exeption for singular matrices'''
    try:
        w = np.linalg.solve(tx.T.dot(tx), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = np.dot(np.transpose(tx),tx)
        inverse = np.linalg.pinv(A)
        w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    
    return cost_function(y,tx,w), w

def ridge_regression(y, tx, lambda_):  
    '''Ridge solving the linear system, handles the case of singular matrices'''
    try:
        w = np.linalg.solve(tx.T.dot(tx) + lambda_*(2*tx.shape[0])*np.identity(tx.shape[1]), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = np.dot(np.transpose(tx),tx) + lambda_/(2*len(y))*np.identity(len(tx))
        inverse = np.linalg.pinv(A)
        w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    return w

'''support functions'''

def cost_function(y, tx,w):
    '''compute the cost function as mean square error.
    Returns the root mse.'''
    e = y - tx.dot(w)
    rmse = np.sqrt(e.dot(e)/y.shape[0])             
    return rmse

def compute_gradient(y, tx, w):
    '''compute the gradient for the mean square error function'''
    e = y - tx.dot(w)
    grad = -1/y.shape[0]*tx.T.dot(e)
    return grad

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

def split_data(y, x, ratio, seed=1):
    '''given y (vector) and x (matrix), return two vectors (train + test) with elements randomly taken from the two.
    ratio is the relative size of the train examples.'''
    np.random.seed(seed)
    indices = list(range(x.shape[0]))
    np.random.shuffle(indices)
    
    indices_train = indices[:round(x.shape[0]*ratio)]
    indices_test = indices[round(x.shape[0]*ratio):]
    
    return y[indices_train],x[indices_train,:],y[indices_test],x[indices_test,:]

def split_data_cross(y, phi, k, k_indices, degree, seed=1):
  
    y_test, phi_test = (y[k_indices[k]],phi[k_indices[k],:])
    
    not_k = [i for i,item in enumerate(y) if i not in k_indices[k]]
    y_train = y[not_k]
    phi_train = phi[not_k,:]
    
    return y_train, phi_train, y_test, phi_test

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
    
def predict(y, x, w):
    y_pred = x.dot(w)
    correct = sum(np.sign(y_pred) == y)/len(y)
    return correct*100

'''Plotting functions'''

def semilog_loss_lambda_plot(loss, lambdas, seed, degree):
    plt.title("lambda vs loss for seed = %i and degree = %i" %(seed, degree))
    plt.xlabel("lambda")
    plt.ylabel("loss")
    plt.semilogx(lambdas, loss, 'r')
    plt.savefig("lambda_vs_loss_simple_splitting_ridge_seed%i.png" %seed)
