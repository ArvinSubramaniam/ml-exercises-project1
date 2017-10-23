import numpy as np
import matplotlib.pyplot as plt

'''support functions'''

def cost_function(y, tx,w):
    e = y - tx.dot(w)
    rmse = np.sqrt(e.dot(e)/y.shape[0])             
    return rmse

def compute_gradient(y, tx, w):
    e = y - tx.dot(w)
    #print (tx.dot(w), "dot product in computing gradient")<--For debugging, esp. in case of unnormalised polynomial
    grad = (-1./(y.shape[0]))*tx.T.dot(e)
    return grad

def split_data(y, x, ratio, seed):
    np.random.seed(seed)
    indices = list(range(y.shape[0]))
    np.random.shuffle(indices)

    indices_train = indices[:round(y.shape[0]*ratio)]
    indices_test = indices[round(y.shape[0]*ratio):]
    y_train = y[indices_train]
    y_test = y[indices_test]
    x_train = x[indices_train,:]
    x_test = x[indices_test,:]
    return y_train, x_train, y_test, x_test

def build_poly(x, degree, normalise=True):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi = np.ones((x.shape[0],x.shape[1] * degree + 1))
    
    # normalise values in x
    if normalise:
        for col in range(x.shape[1]):
            if np.linalg.norm(x[:,col]) != 0:
                #print(col)
                x[:,col] = ((x[:,col]-np.mean(x[:,col]))/np.linalg.norm(x[:,col]))*len(x[:,col])
    
    for col in range(x.shape[1]):
            for deg in range(degree):
                phi[:,x.shape[1] *deg + col] = x[:,col]**(deg+1)
    
    return phi

def build_poly_fra(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    # this function should return the matrix formed
    # by applying the polynomial basis to the input data
    # ***************************************************
    num_samples = len(x)
    ones = np.array([np.ones(num_samples)])
    pol = np.asarray([x**power for power in range(1,degree+1)])
    return np.concatenate((ones, pol), axis=0).T

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

'''Project functions'''

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        w -= gamma * compute_gradient(y,tx,w)
        #print(compute_gradient(y,tx,w),"gradient in GD")
        #print (w,"weights in GD")
    loss = cost_function(y,tx,w)#Computes RMSE loss
    #print (loss,"loss in GD")
    return loss, w

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        index = np.random.randint(len(y))#Picking one element of the training set
        y_batch, x_batch = np.array([y[index]]), np.array([tx[index,:]])#"array" to avoid error in compute_gradient
        w -= gamma * compute_gradient(y_batch,x_batch,w)
        #print (compute_gradient(y,tx,w),"gradient in SGD")
        #print (w,"weights in SGD")
    loss = cost_function(y,tx,w)
    #print (loss,"loss in SGD")
    return loss, w

def least_squares(y, tx):
    
    A = np.dot(np.transpose(tx),tx)
    try:
        inverse = np.linalg.inv(A)
    except np.linalg.linalg.LinAlgError as err:
        inverse = np.linalg.pinv(A)
        
    w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    
    return cost_function(y,tx,w), w

def ridge_regression_fra(y, tx, lambda_):
    """implement ridge regression."""
    w = np.dot(np.linalg.inv(np.dot(tx.T, tx) + lambda_/2/np.size(y)*np.identity(np.size(tx[0,:]))), np.dot(tx.T,y))
    #e = y - np.dot(tx, w)
    #loss = np.dot(e.T,e)/2/np.size(y)
    return w

def ridge_regression(y, tx, lambda_):  
    try:
        w = np.linalg.solve(tx.T.dot(tx) + lambda_*(2*tx.shape[0])*np.identity(tx.shape[1]), tx.T.dot(y))
    except np.linalg.linalg.LinAlgError as err:
        A = np.dot(np.transpose(tx),tx) + lambda_/(2*len(y))*np.identity(len(tx))
        inverse = np.linalg.pinv(A)
        w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    return w


'''Plotting functions'''

def cross_validation_visualization(lambds, mse_tr, mse_te):
    """visualization the curves of mse_tr and mse_te."""
    plt.semilogx(lambds, mse_tr, marker=".", color='b', label='train error')
    plt.semilogx(lambds, mse_te, marker=".", color='r', label='test error')
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.title("cross validation")
    plt.legend(loc=2)
    plt.grid(True)
    plt.savefig("cross_validation")
    
def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    y_test, x_test = (y[k_indices[k]],x[k_indices[k],:])

    not_k = [i for i,item in enumerate(y) if i not in k_indices[k]]
    y_train = y[not_k]
    x_train = x[not_k,:]

    # form data with polynomial degree: TODO
    phi_test = build_poly(x_test, degree, False)
    phi_train = build_poly(x_train, degree, False)
    
    weights = ridge_regression(y_train,phi_train,lambda_)
    
    rmse_tr = cost_function(y_train,phi_train,weights)
    
    rmse_te = cost_function(y_test,phi_test,weights)
    
    return rmse_tr, rmse_te

def split_data_cross(y, phi, k, k_indices, degree, seed=1):
  
    y_test, phi_test = (y[k_indices[k]],phi[k_indices[k],:])
    
    not_k = [i for i,item in enumerate(y) if i not in k_indices[k]]
    y_train = y[not_k]
    phi_train = phi[not_k,:]
    
    return y_train, phi_train, y_test, phi_test

    
def predict(y, x, w):
    y_pred = x.dot(w)
    correct = sum(np.sign(y_pred) == y)/len(y)
    return correct*100
