import numpy as 

'''support functions'''

def cost_function(y, tx,w, lambda_=0):
    
    e = y - np.dot(tx,w)
    rmse = np.sqrt(1/len(y)*np.dot(np.transpose(e),e) + 2*lambda_* np.linalg.norm(w)**2)
                   
    return rmse

def compute_gradient(y, tx, w):
    
    e = y - np.dot(tx,w)
    grad = -1/len(y)*np.dot(np.transpose(tx),np.sign(e))
    
    return grad

def split_data(x, y, batch_size, seed=1):
    
    np.random.seed(seed)
    indices = list(range(len(y)))
    np.random.shuffle(indices)
    
    return x[indices[:batch_size]],y[indices[:batch_size]]

'''Project functions'''

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        w -= gamma * compute_gradient(y,tx,w)
        
    loss = cost_function(y,tx,w)

    return loss, w

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        y_batch, x_batch = split_data(tx, y, batch_size, n_iter)
        w -= gamma * compute_gradient(y_batch,x_batch,w)
        
    loss = cost_function(y,tx,w)

    return loss, w

def least_squares(y, tx):
    
    A = np.dot(np.transpose(tx),tx)
    try:
        inverse = np.linalg.inv(A)
    except np.linalg.linalg.LinAlgError as err:
        inverse = np.linalg.pinv(A)
        
    w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    
    return cost_function(y,tx,w), w

#Ridge regression may have some problems but I cant find the error

def ridge_regression(y, tx, lambda_):  
    A = np.dot(np.transpose(tx),tx) - lambda_*(2*len(y))*np.identity(len(tx))
    try:
        inverse = np.linalg.inv(A)
    except np.linalg.linalg.LinAlgError as err:
        inverse = np.linalg.pinv(A)
        
    w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    
    return cost_function(y,tx,w,lambda_), w
