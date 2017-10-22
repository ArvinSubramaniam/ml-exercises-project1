import numpy as np

'''support functions'''

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
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

def cost_function(y, tx,w, lambda_=0):
    
    e = y - np.dot(tx,w)
    rmse = np.sqrt(1/y.shape[0]*np.dot(np.transpose(e),e) + 2*lambda_* np.linalg.norm(w)**2)
                   
    return rmse

def compute_gradient(y, tx, w):
    
    e = y - np.dot(tx,w)
    #grad = -1*np.dot(np.transpose(tx),np.sign(e))
    grad = -1.*np.dot(np.transpose(tx),e)
    
    return grad

def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient from just few examples n and their corresponding y_n labels."""
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        grad_stoch = compute_gradient(minibatch_y, minibatch_tx, w)
    return grad_stoch

def split_data(x, y, ratio, seed=1):
    np.random.seed(seed)
    indices = list(range(len(y)))
    np.random.shuffle(indices)
    
    indices_train = indices[:round(len(y)*ratio)]
    indices_test = indices[round(len(y)*ratio):]
    
    return x[indices_train,:],y[indices_train],x[indices_test,:],y[indices_test]

'''Project functions'''

def gradient_descent(y, tx, initial_w, max_iters, gamma):
    w = initial_w
    
    for n_iter in range(max_iters):
        w -= gamma * compute_gradient(y,tx,w)
        
    loss = cost_function(y,tx,w)

    return loss, w

def stoch_grad_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        losses = gradient_descent(minibatch_y, minibatch_tx,initial_w, max_iters, gamma)[0]
        ws = gradient_descent(minibatch_y, minibatch_tx,initial_w, max_iters, gamma)[1]
    return losses, ws

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
    A = np.dot(np.transpose(tx),tx) + lambda_/(2*len(y))*np.identity(len(tx))
    try:
        inverse = np.linalg.inv(A)
    except np.linalg.linalg.LinAlgError as err:
        inverse = np.linalg.pinv(A)
        
    w = np.dot(np.dot(inverse,np.transpose(tx)),y)
    
    return cost_function(y,tx,w,lambda_), w
