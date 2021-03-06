import numpy as np
from project_assets.Loading_Datasets import Loader


def sigmoid(Z):
    """ Returns sigmoid(Z), Z
    """
    A = 1.0 / (1 + np.exp(-Z))
    cache = Z
    return A, cache


def load_data(path):
    train_set, test_set = Loader.load_train_set_and_test_set(path)
    train_set_x_orig = np.array(train_set[:, 0])
    train_set_y_orig = np.array(train_set[:, 1])

    test_set_x_orig = np.array(test_set[:, 0])
    test_set_y_orig = np.array(test_set[:, 1])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def initialize_parameters_deep(layer_dims):
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.normal(size=(layer_dims[l], layer_dims[l - 1]))
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    """Returns Z, (A, W, b)"""
    Z = dot(W, A) + b

    cache = (A, W, b)

    return Z, cache


def dot(W, A):
    result = np.zeros((np.shape(W)[0], np.shape(A)[1]))
    for i in range(np.shape(W)[0]):
        for j in range(np.shape(W)[1]):
            for k in range(np.shape(A)[1]):
                result[i][k] += W[i][j] * A[j][k]
    return result


def linear_activation_forward(A_prev, W, b):
    """Returns A, ((A_prev, W, b), Z)"""
    # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, parameters):
    """returns AL, caches=[((A_prev, W, b), Z)]"""
    caches = []
    A = X
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)])
        caches.append(cache)

    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)])
    caches.append(cache)

    return AL, caches


def predict(X, y, parameters, file=None):
    m = np.shape(X[0])[0]
    p = np.zeros((4, m))

    # Forward propagation
    prob, caches = L_model_forward(X, parameters)

    # convert prob to 0/1 predictions
    for i in range(0, np.shape(prob)[1]):
        p[np.argmax(prob[:, i]), i] = 1

    ss = 0
    for i in range(m):
        ss += np.sum(p[:, i] == y[:, i]) == 4
    if file is None:
        print("Accuracy: " + str(100.0 * (ss / m)))
    else:
        file.write("Accuracy: " + str(100.0 * (ss / m)))
    return ss
