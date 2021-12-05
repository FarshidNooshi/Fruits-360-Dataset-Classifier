import numpy as np
from matplotlib import pyplot as plt

from project_assets.Loading_Datasets import Loader


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_backward(dA, cache):
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


def load_data(path):
    train_set, test_set = Loader.load_train_set_and_test_set(path)
    train_set_x_orig = np.array(train_set[:, 0])
    train_set_y_orig = np.array(train_set[:, 1])

    test_set_x_orig = np.array(test_set[:, 0])
    test_set_y_orig = np.array(test_set[:, 1])

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig


def initialize_parameters_deep(layer_dims):
    layer_dims_copy = np.copy(layer_dims)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.normal(size=(layer_dims_copy[l], layer_dims_copy[l - 1]))
        parameters['b' + str(l)] = np.zeros((layer_dims_copy[l], 1))

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def linear_forward(A, W, b):
    Z = dot(W, A) + b

    assert (Z.shape == (np.shape(W)[0], np.shape(A)[1]))
    cache = (A, W, b)

    return Z, cache


def dot(W, A):
    assert (np.shape(W)[1] == np.shape(A)[0])
    result = np.zeros((np.shape(W)[0], np.shape(A)[1]))
    for i in range(np.shape(W)[0]):
        for j in range(np.shape(W)[1]):
            for k in range(np.shape(A)[1]):
                result[i][k] += W[i][j] * A[j][k]
    return result


def linear_activation_forward(A_prev, W, b):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = sigmoid(Z)
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)
    return A, cache


def L_model_forward(X, parameters):
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


def compute_cost(AL, Y):
    cost = np.sum(np.power(np.subtract(AL, Y), 2))
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache

    dW = dot(dZ, A_prev.T)
    db = dZ
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache
    dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(grads, AL, Y, caches):
    t_grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
    dAL = np.multiply(np.subtract(AL, Y), 2)
    current_cache = caches[L - 1]  # Last Layer
    t_grads["dA" + str(L - 1)], t_grads["dW" + str(L)], t_grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                            current_cache)
    if grads.__contains__("dA" + str(L - 1)):
        grads["dA" + str(L - 1)] += t_grads["dA" + str(L - 1)]
        grads["dW" + str(L)] += t_grads["dW" + str(L)]
        grads["db" + str(L)] += t_grads["db" + str(L)]
    else:
        grads["dA" + str(L - 1)] = t_grads["dA" + str(L - 1)]
        grads["dW" + str(L)] = t_grads["dW" + str(L)]
        grads["db" + str(L)] = t_grads["db" + str(L)]
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache)
        if grads.__contains__("dA" + str(l)):
            grads["dA" + str(l)] += dA_prev_temp
            grads["dW" + str(l + 1)] += dW_temp
            grads["db" + str(l + 1)] += db_temp
        else:
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate, batch_size):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (
                    grads["dW" + str(l + 1)] / batch_size)
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (
                    grads["db" + str(l + 1)] / batch_size)
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=1, num_epochs=5, batch_size=10, print_cost=False):
    costs = []

    parameters = initialize_parameters_deep(layers_dims)
    for i in range(0, num_epochs):
        temp = np.concatenate((X, Y), axis=0)
        temp = temp[:, np.random.permutation(temp.shape[1])]
        X = temp[0:102, :]
        Y = temp[102:, :]
        num_batches = (X.shape[1] + batch_size - 1) // batch_size
        begin = 0
        ss = 0
        cost = 0
        for j in range(0, num_batches):
            grads = {}
            for k in range(batch_size):
                x = X[:, begin:begin + 1]
                y = Y[:, begin:begin + 1]
                AL, caches = L_model_forward(x, parameters)
                p = np.zeros((layers_dims[len(layers_dims) - 1], 1))
                p[np.argmax(AL)] = 1
                AL = p
                cost += compute_cost(AL, y)
                grads = L_model_backward(grads, AL, y, caches)
                ss += np.sum(AL == y) == layers_dims[len(layers_dims) - 1]
                begin += 1
            parameters = update_parameters(parameters, grads, learning_rate, batch_size)

        if print_cost:
            print(f"Cost after epoch {i}: {cost} and accuracy: {str(100.0 * (ss / X.shape[1]))}")
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters
