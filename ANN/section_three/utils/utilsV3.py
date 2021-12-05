import numpy
import numpy as np
from matplotlib import pyplot as plt

from ANN.section_one.utils.utilsV1 import initialize_parameters_deep, L_model_forward, dot


def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    assert (dZ.shape == Z.shape)
    return dZ


def generate_output_layer(AL, layers_dims):
    p = np.zeros((layers_dims[len(layers_dims) - 1], 1))
    p[np.argmax(AL)] = 1
    AL = p
    return AL


def compute_cost(AL, Y):
    cost = np.sum(np.power(np.subtract(AL, Y), 2))
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return cost


def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache
    dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def linear_backward(dZ, cache):
    A_prev, W, b = cache

    dW = dZ @ A_prev.T
    db = dZ
    dA_prev = W.T @ dZ

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

# def linear_backward(dZ, cache):
#     A_prev, W, b = cache
#
#     dW = dot(dZ, A_prev.T)
#     db = dZ
#     dA_prev = dot(W.T, dZ)
#
#     assert (dA_prev.shape == A_prev.shape)
#     assert (dW.shape == W.shape)
#     assert (db.shape == b.shape)
#
#     return dA_prev, dW, db


# def L_model_backward(grads, AL, Y, caches):
#     t_grads = {}
#     L = len(caches)  # the number of layers
#     Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
#     dAL = np.multiply(np.subtract(AL, Y), 2)
#     current_cache = caches[L - 1]  # Last Layer
#     t_grads["dA" + str(L - 1)], t_grads["dW" + str(L)], t_grads["db" + str(L)] = linear_activation_backward(dAL,
#                                                                                                             current_cache)
#
#     dA_prev = np.copy(t_grads["dA" + str(L - 1)])
#     if grads.__contains__("dA" + str(L - 1)):
#         grads["dA" + str(L - 1)].__add__(t_grads["dA" + str(L - 1)])
#         grads["dW" + str(L)].__add__(t_grads["dW" + str(L)])
#         grads["db" + str(L)].__add__(t_grads["db" + str(L)])
#     else:
#         grads["dA" + str(L - 1)] = t_grads["dA" + str(L - 1)]
#         grads["dW" + str(L)] = t_grads["dW" + str(L)]
#         grads["db" + str(L)] = t_grads["db" + str(L)]
#     for l in reversed(range(L - 1)):
#         current_cache = caches[l]
#         dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev, current_cache)
#         dA_prev = np.copy(dA_prev_temp)
#         if grads.__contains__("dA" + str(l)):
#             grads["dA" + str(l)].__add__(dA_prev_temp)
#             grads["dW" + str(l + 1)].__add__(dW_temp)
#             grads["db" + str(l + 1)].__add__(db_temp)
#         else:
#             grads["dA" + str(l)] = dA_prev_temp
#             grads["dW" + str(l + 1)] = dW_temp
#             grads["db" + str(l + 1)] = db_temp
#     return grads

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
    dA_prev = t_grads["dA" + str(L - 1)]
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev, current_cache)
        if grads.__contains__("dA" + str(l)):
            grads["dA" + str(l)] += dA_prev_temp
            grads["dW" + str(l + 1)] += dW_temp
            grads["db" + str(l + 1)] += db_temp
        else:
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp
        dA_prev = dA_prev_temp
    return grads



def update_parameters(parameters, grads, learning_rate, batch_size):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(L):
        avg_dW = np.divide(grads["dW" + str(l + 1)], batch_size)
        avg_db = np.divide(grads["db" + str(l + 1)], batch_size)
        parameters["W" + str(l + 1)] = np.subtract(parameters["W" + str(l + 1)], np.multiply(avg_dW, learning_rate))
        parameters["b" + str(l + 1)] = np.subtract(parameters["b" + str(l + 1)], np.multiply(avg_db, learning_rate))
    return parameters


def L_layer_model(X, Y, layers_dims, learning_rate=1, num_epochs=5, batch_size=10, print_cost=False):
    costs = []

    parameters = initialize_parameters_deep(layers_dims)
    num_batches = (X.shape[1] + batch_size - 1) // batch_size
    for i in range(0, num_epochs):
        temp = np.concatenate((X, Y), axis=0)
        temp = temp[:, np.random.permutation(temp.shape[1])]
        X = temp[0:X.shape[0], :]
        Y = temp[X.shape[0]:, :]
        begin = 0
        correct_answers = 0
        cost = 0
        for j in range(0, num_batches):
            grads = {}
            for k in range(batch_size):
                x = X[:, begin:begin + 1]
                y = Y[:, begin:begin + 1]
                AL, caches = L_model_forward(x, parameters)
                AL = generate_output_layer(AL, layers_dims)
                cost += compute_cost(AL, y)
                grads = L_model_backward(grads, AL, y, caches)
                correct_answers += np.sum(AL == y) == layers_dims[len(layers_dims) - 1]
                begin += 1
            parameters = update_parameters(parameters, grads, learning_rate, batch_size)
        cost /= X.shape[1]
        if print_cost:
            print(f"Cost after epoch {i}: {cost} and accuracy: {str(100.0 * (correct_answers / X.shape[1]))}")
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('epochs')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


