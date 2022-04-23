import numpy as np
from matplotlib import pyplot as plt

from ANN.section_one.utils.utilsV1 import initialize_parameters_deep, dot, L_model_forward


def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def compute_cost(AL, Y):
    temp = AL - Y
    cost = np.sum((temp * temp))
    cost = np.squeeze(cost)
    return cost


def linear_backward(dZ, cache):
    A_prev, W, b = cache

    dW = np.zeros((np.shape(dZ)[0], np.shape(A_prev)[0]))
    for i in range(np.shape(dZ)[0]):
        for j in range(np.shape(dZ)[1]):
            for k in range(np.shape(A_prev)[0]):
                dW[i][k] += dZ[i][j] * A_prev[k][j]

    db = dZ

    dA_prev = np.zeros((np.shape(W)[1], np.shape(dZ)[1]))
    for i in range(np.shape(W)[1]):
        for j in range(np.shape(W)[0]):
            for k in range(np.shape(dZ)[1]):
                dA_prev[i][k] += W[j][i] * dZ[j][k]

    return dA_prev, dW, db


def linear_activation_backward(dA, cache):
    linear_cache, activation_cache = cache  # linear = (A_prev, W, b), activation = Z
    dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db


def L_model_backward(grads, AL, Y, caches):  # caches = [((A_prev, W, b), Z)]
    t_grads = {}
    L = len(caches)  # the number of layers
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL
    dAL = 2.0 * (AL - Y)
    current_cache = caches[L - 1]  # Last Layer
    t_grads["dA" + str(L - 1)], t_grads["dW" + str(L)], t_grads["db" + str(L)] = linear_activation_backward(dAL,
                                                                                                            current_cache)

    dA_prev = t_grads["dA" + str(L - 1)]
    grads["dA" + str(L - 1)] += t_grads["dA" + str(L - 1)]
    grads["dW" + str(L)] += t_grads["dW" + str(L)]
    grads["db" + str(L)] += t_grads["db" + str(L)]
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(dA_prev, current_cache)
        dA_prev = dA_prev_temp
        grads["dA" + str(l)] += dA_prev_temp
        grads["dW" + str(l + 1)] += dW_temp
        grads["db" + str(l + 1)] += db_temp
    return grads


def update_parameters(parameters, grads, learning_rate, batch_size):
    L = len(parameters) // 2  # number of layers in the neural network

    for l in range(1, L + 1):
        parameters["W" + str(l)] = parameters["W" + str(l)] - (learning_rate * (grads["dW" + str(l)] / batch_size))
        parameters["b" + str(l)] = parameters["b" + str(l)] - (learning_rate * (grads["db" + str(l)] / batch_size))
    return parameters


def init_grads(parameters, layer_dims):
    L = len(parameters) // 2  # number of layers in the neural network
    grads = {}

    for l in range(0, L):
        grads['dW' + str(l + 1)] = np.zeros((layer_dims[l + 1], layer_dims[l]))
        grads['dA' + str(l)] = np.zeros((layer_dims[l], 1))
        grads['db' + str(l + 1)] = np.zeros((layer_dims[l + 1], 1))

    return grads


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
            grads = init_grads(parameters, layers_dims)
            number_of_predictions = min(batch_size, (X.shape[1] - j * batch_size))
            for k in range(number_of_predictions):
                x = X[:, begin:begin + 1]
                y = Y[:, begin:begin + 1]
                AL, caches = L_model_forward(x, parameters)  # caches = [((A_prev, W, b), Z)]
                cost += compute_cost(AL, y)
                grads = L_model_backward(grads, AL, y, caches)
                correct_answers = update_number_of_correct_predictions(AL, correct_answers, y)
                begin += 1
            parameters = update_parameters(parameters, grads, learning_rate=learning_rate,
                                           batch_size=number_of_predictions)
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


def update_number_of_correct_predictions(AL, correct_answers, y):
    predicted_number = np.argmax(AL)
    real_number = np.argmax(y)
    if predicted_number == real_number:
        correct_answers += 1
    return correct_answers
