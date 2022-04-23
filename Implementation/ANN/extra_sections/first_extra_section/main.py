import time

import numpy as np

from ANN.extra_sections.first_extra_section.utils.utils_extra_V1 import L_layer_model
from ANN.section_one.credentials import get_path_of_Datasets, get_path_of_documents
from ANN.section_one.utils.utilsV1 import load_data, predict


def run_program(file, number_of_epochs, batch_size, learning_rate):
    path = get_path_of_Datasets()
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = load_data(path)
    x_train = np.zeros((102, 1962))
    y_train = np.zeros((4, 1962))
    x_test = np.zeros((102, 662))
    y_test = np.zeros((4, 662))
    for i in range(1962):
        for j in range(102):
            x_train[j, i] = train_set_x_orig[i][j]
    for i in range(1962):
        for j in range(4):
            y_train[j, i] = train_set_y_orig[i][j]
    for i in range(662):
        for j in range(102):
            x_test[j, i] = test_set_x_orig[i][j]
    for i in range(662):
        for j in range(4):
            y_test[j, i] = test_set_y_orig[i][j]

    start_time = time.time()
    train_value, parameters = L_layer_model(x_train, y_train, [102, 150, 60, 4], num_epochs=number_of_epochs, learning_rate=learning_rate, batch_size=batch_size, print_cost=True, file=file)
    test_value = predict(x_test, y_test, parameters, file=file)
    file.write("\n--- %s seconds ---" % (time.time() - start_time))
    return train_value, test_value


with open(f"{get_path_of_documents()}/extra sections/first/report.txt", "w") as f:
    sum_of_trains = 0
    sum_of_tests = 0
    Learning_rate = [0.1, 1, 10]
    Number_of_epochs = [1, 5, 20]
    Batch_size = [1, 10, 100]

    for i in range(3):
        for j in range(3):
            for k in range(3):
                f.write(f"\n---------running program with learning_rate = {Learning_rate[i]}, Number of epochs = {Number_of_epochs[j]}, Batch size = {Batch_size[k]}---------\n")
                val_train, val_test = run_program(f, Number_of_epochs[j], Batch_size[k], Learning_rate[i])

                f.write(f"\n\n----cost in train data is {val_train / 200}-----")
                f.write(f"\n\n----accuracy in test data is {val_test / 662}-----")

