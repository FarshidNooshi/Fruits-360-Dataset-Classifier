import time

import numpy as np

from ANN.extra_sections.third_extra_section.utils.utils_extra_V3 import L_layer_model, load_data, predict
from ANN.section_one.credentials import get_path_of_Datasets, get_path_of_documents


def run_program(file):
    path = get_path_of_Datasets()
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = load_data(path)
    x_train = np.zeros((119, 2879))
    y_train = np.zeros((6, 2879))
    x_test = np.zeros((119, 971))
    y_test = np.zeros((6, 971))
    for i in range(1962):
        for j in range(119):
            x_train[j, i] = train_set_x_orig[i][j]
    for i in range(1962):
        for j in range(6):
            y_train[j, i] = train_set_y_orig[i][j]
    for i in range(971):
        for j in range(119):
            x_test[j, i] = test_set_x_orig[i][j]
    for i in range(971):
        for j in range(6):
            y_test[j, i] = test_set_y_orig[i][j]

    start_time = time.time()
    train_value, parameters = L_layer_model(x_train, y_train, [119, 150, 60, 10, 6], num_epochs=10, print_cost=True, file=file)
    test_value = predict(x_test, y_test, parameters, file=file)
    file.write("\n--- %s seconds ---" % (time.time() - start_time))
    return train_value, test_value


with open(f"{get_path_of_documents()}/extra sections/third/report.txt", "w") as f:
    sum_of_trains = 0
    sum_of_tests = 0
    number_of_iterations = 5
    for i in range(number_of_iterations):
        f.write(f"\nrunning program with i = {i}\n")
        val_train, val_test = run_program(f)
        sum_of_trains += val_train
        sum_of_tests += val_test

    f.write(f"\n\n----average cost in train data is {sum_of_trains / number_of_iterations * 10}-----")
    f.write(f"\n\n----average accuracy in test data is {sum_of_tests / (971 * number_of_iterations)}-----")

