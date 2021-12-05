import numpy as np

from ANN.section_two.utils.utilsV2 import L_layer_model, load_data

path = "/Users/farshid/Desktop/CI/project_assets/Datasets"
train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig = load_data(path)

x_train = np.zeros((102, 1962))
y_train = np.zeros((4, 1962))
for i in range(1962):
    for j in range(102):
        x_train[j, i] = train_set_x_orig[i][j]
for i in range(1962):
    for j in range(4):
        y_train[j, i] = train_set_y_orig[i][j]

x_section_three = x_train[:, 0:200]
y_section_three = y_train[:, 0:200]

parameters = L_layer_model(x_section_three, y_section_three, [102, 150, 60, 4], print_cost=True)
