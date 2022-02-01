import numpy as np
import random
import pickle


class Loader:

    @staticmethod
    def load_train_set_and_test_set(path):
        # loading training set features
        with open(path + "/new/train_set_features.pkl", "rb") as f:
            train_set_features2 = pickle.load(f)

        # reducing feature vector length
        features_STDs = np.std(a=train_set_features2, axis=0)
        train_set_features = train_set_features2[:, features_STDs > 52.3]

        # changing the range of data between 0 and 1
        train_set_features = np.divide(train_set_features, train_set_features.max())

        # loading training set labels
        with open(path + "/new/train_set_labels.pkl", "rb") as f:
            train_set_labels = pickle.load(f)

        # ------------
        # loading test set features
        with open(path + "/new/test_set_features.pkl", "rb") as f:
            test_set_features2 = pickle.load(f)

        # reducing feature vector length
        features_STDs = np.std(a=test_set_features2, axis=0)
        test_set_features = test_set_features2[:, features_STDs > 47.7]

        # changing the range of data between 0 and 1
        test_set_features = np.divide(test_set_features, test_set_features.max())

        # loading test set labels
        with open(path + "/new/test_set_labels.pkl", "rb") as f:
            test_set_labels = pickle.load(f)

        # ------------
        # preparing our training and test sets - joining datasets and lables
        train_set = []
        test_set = []

        for i in range(len(train_set_features)):
            label = np.array([0, 0, 0, 0, 0, 0])
            label[int(train_set_labels[i])] = 1
            label = label.reshape(6, 1)
            train_set.append((train_set_features[i].reshape(119, 1), label))

        for i in range(len(test_set_features)):
            label = np.array([0, 0, 0, 0, 0, 0])
            label[int(test_set_labels[i])] = 1
            label = label.reshape(6, 1)
            test_set.append((test_set_features[i].reshape(119, 1), label))

        # shuffle
        random.shuffle(train_set)
        random.shuffle(test_set)

        # print size
        # print(len(train_set), np.shape(train_set))  # 1962
        # print(len(test_set))  # 662
        train_set = np.array(train_set)
        test_set = np.array(test_set)
        return train_set, test_set
