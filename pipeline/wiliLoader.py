import pandas as pd

class WiliLoader():
    def load(self, data_path):
        """
          Loads the WiLI dataset from the designated location.

          Args:
            data_path: The path to the folder containing the data
          
          Returns:
            train_data: An array of the data portion of the training dataset
            train_labels: An array of the label portion of the training dataset
            test_data: An array of the data portion of the testing dataset
            test_labels: An array of the label portion of the testing dataset
        """
        # Open all files and split them by the line breaks.
        train_data = open(data_path + '/x_train.txt').read().splitlines()
        train_labels = open(data_path + '/y_train.txt').read().splitlines()
        test_data = open(data_path + '/x_test.txt').read().splitlines()
        test_labels = open(data_path + '/y_test.txt').read().splitlines()
        return train_data, train_labels, test_data, test_labels