import os

class WiliLoader():
    def load(self, data_path, num_lines_train, num_lines_test):
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
        train_data = open(os.path.join(data_path, 'x_train.txt')).read().splitlines()[:num_lines_train]
        train_labels = open(os.path.join(data_path, 'y_train.txt')).read().splitlines()[:num_lines_train]
        test_data = open(os.path.join(data_path, 'x_test.txt')).read().splitlines()[:num_lines_test]
        test_labels = open(os.path.join(data_path, 'y_test.txt')).read().splitlines()[:num_lines_test]
        return train_data, train_labels, test_data, test_labels