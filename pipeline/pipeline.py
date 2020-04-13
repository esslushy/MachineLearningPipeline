import os
import json
import datetime
import pickle

import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from dataLoader import DataLoader
from wiliLoader import WiliLoader
from preprocessor import Preprocessor
from featureExtractor import FeatureExtractor

class Pipeline(object):

    def __init__(self, configFile):
        # Load the specific configuration file
        self.config = json.load(open(configFile, 'r'))
        # Build the paths for the pipeline
        self.experiment_path = os.path.join('../', self.config['experimentName'])
        self.preprocessing_path = os.path.join(self.experiment_path, self.config['preprocessingPath'])
        self.feature_path = os.path.join(self.experiment_path, self.config['featurePath'])
        self.model_path = os.path.join(self.experiment_path, self.config['modelPath'])
        self.metric_path = os.path.join(self.experiment_path, self.config['metricPath'])
        # Create the folder for the experiment if it doesn't already exist
        if not os.path.exists(self.experiment_path):
            os.mkdir(self.experiment_path)

    def execute(self):
        # Execute the pipeline
        print('Loading Data - ' + self.timestamp())
        train_data, train_labels, test_data, test_labels = self.loadData()
        print('Preprocessing Data - ' + self.timestamp())
        clean_train, clean_test = self.preprocessData(train_data, test_data)
        print('Extracting Features - ' + self.timestamp())
        train_vectors, test_vectors = self.extractFeatures(clean_train, clean_test)
        print('Training Model - ' + self.timestamp())
        model = self.fitModel(train_vectors, train_labels)
        print('Evaluating Model - ' + self.timestamp())
        self.evaluate(model, test_vectors, test_labels)

    def loadData(self):
        # Load the data as specified by the config file
        dataLoader = self.resolve('dataLoader', self.config['dataLoader'])()
        return dataLoader.load(self.config['dataPath'])

    def preprocessData(self, train_data, test_data):
        # Preprocessor
        preprocessor = Preprocessor()
        # Make preprocessing path if it doesnt exist
        if not os.path.exists(self.preprocessing_path):
            os.mkdir(self.preprocessing_path)
        # Check if preprocessing training artifact exists
        if os.path.exists(os.path.join(self.preprocessing_path, 'train_data.txt')):
            # Load train data if it does
            train_data = open(os.path.join(self.preprocessing_path, 'train_data.txt')).read().splitlines()
        else:
            # Preprocess the data as specified in the config file
            for step in self.config['preprocessing']:
                train_data = preprocessor.process(step, train_data)
            # Save the training data artifact
            with open(os.path.join(self.preprocessing_path, 'train_data.txt'), 'w+') as f:
                # Write the array with each datapoint on a new line
                f.write('\n'.join(train_data))
                f.close()
        # Check if preprocessing testing artifact exists
        if os.path.exists(os.path.join(self.preprocessing_path, 'test_data.txt')):
            # Load test data if it does
            test_data = open(os.path.join(self.preprocessing_path, 'test_data.txt')).read().splitlines()
        else:
            # Preprocess the data as specified in the config file
            for step in self.config['preprocessing']:
                test_data = preprocessor.process(step, test_data)
            # Save the testing data artifact
            with open(os.path.join(self.preprocessing_path, 'test_data.txt'), 'w+') as f:
                # Write the array with each datapoint on a new line
                f.write('\n'.join(test_data))
                f.close()
        return train_data, test_data

    def extractFeatures(self, train_data, test_data):
        # Construct Feature Extractor
        fe = FeatureExtractor()
        fe.buildVectorizer(train_data)
        # Make feature path if it doesnt exist
        if not os.path.exists(self.feature_path):
            os.mkdir(self.feature_path)
        # Check if train vectors already exist
        if os.path.exists(os.path.join(self.feature_path, 'train_vectors.npy')):
            # If it does, load them
            train_vectors = numpy.load(os.path.join(self.feature_path, 'train_vectors.npy'))
        else:
            # Make the train vectors
            train_vectors = [fe.process(feature, train_data) for feature in self.config['features']]
            if len(train_vectors) > 1:
                train_vectors = numpy.concatenate(train_vectors, axis=1)
            else:
                train_vectors = train_vectors[0]
            # Save the train vectors
            numpy.save(os.path.join(self.feature_path, 'train_vectors.npy'), train_vectors)
        # Check if test vectors already exist
        if os.path.exists(os.path.join(self.feature_path, 'test_vectors.npy')):
            # If it does, load them
            test_vectors = numpy.load(os.path.join(self.feature_path, 'test_vectors.npy'))
        else:
            # Make the test vectors
            test_vectors = [fe.process(feature, test_data) for feature in self.config['features']]
            if len(test_vectors) > 1:
                test_vectors = numpy.concatenate(test_vectors, axis=1)
            else:
                test_vectors = test_vectors[0]
            # Save the test vectors
            numpy.save(os.path.join(self.feature_path, 'test_vectors.npy'), test_vectors)
        return train_vectors, test_vectors

    def fitModel(self, train_vectors, train_labels):
        # Make model path if it doesnt exist
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        # Check if model exists
        if os.path.exists(os.path.join(self.model_path, self.config['model'].replace(' ', '_') + '.pickle')):
            # If it does, load it
            model = pickle.load(open(os.path.join(self.model_path, self.config['model'].replace(' ', '_') + '.pickle'), 'rb'))
        else:
            # Fit the model specified in the config file (with specified args)
            model = self.resolve('model', self.config['model'])
            model = model().fit(train_vectors, train_labels)
            # Save the model
            pickle.dump(model, open(os.path.join(self.model_path, self.config['model'].replace(' ', '_') + '.pickle'), 'wb'))
        return model

    def evaluate(self, model, test_data, test_labels):
        #Evaluate using the metrics specified in the config file
        predictions = model.predict(test_data)
        results = {}
        for metric in self.config['metrics']:
            results[metric] = self.resolve('metrics', metric)(predictions, test_labels)
        self.output(results)
        print(results)

    def output(self, results):
        output_file = os.path.join(self.config['outputPath'], self.config['experimentName'])
        F = open(output_file, 'w')
        F.write(json.dumps(self.config) + '\n')
        for metric in results:
            F.write(metric + ',%f\n' % results[metric])
        F.close()

    def resolve(self, category, setting):
        #Resolve a specific config string to function pointers or list thereof
        configurations = {'dataLoader': {'baseLoader': DataLoader,
                                         'WiliLoader': WiliLoader},
                          'model': {'Naive Bayes': GaussianNB},
                          'metrics': {'accuracy': accuracy_score}
                         }
        #These asserts will raise an error if the config string is not found
        assert category in configurations
        assert setting in configurations[category]
        return configurations[category][setting]

    def timestamp(self):
        now = datetime.datetime.now()
        return ('%02d:%02d:%02d' % (now.hour, now.minute, now.second))

if __name__ == '__main__':
    p = Pipeline('config.json')
    p.execute()
