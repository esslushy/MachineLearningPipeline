import os
import json
import datetime

import numpy
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

from dataLoader import DataLoader
from wiliLoader import WiliLoader
from preprocessor import Preprocessor
from featureExtractor import FeatureExtractor

class Pipeline(object):

    def __init__(self, configFile):
        #Load the specific configuration file
        self.config = json.load(open(configFile, 'r'))

    def execute(self):
        #Execute the pipeline
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
        #Load the data as specified by the config file
        dataLoader = self.resolve('dataLoader', self.config['dataLoader'])()
        return dataLoader.load(self.config['dataPath'])

    def preprocessData(self, train_data, test_data):
        #Preprocess the data as specified in the config file
        preprocessor = Preprocessor()
        for step in self.config['preprocessing']:
            train_data = preprocessor.process(step, train_data)
            test_data = preprocessor.process(step, test_data)
        return train_data, test_data

    def extractFeatures(self, train_data, test_data):
        fe = FeatureExtractor()
        fe.buildVectorizer(train_data)
        train_vectors = [fe.process(feature, train_data) for feature in self.config['features']]
        if len(train_vectors) > 1:
            train_vectors = numpy.concatenate(train_vectors, axis=1)
        else:
            train_vectors = train_vectors[0]
        test_vectors = [fe.process(feature, test_data) for feature in self.config['features']]
        if len(test_vectors) > 1:
            test_vectors = numpy.concatenate(test_vectors, axis=1)
        else:
            test_vectors = test_vectors[0]
        return train_vectors, test_vectors

    def fitModel(self, train_vectors, train_labels):
        #Fit the model specified in the config file (with specified args)
        model = self.resolve('model', self.config['model'])
        return model().fit(train_vectors, train_labels)

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
