from sklearn.feature_extraction.text import CountVectorizer

class FeatureExtractor():
    def buildVectorizer(self, data):
        """
          Constructs a CountVectorizer based on the given data.

          Args:
            data: Data to train the CountVectorizer
        """
        # Instantiate CountVectorizer
        self.vectorizer = CountVectorizer(analyzer='char_wb')
        # Train CountVectorizer
        self.vectorizer.fit(data)

    def process(self, feature, data):
        """
          Processes a set of data in a way determined by the feature.

          Args:
            feature: Method to extract features from the data
            data: the data to extract features from

          Returns:
            data: The feature extracted data
        """
        # Transform the data
        data = self.vectorizer.transform(data)
        return data