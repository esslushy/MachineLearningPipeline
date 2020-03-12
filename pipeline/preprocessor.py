import numpy as np
import warnings

class Preprocessor():
    def process(self, step, data):
        """
          Processes the data in a certain way defined by step.

          Args:
            step: kind of preprocessing to be done. Either 'fillnan' or 'lowercase'
            data: the data to be processed
          
          Returns:
            data: the preprocessed and cleaned data.
        """
        if step == 'fillnan':
            # Replace all NaNs with empty strings
            data = np.nan_to_num(data, nan='')
        elif step == 'lowercase':
            data = [d.lower() for d in data]
        else:
            warnings.warn('Unknown step typed recieved. No preprocessing will be done.')
        return data