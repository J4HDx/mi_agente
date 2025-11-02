import unittest
import pandas as pd
from utils.data_utils import load_data, preprocess_data, split_data

class TestDataUtils(unittest.TestCase):

    def test_load_data(self):
        data = load_data('data/raw/dataset_original.csv')
        self.assertIsInstance(data, pd.DataFrame)

    def test_preprocess_data(self):
        data = pd.DataFrame({'A': [1, 2, None], 'B': [4, None, 6]})
        processed_data = preprocess_data(data)
        self.assertFalse(processed_data.isnull().values.any())

    def test_split_data(self):
        data = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8]})
        X_train, X_test, y_train, y_test = split_data(data)
        self.assertEqual(len(X_train), 3)
        self.assertEqual(len(X_test), 1)

if __name__ == '__m
ain__':
    unittest.main()
