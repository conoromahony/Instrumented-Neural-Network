import unittest
import numpy as np
import sys
import os

# The following code says to go to the parent directory to get Show-Neural-Network.py
# os.path.dirname(__file__): This gets the directory of the current script (which is test_Show-Neural-Network.py).
# os.path.abspath(...): This gets the absolute path of the parent directory of test_Show-Neural-Network.py.
# os.path.join(..., '..'): This constructs the parent directory path.
# sys.path.insert(0, ...): This inserts the parent directory path to the beginning of the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Simple-Neural-Network import load_data, init_params, forward_prop, backward_prop, update_params, get_accuracy


class TestNeuralNetwork(unittest.TestCase):
    
    def test_load_data(self):
        # Test loading data from a CSV file
        file_path = 'test_data.csv'
        # Assuming 'test_data.csv' contains sample data in the correct format
        data = load_data(file_path)
        self.assertIsNotNone(data)
        # Add more specific tests for the loaded data if needed
    
    def test_init_params(self):
        # Test initializing parameters
        W1, b1, W2, b2 = init_params()
        self.assertIsNotNone(W1)
        self.assertIsNotNone(b1)
        self.assertIsNotNone(W2)
        self.assertIsNotNone(b2)
        # Add more specific tests for the initialized parameters if needed
    
    def test_forward_backward_prop(self):
        # Test forward and backward propagation
        X = np.random.randn(784, 100)  # Sample input data
        Y = np.random.randint(0, 10, size=(100,))  # Sample labels
        W1, b1, W2, b2 = init_params()
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2, loss = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        self.assertIsNotNone(dW1)
        self.assertIsNotNone(db1)
        self.assertIsNotNone(dW2)
        self.assertIsNotNone(db2)
        self.assertIsNotNone(loss)
        # Add more specific tests for forward and backward propagation if needed
    
    def test_update_params(self):
        # Test updating parameters
        alpha = 0.01  # Sample learning rate
        W1, b1, W2, b2 = init_params()
        dW1 = np.random.randn(*W1.shape)
        db1 = np.random.randn(*b1.shape)
        dW2 = np.random.randn(*W2.shape)
        db2 = np.random.randn(*b2.shape)
        updated_W1, updated_b1, updated_W2, updated_b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        self.assertIsNotNone(updated_W1)
        self.assertIsNotNone(updated_b1)
        self.assertIsNotNone(updated_W2)
        self.assertIsNotNone(updated_b2)
        # Add more specific tests for updating parameters if needed
    
    def test_get_accuracy(self):
        # Test calculating accuracy
        predictions = np.array([0, 1, 2, 3, 4])  # Sample predictions
        labels = np.array([0, 1, 2, 3, 9])  # Sample labels
        accuracy = get_accuracy(predictions, labels)
        self.assertAlmostEqual(accuracy, 0.8, delta=0.01)  # Assuming expected accuracy is 80%
        # Add more specific tests for calculating accuracy if needed

if __name__ == '__main__':
    unittest.main()