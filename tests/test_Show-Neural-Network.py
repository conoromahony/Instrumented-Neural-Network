import unittest
import sys
import os

# The following code says to go to the parent directory to get Show-Neural-Network.py
# os.path.dirname(__file__): This gets the directory of the current script (which is test_Show-Neural-Network.py).
# os.path.abspath(...): This gets the absolute path of the parent directory of test_Show-Neural-Network.py.
# os.path.join(..., '..'): This constructs the parent directory path.
# sys.path.insert(0, ...): This inserts the parent directory path to the beginning of the Python path.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Show-Neural-Network import app


class TestApp(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_first_iteration_route(self):
        response = self.app.get('/first')
        self.assertEqual(response.status_code, 200)

    def test_previous_iteration_route(self):
        response = self.app.get('/previous')
        self.assertEqual(response.status_code, 200)

    def test_next_iteration_route(self):
        response = self.app.get('/next')
        self.assertEqual(response.status_code, 200)

    def test_last_iteration_route(self):
        response = self.app.get('/last')
        self.assertEqual(response.status_code, 200)

    def test_iteration_route(self):
        # Test for an existing iteration
        response = self.app.get('/iteration/0')
        self.assertEqual(response.status_code, 200)
        
        # Test for a non-existing iteration (should return 404)
        response = self.app.get('/iteration/1000')
        self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()