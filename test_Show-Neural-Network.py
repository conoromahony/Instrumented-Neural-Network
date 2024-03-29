import unittest
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