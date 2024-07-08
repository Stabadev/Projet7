import sys
import os
import unittest

# Affiche le sys.path actuel
print("Current sys.path:", sys.path)

# Ajouter le chemin du répertoire `src` au PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Affiche le sys.path après modification
print("Updated sys.path:", sys.path)

from appV2 import app, get_client_data

class AppTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_index(self):
        result = self.app.get('/site/projet7/')
        self.assertEqual(result.status_code, 200)
        self.assertIn(b'Bienvenue dans l\'interface de pr\xc3\xa9diction de remboursement', result.data)

    def test_predict(self):
        client_id = 201132  # Client ID pour le test
        client_data = get_client_data(client_id)
        if client_data is not None:
            response = self.app.post('/site/projet7/predict', data=dict(client_id=str(client_id)))
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Prediction:', response.data)
            self.assertIn(b'Global Feature Importances', response.data)
            self.assertIn(b'Local Feature Importances', response.data)
        else:
            self.skipTest("Client ID not found in test data")

if __name__ == '__main__':
    unittest.main()
