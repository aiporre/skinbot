import unittest
from .models import make_model
class Models(unittest.TestCase):
    def test_make_models(self):
        model_names_all = ['resnet101']
        for mn in model_names_all:
            make_model(mn)

    def test_make_models_with_optimizers(self):
        model_names_all = ['resnet101']
        for mn in model_names_all:
            make_model(mn, optimizer='SGD')
if __name__ == '__main__':
    unittest.main()
