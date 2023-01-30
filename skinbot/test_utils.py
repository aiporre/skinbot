from unittest import TestCase
from skinbot.utils import change_models_to_names


class Test(TestCase):
    def test_change_models_to_eq(self):
        model_path = "../models"
        change_models_to_names(model_path, to_equal_sign=True)
