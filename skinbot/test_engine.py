from unittest import TestCase
from skinbot.engine import keep_best_two
import os
import tempfile

class ModelPaths(TestCase):
    def test_keep_best_two(self):
        # get a temporary directory
        filenames = ['best_fold=0_resnet101_cropSingle_model_10_accuracy=0.4136.pt',
                     'best_fold=0_resnet101_cropSingle_model_10_accuracy=0.4146.pt',
                     'best_fold=0_resnet101_cropSingle_model_10_accuracy=0.9999.pt']

        expected_filenames = ['best_fold=0_resnet101_cropSingle_model_10_accuracy=0.4146.pt',
                     'best_fold=0_resnet101_cropSingle_model_10_accuracy=0.9999.pt']
        with tempfile.TemporaryDirectory() as tmpdirname:
            # create a empty file
            for filename in filenames:
                with open(os.path.join(tmpdirname, filename), 'w') as f:
                    pass
            print('======>>>>  before')
            for f in os.listdir(tmpdirname):
                print(f)
            # keep only the best two
            keep_best_two(tmpdirname, 0, 'resnet101', 'cropSingle')
            print('======>>>>  after')
            for f in os.listdir(tmpdirname):
                self.assertIn(f, expected_filenames)
                print(f)
    def test_keep_best_two_only_two_file(self):
        # get a temporary directory
        filenames = ['best_fold=0_resnet101_cropSingle_model_10_accuracy=0.4146.pt',
                     'best_fold=0_resnet101_cropSingle_model_10_accuracy=0.9999.pt']

        expected_filenames = ['best_fold=0_resnet101_cropSingle_model_10_accuracy=0.4146.pt',
                              'best_fold=0_resnet101_cropSingle_model_10_accuracy=0.9999.pt']
        with tempfile.TemporaryDirectory() as tmpdirname:
            # create a empty file
            for filename in filenames:
                with open(os.path.join(tmpdirname, filename), 'w') as f:
                    pass
            print('======>>>>  before')
            for f in os.listdir(tmpdirname):
                print(f)
            # keep only the best two
            keep_best_two(tmpdirname, 0, 'resnet101', 'cropSingle')
            print('======>>>>  after')
            for f in os.listdir(tmpdirname):
                self.assertIn(f, expected_filenames)
                print(f)

    def test_keep_best_two_only_one_file(self):
        # get a temporary directory
        filenames = ['best_fold=0_resnet101_cropSingle_model_10_accuracy=0.9999.pt']

        expected_filenames = ['best_fold=0_resnet101_cropSingle_model_10_accuracy=0.9999.pt']
        with tempfile.TemporaryDirectory() as tmpdirname:
            # create a empty file
            for filename in filenames:
                with open(os.path.join(tmpdirname, filename), 'w') as f:
                    pass
            print('======>>>>  before')
            for f in os.listdir(tmpdirname):
                print(f)
            # keep only the best two
            keep_best_two(tmpdirname, 0, 'resnet101', 'cropSingle')
            print('======>>>>  after')
            for f in os.listdir(tmpdirname):
                self.assertIn(f, expected_filenames)
                print(f)
