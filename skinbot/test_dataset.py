from unittest import TestCase

from skinbot.dataset import read_labels_xls


class Test(TestCase):
    def test_read_labels_xls_as_dict(self):
        root = '../data'
        labels = read_labels_xls(root, concat=False)
        for k, v in labels.items():
            print('key: ', k)
            print('value: ', v.head())

    def test_read_labels_xls(self):
        root = '../data'
        labels = read_labels_xls(root, concat=True)
        description = labels.describe()
        # print all the columns description A
        for d in description.columns:
            print(d)
            print(description[d])
