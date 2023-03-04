from unittest import TestCase
from skinbot.utils import change_models_to_names
from skinbot.config import Config, read_config
from skinbot.utils_models import SmallCNN


class Test(TestCase):
    def setUp(self) -> None:
        self.config = read_config()
        self.C = Config()
        self.C.set_config(self.config)
    def test_smallcnn_create(self):
        smallcnn = SmallCNN(10)
        print(smallcnn)
        print(smallcnn.num_middle)

