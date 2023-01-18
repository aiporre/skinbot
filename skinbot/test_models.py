import unittest
import torch
from skinbot.config import Config, read_config
from .models import get_model
class Models(unittest.TestCase):
    def setUp(self) -> None:
        self.config = read_config()
        self.C = Config()
        self.C.set_config(self.config)
    def test_make_models(self):
        model_names_all = ['resnet101']
        for mn in model_names_all:
            get_model(mn)

    def test_make_models_with_optimizers(self):
        model_names_all = ['resnet101']
        for mn in model_names_all:
            get_model(mn, optimizer='SGD')


    def test_u_net(self):
        unet = get_model('unet')
        # even size h and w image
        # image = torch.rand((1,3,100,100))
        # pred = unet(image)
        # print('prediction shape: ', pred.shape)
        # image = torch.rand((1, 3, 225, 225))
        # pred = unet(image)
        # print('prediction shape: ', pred.shape)
        image = torch.rand((1, 3, 579, 521))
        pred = unet(image)
        print('prediction shape: ', pred.shape)

    def test_u_net_interpolation(self):
        unet = get_model('unet', learnable_upsample=False)
        # even size h and w image
        # image = torch.rand((1,3,100,100))
        # pred = unet(image)
        # print('prediction shape: ', pred.shape)
        # image = torch.rand((1, 3, 225, 225))
        # pred = unet(image)
        # print('prediction shape: ', pred.shape)
        image = torch.rand((1, 3, 579, 521))
        pred = unet(image)
        print('prediction shape: ', pred.shape)

    def test_u_net(self):
        unet = get_model('unet')
        image = torch.rand((1, 3, 579, 521))
        H = 579
        W = 521
        labels = torch.randint(0,13, (1, H, W))
        pred = unet(image)
        print('prediction shape: ', pred.shape)
        criterion =  torch.nn.CrossEntropyLoss()
        loss = criterion(pred, labels)
        print(loss)





if __name__ == '__main__':
    unittest.main()
