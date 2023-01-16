import torch


class EuclideanLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
    def forward(self, x, y):
        x = torch.softmax(x, dim=1)
        y = y.float()
        return self.mse(x, y)


class CosineLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cosine = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    def forward(self, x, y):
        x = torch.softmax(x, dim=1)
        return torch.mean(1 - self.cosine(x, y))


class MulticlassLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, y):
        #x = torch.softmax(x, dim=1)
        x = self.sigmoid(x)
        x = torch.clamp(x, min=1e-6)
        # return torch.mean(torch.sum(-torch.ceil(y) * torch.log(x) - torch.floor(1 - y) * torch.log(1 - x), dim=1))
        # return torch.mean(torch.sum(-torch.ceil(y) * torch.log(x) - torch.floor(1 - y) * torch.log(1 - x), dim=1))
        # print('torch.ceil(y)', torch.ceil(y) )
        # with torch.no_grad():
        #     print('x(prob)', x[0])
        # return torch.sum(torch.sum(-torch.ceil(y) * torch.log(x), dim=1))
        return torch.mean(torch.sum(-torch.ceil(y) * torch.log(x) - torch.floor(1 - y) * torch.log(1 - x), dim=1))
