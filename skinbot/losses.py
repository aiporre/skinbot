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


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # nn.CrossEntropyLoss()
        BCE_loss = torch.nn.CrossEntropyLoss(reduce=False)(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
