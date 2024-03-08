import torch
from torch import nn as nn
from torch.nn import functional as F
from torchvision.models import densenet121
from basicsr.utils.registry import LOSS_REGISTRY
from torchkeras import summary


class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        densenet121_model = densenet121(pretrained=True)
        self.densenet121_model = nn.Sequential(*list(densenet121_model.features.children())[:6]) # after 18th dense block, before 2th Transition layer

    def forward(self, img):
        r = self.densenet121_model(img)
        return r



@LOSS_REGISTRY.register()
class DenseNet121Loss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(DenseNet121Loss, self).__init__()
        self.loss_weight = loss_weight
        self.FeatureExtractor = FeatureExtractor()
        self.Loss = torch.nn.L1Loss()

    def forward(self, pred, target, **kwargs):
        loss = self.Loss(self.FeatureExtractor(pred), self.FeatureExtractor(target))
        return self.loss_weight * loss


