from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import torch

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # self.model = models.vgg16(pretrained=True)
        self.model = models.vgg16(pretrained=True).features
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])
        self.input_size = 224

    def forward(self, x):
        x = self.model(x)
        # x = self.model.features(x)
        return x
