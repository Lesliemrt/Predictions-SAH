import torch
from torch import nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, in_features, prob):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=prob)
        self.output = nn.Linear(in_features, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.gap(x)
        # x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.output(x)
        # x = self.sigmoid(x)

        return x

def get_model(prob=0.5):    
    model = models.densenet169(pretrained=True) # pretrained on ImageNet

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # # Unfreezing the last parts of the model
    # for name, param in model.features.denseblock4.named_parameters():
    #     if "denselayer30" in name or "denselayer31" in name or "denselayer32" in name:
    #         param.requires_grad = True

    in_features = model.classifier.in_features
    model.classifier = Model(in_features, prob)

    return model

