import torch
from torch import nn
from torchvision import models

class Model(nn.Module):
    def __init__(self, in_features, prob):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=prob)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(in_features, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 32)
        self.linear7 = nn.Linear(32,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)
        return x

def get_model(prob=0.5):    
    model = models.densenet169(pretrained=True) # pretrained on ImageNet

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    in_features = model.classifier.in_features
    model.classifier = Model(in_features, prob)

    return model