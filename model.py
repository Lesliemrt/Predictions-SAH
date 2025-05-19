import torch
from torch import nn
from torchvision.models import resnet50, densenet121, densenet169
from configs import DATA_DIR
import configs

class Classifier(nn.Module):
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
    
class Classifier_Many_Layers(nn.Module):
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

def get_model_(prob=0.5):    
    model = densenet169(pretrained=True) # pretrained on ImageNet

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # # Unfreezing the last parts of the model
    # for name, param in model.features.denseblock4.named_parameters():
    #     if "denselayer30" in name or "denselayer31" in name or "denselayer32" in name:
    #         param.requires_grad = True

    in_features = model.classifier.in_features
    model.classifier = Classifier(in_features, prob)

    return model

def get_model_densenet121(prob=0.5):    
    device = configs.device

    model = densenet121(pretrained=False)

    state_dict=torch.load(f"{DATA_DIR}RadImageNet_pytorch/ResNet50.pt", map_location = device)
    model.load_state_dict(state_dict, strict=False)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    in_features = model.classifier.in_features
    model.classifier = Classifier(in_features, prob)
    
    # # Unfreezing the last parts of the model
    # for name, param in model.features.denseblock4.named_parameters():
    #     if "denselayer30" in name or "denselayer31" in name or "denselayer32" in name:
    #         param.requires_grad = True

    return model

def get_model(prob=0.5, base_model=densenet169, pretrained = True, classifier=Classifier):    
    device = configs.device

    model = base_model(pretrained=pretrained)
    if pretrained == False :
        state_dict=torch.load(f"{DATA_DIR}RadImageNet_pytorch/{base_model.__name__}.pt", map_location = device)
        model.load_state_dict(state_dict, strict=False)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    in_features = model.classifier.in_features
    model.classifier = classifier(in_features, prob)
    
    # # Unfreezing the last parts of the model
    # for name, param in model.features.denseblock4.named_parameters():
    #     if "denselayer30" in name or "denselayer31" in name or "denselayer32" in name:
    #         param.requires_grad = True

    return model

