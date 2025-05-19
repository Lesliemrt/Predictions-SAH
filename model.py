import torch
from torch import nn
from torchvision import models
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
    
# class Backbone(nn.Module):
#     def __init__(self):
#         super().__init__()
#         base_model = models.resnet50(pretrained=False)
#         encoder_layers = list(base_model.children())
#         self.backbone = nn.Sequential(*encoder_layers[:9])
    
#     def forward(self, x):
#         return self.backbone(x)

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
    model.classifier = Classifier(in_features, prob)

    return model

def get_model_densenet121(prob=0.5):    
    device = configs.device

    model = models.densenet121(pretrained=False)

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

