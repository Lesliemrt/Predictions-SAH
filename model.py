import torch
from torch import nn
from torchvision.models import resnet50, densenet121, densenet169
import onnxruntime as rt
from configs import DATA_DIR
import configs
import utils

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

class Densenet169_onnx(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        providers = ['CPUExecutionProvider']
        output_path = DATA_DIR+"densenet169_model.onnx"
        m = rt.InferenceSession(output_path, providers=providers)
        input_name = m.get_inputs()[0].name

        x_numpy = x.detach().cpu().numpy()
        onnx_pred = self.m.run(None, {input_name: x_numpy})
        features = torch.tensor(onnx_pred[0])
        return features

class CombineBackboneClassifier(nn.Module):
    def __init__(self, backbone, classifier):
        super().__init__()
        self.backbone = backbone
        self.classifier = classifier

    def forward(self, x):
        print("model densenet169 avec poids onnx pretrained on RSNA")
        with torch.no_grad():  # on ne backprop pas sur ONNX
            features = self.backbone(x)
        out = self.classifier(features)
        return out

# densenet169(pretrained = True) : pretrained on ImageNet
# densenet169(pretrained = False) : pretrained on RSNA2019 data set
# densenet121(pretrained = True) : pretrained on ImageNet
# densenet121(pretrained = False) : pretrained on RadImageNet
def get_model(prob=0.5, model=densenet169, pretrained=True, classifier=Classifier):    
    device = configs.device

    if pretrained == True:
        if model == densenet169:
            model = densenet169(pretrained=True)
        if model == densenet121:
            model = densenet121(pretrained = True)
        # Freeze parameters so we don't backprop through them
        for param in model.parameters():
            param.requires_grad = False
        
        in_features = model.classifier.in_features
        model.classifier = classifier(in_features, prob)

    if pretrained == False : 
        if model == densenet121:
            model = densenet121(pretrained = False)
            state_dict=torch.load(f"{DATA_DIR}RadImageNet_pytorch/densenet121.pt", map_location = device)
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
                        # 4. Afficher ce qui a été ignoré
            print("⚠️ Clés manquantes dans le state_dict (non chargées) :", missing_keys[:10], len(missing_keys))
            print("⚠️ Clés inattendues (ignorées car pas dans le modèle) :", unexpected_keys[:10], len(unexpected_keys))

            # Freeze parameters so we don't backprop through them
            for param in model.parameters():
                param.requires_grad = False
            
            in_features = model.classifier.in_features
            model.classifier = classifier(in_features, prob)       
        
        if model == densenet169:
            # model = CombineBackboneClassifier(backbone = Densenet169_onnx, classifier = Classifier)

            model = densenet169(pretrained = False)
            state_dict = torch.load(f"{DATA_DIR}model_epoch_best_4.pth", map_location=device)['state_dict']
            state_dict = utils.adapt_name(state_dict)
            model.load_state_dict(state_dict, strict=False)

            # Freeze parameters so we don't backprop through them
            for param in model.parameters():
                param.requires_grad = False
            
            in_features = model.classifier.in_features
            model.classifier = classifier(in_features, prob)     
    
    return model

