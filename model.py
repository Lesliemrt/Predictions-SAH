import torch
from torch import nn
from torchvision.models import resnet50, densenet121, densenet169
from torch.nn import functional as F
import onnxruntime as rt
from configs import DATA_DIR, DIR
import configs
import utils
from keras_to_pytorch.densenet_from_IR import KitModel

class Classifier(nn.Module):
    def __init__(self, in_features, prob):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=prob)
        self.output = nn.Linear(in_features, 1)

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
    
class MLP(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.linear1 = nn.Linear(8, 8) # 6 = dim of meta data
        self.linear2 = nn.Linear(8, in_features)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class CombineModel(nn.Module):
    def __init__(self, image_backbone, meta_backbone, classifier, metadata):
        super().__init__()
        self.image_backbone = image_backbone
        self.meta_backbone = meta_backbone
        self.classifier = classifier
        self.metadata = metadata
    def forward(self, image, meta):
        image_output = self.image_backbone(image)
        meta_output = self.meta_backbone(meta)
        if meta_output.dim() == 1:
            meta_output = meta_output.unsqueeze(0)
        combined = torch.cat((image_output, meta_output), dim=1)
        if self.metadata == True:
            output = self.classifier(combined)
        else : 
            output = self.classifier(image_output)
        return output

def get_model(prob=0.5, image_backbone="densenet169", pretrained="imagenet", classifier=Classifier, metadata = True):    
    device = configs.device

    if pretrained == False:
        if image_backbone == "densenet169":
            image_backbone = densenet169(pretrained=False)
        if image_backbone == "densenet121":
            image_backbone = densenet121(pretrained = False)

    if pretrained == "imagenet":
        if image_backbone == "densenet169":
            image_backbone = densenet169(pretrained=True)
        if image_backbone == "densenet121":
            image_backbone = densenet121(pretrained = True)

    if pretrained == "medical" : 
        if image_backbone == "densenet121":
            image_backbone = densenet121(pretrained = False)
            state_dict=torch.load(f"{DATA_DIR}RadImageNet_pytorch/densenet121.pt", map_location = device)
            missing_keys, unexpected_keys = image_backbone.load_state_dict(state_dict, strict=False)
                        # 4. Afficher ce qui a été ignoré
            print("⚠️ Clés manquantes dans le state_dict (non chargées) :", missing_keys[:10], len(missing_keys))
            print("⚠️ Clés inattendues (ignorées car pas dans le modèle) :", unexpected_keys[:10], len(unexpected_keys))     
        
        if image_backbone == "densenet169":
            # image_backbone = densenet169(pretrained = False)
            # state_dict = torch.load(f"{DATA_DIR}model_epoch_best_4.pth", map_location=device)['state_dict']
            # state_dict = utils.adapt_name(state_dict)
            # image_backbone.load_state_dict(state_dict, strict=False)
            image_backbone = KitModel("/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/densenet_from_IR_weights.npy")
            image_backbone.classifier = nn.Identity()

    # Freeze parameters so we don't backprop through them
    if pretrained in ["imagenet", "medical"]:
        for param in image_backbone.parameters():
            param.requires_grad = False

    meta_backbone = MLP(1000) # meta_output.shape = 1000 because image_output.shape = 1000 and must be equal (for same weights)
    if metadata == True : 
        classifier = classifier(2000, prob)  # 2000 = image_output.shape + meta_output.shape
    else : 
        classifier = classifier(1664, prob)
    model = CombineModel(image_backbone, meta_backbone, classifier, metadata)
    
    return model



# For model pretrained by jaymin on dataset RSNA2019 contest    
class Densenet169_onnx(nn.Module): 
    def __init__(self):
        super().__init__()
        providers = ['CPUExecutionProvider']
        output_path = DATA_DIR + "densenet169_model.onnx"
        self.m = rt.InferenceSession(output_path, providers=providers)
        self.input_name = self.m.get_inputs()[0].name

    def forward(self, x):
        device = configs.device
        x_numpy = x.permute(0, 2, 3, 1).detach().cpu().numpy()
        onnx_pred = self.m.run(None, {self.input_name: x_numpy})
        features = torch.tensor(onnx_pred[0], dtype=torch.float32, device=device)  # force device
        return features


class CombineModel_onnx(nn.Module):
    def __init__(self, image_backbone, meta_backbone, classifier):
        super().__init__()
        self.image_backbone = image_backbone
        self.meta_backbone = meta_backbone
        self.classifier = classifier
    def forward(self, image, meta):
        image_output = self.image_backbone(image) # not trainable
        meta_output = self.meta_backbone(meta)
        if meta_output.dim() == 1:
            meta_output = meta_output.unsqueeze(0)
        combined = torch.cat((image_output, meta_output), dim=1)
        output = self.classifier(combined)
        return output


def get_model_onnx(classifier_class=Classifier, in_features=2000, prob=0.5):
    image_backbone = Densenet169_onnx()
    meta_backbone = MLP(1000)
    classifier = classifier_class(in_features, prob)
    model = CombineModel_onnx(image_backbone=image_backbone, meta_backbone = meta_backbone, classifier=classifier)
    return model

class Model_6classes_onnx(nn.Module):
    def __init__(self, prob, in_features):
        super().__init__()
        self.base_model = Densenet169_onnx()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=prob)
        self.linear = nn.Linear(in_features, 6)
    def forward(self, x):
        x = self.base_model(x)
        print("Shape base model:", x.shape)
        x = self.linear(x)
        return x
    
class DenseNet169_change_avg(nn.Module):
    def __init__(self):
        super(DenseNet169_change_avg, self).__init__()
        self.densenet169 = densenet169(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d(1)  
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(1664, 6)
        self.sigmoid = nn.Sigmoid()   

    def forward(self, x):
        x = self.densenet169(x)      
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.mlp(x)

        return x

class Model_6_classes(nn.Module):
    def __init__(self,prob):
        super().__init__()
        self.base_model = densenet169(pretrained=False)
        self.base_model.classifier = nn.Identity()
        self.linear = nn.Linear(1664, 6)
        self.dropout = nn.Dropout(p=prob)
    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x
    


# Other mdoel (from from https://github.com/okotaku/kaggle_rsna2019_3rd_solution)
from pretrainedmodels import se_resnext50_32x4d
encoders = {
    "se_resnext50_32x4d": {
        "encoder": se_resnext50_32x4d,
        "out_shape": 2048
    },
}

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class SEBlock(nn.Module):
    def __init__(self, in_ch, r=8):
        super(SEBlock, self).__init__()

        self.linear_1 = nn.Linear(in_ch, in_ch//r)
        self.linear_2 = nn.Linear(in_ch//r, in_ch)

    def forward(self, x):
        input_x = x

        x = F.relu(self.linear_1(x), inplace=True)
        x = self.linear_2(x)
        x = F.sigmoid(x)

        x = input_x * x

        return x

class CnnModel(nn.Module):
    def __init__(self, num_classes, encoder="se_resnext50_32x4d", pretrained="imagenet"):
        super().__init__()
        self.net = encoders[encoder]["encoder"](pretrained=pretrained)

        self.net.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        out_shape = encoders[encoder]["out_shape"]

        self.net.last_linear = nn.Sequential(
            Flatten(),
            SEBlock(out_shape),
            nn.Dropout(),
            nn.Linear(out_shape, num_classes)
        )


    def fresh_params(self):
        return self.net.last_linear.parameters()

    def forward(self, x):
        return self.net(x)