import torch
import torch.nn as nn
from torchvision.models import densenet169
import numpy as np

folder_path = '/export/usuarios01/lmurat/Datos/Predictions-SAH/keras_to_pytorch/'
# 3. Create pytorch model
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

torch_model = Model_6_classes(prob = 0.15)

# 4. Transfer to PyTorch layer per layer
# Exemple simplifié : suppose que les noms de couches correspondent
keras_weights = np.load(folder_path+'keras_weights.npy', allow_pickle=True).item()

def keras_to_pytorch():
    with torch.no_grad():

        for name, param in torch_model.named_parameters():
            # Extraire le nom "simplifié"
            simplified_name = name.split(".")[0]
            if simplified_name in keras_weights:
                w = keras_weights[simplified_name]
                if 'weight' in name and w[0].ndim >= 2:
                    param.copy_(torch.tensor(np.transpose(w[0], (3, 2, 0, 1))) if w[0].ndim == 4 else torch.tensor(w[0]))
                elif 'bias' in name and len(w) > 1:
                    param.copy_(torch.tensor(w[1]))

# 5. Enregistrer le modèle PyTorch
torch.save(torch_model.state_dict(), folder_path + "converted_densenet169.pth")

#verif
state_dict = torch.load(folder_path + "converted_densenet169.pth")

for key, value in state_dict.items():
    print(f"{key}: {value.shape}")


keras_keys = list(keras_weights.keys())
torch_keys = list(torch_model.state_dict().keys())

for k in range(min(len(keras_keys), len(torch_keys))):
    print("noms keras :", keras_keys[k])
    print("noms torch :", torch_keys[k])
    print("-" * 50)
