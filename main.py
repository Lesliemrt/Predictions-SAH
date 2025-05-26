import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from configs import SEED
from train import Model_extented
import dataloader
import utils
import train
import configs

# For reproductibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data
trainloader, validloader, testloader = dataloader.create_dataloader()

# Load model
from model import get_model, Classifier, Classifier_Many_Layers
# prob = prob for dropout
# model = densenet169 or densenet121
# pretrained = True for pretraining on ImageNet or False for pretraining on Medical Images
# classifier = model.Classifier or model.Classifier_Many_Layers
model = get_model(prob=0.5, model="densenet169", pretrained = False, classifier=Classifier_Many_Layers)
my_model=Model_extented(model, epochs=8, lr=1e-3)

# Training
my_model.trainloop(trainloader, validloader, testloader)

# à bouger de place
# Save weights
# weights_path = DATA_DIR + 'dense_1_weights.pth'
# torch.save(model.state_dict(), weights_path)

# Training loss and validation loss
plt.plot(my_model.loss_during_training,label='Training Loss')
plt.plot(my_model.valid_loss_during_training,label='Validation Loss')
plt.legend()
# plt.show()
plt.savefig(f"{configs.DATA_DIR}/results/loss.png") 
plt.close()

# Training and validation accuracy
plt.plot(my_model.accuracy_during_training,label='Training Accuracy')
plt.plot(my_model.valid_accuracy_during_training,label='Validation Accuracy')
plt.legend()
# plt.show()
plt.savefig(f"{configs.DATA_DIR}/results/accuracy.png") 
plt.close()



# eval_performance_train = my_model.eval_training_performance(trainloader)
# eval_performance_valid = my_model.eval_training_performance(validloader)
# print(f"accuracy/len(trainloader) : {eval_performance_train[0]}, recall : {eval_performance_train[1]}")
# print(f"accuracy/len(validloader) : {eval_performance_valid[0]}, recall : {eval_performance_valid[1]}")

# Saliency maps
print(my_model.saliency(testloader, index=1))

# Grad cam
print(my_model.gradcam(testloader, index = 0))

# print(my_model.visualize_predictions(train_df, 10))

# Results testloader (to save time)
all_labels, all_probs = my_model.return_outputs(testloader)       

# Save the results
all_preds = (all_probs >= 0.5).astype(int)
df = pd.DataFrame({
    "True label": all_labels,
    "Predicted label": all_preds,
    "Confidence": all_probs
})
df.to_excel('results.xlsx', index=False)

# Auc roc curve
print("Auc-roc score : ",my_model.eval_performance(all_labels, all_probs))

# Calibration eval
print(my_model.calibration_plot(all_labels, all_probs))




# Save the missing data in a file for later
# df = pd.DataFrame(utils.non_existant_file, columns=['Missing patient', "-", "ST", "SE", "IM"])
# df.to_excel('non_existant_file.xlsx', index=False)


