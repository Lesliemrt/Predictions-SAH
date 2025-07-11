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
from model import get_model, Classifier, Classifier_Many_Layers

print("torch.cuda.is_available()", torch.cuda.is_available()) 

# For reproductibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load data
df = dataloader.load_data(target_output=configs.target_output)
train_patients, valid_patients, test_patients = dataloader.split_data(df = df, random_seed=SEED)
trainloader, validloader, testloader = dataloader.create_dataloader(df, train_patients, valid_patients, test_patients, target_output=configs.target_output)

# Load model
# prob = prob for dropout
# model = densenet169 or densenet121 or se_resnext50_32x4d (pretrained on medical for weights from 3rd contest)
# pretrained = "imagenet"" for pretraining on ImageNet / "medical" for pretraining on Medical Images / False for no training
# classifier = model.Classifier or model.Classifier_Many_Layers
model = get_model(prob=0.5, image_backbone="se_resnext50_32x4d", pretrained = "medical", classifier=Classifier_Many_Layers, num_classes = configs.num_classes, metadata = True)
# model = get_model_onnx(classifier_class=Classifier, in_features=2664, prob=0.5)
my_model=Model_extented(model, epochs=5, lr=1e-3)

# Training
my_model.trainloop(trainloader, validloader)

# à bouger de place
# Save weights
# weights_path = DATA_DIR + 'dense_1_weights.pth'
# torch.save(model.state_dict(), weights_path)

# Training loss and validation loss
plt.plot(my_model.loss_during_training,label='Training Loss')
plt.plot(my_model.valid_loss_during_training,label='Validation Loss')
plt.title(f"Training and validation loss for {configs.target_output}")
plt.legend()
# plt.show()
plt.savefig(f"{configs.DIR}/results/loss.png") 
plt.close()

# Training and validation accuracy
plt.plot(my_model.accuracy_during_training,label='Training Accuracy')
plt.plot(my_model.valid_accuracy_during_training,label='Validation Accuracy')
plt.title(f"Training and validation accuracy for {configs.target_output}")
plt.legend()
# plt.show()
plt.savefig(f"{configs.DIR}/results/accuracy.png") 
plt.close()



# eval_performance_train = my_model.eval_training_performance(trainloader)
# eval_performance_valid = my_model.eval_training_performance(validloader)
# print(f"accuracy/len(trainloader) : {eval_performance_train[0]}, recall : {eval_performance_train[1]}")
# print(f"accuracy/len(validloader) : {eval_performance_valid[0]}, recall : {eval_performance_valid[1]}")

# Saliency maps
print(my_model.saliency(testloader, num_images_to_show=10))

# Results testloader (to save time)
all_labels, all_probs = my_model.return_outputs(testloader)       

# Save the results
if configs.num_classes == 2:
    all_preds = (all_probs >= 0.3).astype(int)
    confidence = all_probs
else : 
    all_preds = np.argmax(all_probs, axis=1)
    confidence = np.max(all_probs, axis=1)
df = pd.DataFrame({
    "True label": all_labels,
    "Predicted label": all_preds,
    "Confidence": confidence
})
if configs.num_classes > 2:
    class_scores = pd.DataFrame(all_probs, columns=[f"class_{i}" for i in range(all_probs.shape[1])])
    df = pd.concat([df, class_scores], axis=1)
df.to_excel(f'{configs.DIR}results/results.xlsx', index=False)

# Auc roc curve
print("Auc-roc score : ",my_model.eval_performance(all_labels, all_probs))

# Calibration eval
print(my_model.calibration_plot(all_labels, all_probs))

print(f"Output : {configs.target_output}")




# Save the missing data in a file for later
# df = pd.DataFrame(utils.non_existant_file, columns=['Missing patient', "-", "ST", "SE", "IM"])
# df.to_excel('non_existant_file.xlsx', index=False)


