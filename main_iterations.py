import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import csv
from sklearn.metrics import roc_auc_score, roc_curve, auc

from train import Model_extented
import dataloader
import utils
import train
import configs
from model import get_model, Classifier, Classifier_Many_Layers

all_fpr = []
all_tpr = []
auc_values = []
training_accuracy_values = []
validation_accuracy_values = []
lw = 2 #line width
nb_iterations = 10
results = []

df = dataloader.load_data(target_output=configs.target_output)

plt.figure()
for k in range(nb_iterations):

    seed = np.random.randint(10000, 99999)
    print(f"=========== ITERATION {k}/{nb_iterations-1} ------ SEED = {seed}")

    # For reproductibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    train_patients, valid_patients, test_patients = dataloader.split_data(df = df, random_seed=seed)
    trainloader, validloader, _ = dataloader.create_dataloader(df, train_patients, valid_patients, test_patients, target_output=configs.target_output)

    # Load model
    model = get_model(prob=0.5, image_backbone="se_resnext50_32x4d", pretrained = "medical", classifier=Classifier_Many_Layers, metadata=True) #prob = prob for dropout
    my_model=Model_extented(model, epochs=5, lr=1e-3)

    # Training
    my_model.trainloop(trainloader, validloader)

    # # Training loss and validation loss
    # plt.plot(my_model.loss_during_training,label='Training Loss')
    # plt.plot(my_model.valid_loss_during_training,label='Validation Loss')
    # plt.legend()

    # Add the values of each iteration to lists :
    training_accuracy_values.append(my_model.eval_training_performance(trainloader)[0])
    validation_accuracy_values.append(my_model.eval_training_performance(validloader)[0])

    results_auc_roc = my_model.auc_roc(validloader)

    # Save scores and weights
    results.append({'seed': seed, 'auc_roc_val': results_auc_roc[0]})

    path = f"{configs.DIR}checkpoints/model_seed_{seed}_auc_{results_auc_roc[0]:.4f}.pt"
    torch.save(model.state_dict(), path)

    # To print curves
    auc_values.append(results_auc_roc[0])
    fpr = results_auc_roc[1]
    tpr = results_auc_roc[2]
    plt.plot(fpr, tpr, lw=lw, label=f'Seed {seed} (AUC = {results_auc_roc[0]:.2f})')

avg_roc_auc = np.mean(auc_values)
print("Average ROC AUC:", avg_roc_auc)
max_roc_auc = np.max(auc_values)
print("Max ROC AUC:", max_roc_auc)
avg_train_accuracy = np.mean(training_accuracy_values)
avg_valid_accuracy = np.mean(validation_accuracy_values)
print(f"Average training accuracy : {avg_train_accuracy}, Average validation accuracy : {avg_valid_accuracy}")
# Plot the curve
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic Curve \n Avg ROC AUC: {avg_roc_auc:.2f}  Max ROC AUC: {max_roc_auc:.2f} ')
if nb_iterations<11 :
    plt.legend(loc="lower right")
plt.tight_layout()
# plt.show()
plt.savefig(f"{configs.DIR}/results/auc roc iterations.png") 
plt.close()


# Save the list of all models
path = f"{configs.DIR}checkpoints/auc_roc_val_scores.csv"
with open(path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["seed", "auc_roc_val"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Select the 5 best 
top_5_models = sorted(results, key=lambda x: x['auc_roc_val'], reverse=True)[:5]
print("Top 5 models:", top_5_models)

# Split
train_patients, valid_patients, test_patients = dataloader.split_data(df = df, random_seed=configs.SEED)
_, _, testloader = dataloader.create_dataloader(df, train_patients, valid_patients, test_patients, target_output=configs.target_output)

# Predict for each 5
all_predictions = []
labels_ref = None
for item in top_5_models:
    seed = item['seed']
    auc_roc_val = item['auc_roc_val']
    model = get_model(prob=0.5, image_backbone="se_resnext50_32x4d", pretrained = "medical", classifier=Classifier_Many_Layers, metadata=True) #prob = prob for dropout
    path = f"checkpoints/model_seed_{seed}_auc_{auc_roc_val:.4f}.pt"
    state_dict = torch.load(path, map_location=configs.device)
    model.load_state_dict(state_dict)
    my_model=Model_extented(model, epochs=5, lr=1)
    labels, probs = my_model.return_outputs(testloader)

    if labels_ref is None:
        labels_ref = labels
    else:
        assert np.array_equal(labels_ref, labels)

    all_predictions.append(probs)

# Take the mean of predictions
mean_predictions = np.mean(np.stack(all_predictions), axis=0)

# Final auc roc score
fpr, tpr, _ = roc_curve(labels_ref, mean_predictions) #false positiv rate and true positiv rate
auc_roc = auc(fpr, tpr) 
plt.figure()
plt.plot(fpr, tpr, lw=lw, label=f'(AUC = {auc_roc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic Curve \n 5 models concatenation for {configs.target_output} ')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{configs.DIR}/results/auc roc 5-model ensemble.png") 
plt.close()
print(f"AUC ROC (5-model ensemble) for {configs.target_output} = {auc_roc:.4f}")