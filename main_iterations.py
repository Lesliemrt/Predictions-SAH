import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd


from train import Model_extented
import dataloader
import utils
import train
import configs

all_fpr = []
all_tpr = []
auc_values = []
training_accuracy_values = []
validation_accuracy_values = []
lw = 2 #line width
nb_iterations = 10

plt.figure()
for k in range(nb_iterations):

    configs.SEED = np.random.randint(10000, 99999)
    print(f"=========== ITERATION {k}/{nb_iterations-1} ------ SEED = {configs.SEED}")

    # For reproductibility
    np.random.seed(configs.SEED)
    torch.manual_seed(configs.SEED)
    torch.cuda.manual_seed_all(configs.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    trainloader, validloader, testloader = dataloader.create_dataloader()

    # Load model
    from model import get_model, Classifier, Classifier_Many_Layers
    model = get_model(prob=0.5, image_backbone="densenet169", pretrained = False, classifier=Classifier) #prob = prob for dropout
    my_model=Model_extented(model, epochs=8, lr=1e-3)

    # Training
    my_model.trainloop(trainloader, validloader, testloader)

    # # Training loss and validation loss
    # plt.plot(my_model.loss_during_training,label='Training Loss')
    # plt.plot(my_model.valid_loss_during_training,label='Validation Loss')
    # plt.legend()

    # Add the values of each iteration to lists :
    training_accuracy_values.append(my_model.eval_training_performance(trainloader)[0])
    validation_accuracy_values.append(my_model.eval_training_performance(validloader)[0])

    results_auc_roc = my_model.auc_roc_iteration(testloader)
    auc_values.append(results_auc_roc[0])
    fpr = results_auc_roc[1]
    tpr = results_auc_roc[2]

    plt.plot(fpr, tpr, lw=lw, label=f'Seed {configs.SEED} (AUC = {results_auc_roc[0]:.2f})')

avg_roc_auc = np.mean(auc_values)
print("Average ROC AUC:", avg_roc_auc)

avg_train_accuracy = np.mean(training_accuracy_values)
avg_valid_accuracy = np.mean(validation_accuracy_values)
print(f"Average training accuracy : {avg_train_accuracy}, Average validation accuracy : {avg_valid_accuracy}")

# Plot the curve
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic Curve \n Average ROC AUC: {avg_roc_auc}')
if nb_iterations<10 :
    plt.legend(loc="lower right")
plt.tight_layout()
# plt.show()
plt.savefig(f"{configs.DATA_DIR}/results/auc roc iterations.png") 
plt.close()


