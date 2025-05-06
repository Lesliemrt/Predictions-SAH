import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd

from configs import SEED
from configs import DATA_DIR
from dataloader import create_dataloader
from dataloader import test_df
from dataloader import train_df
from train import Model_extented
import utils
import train


all_test_predictions = []
all_test_labels = []
auc_values = []

nb_iterations = 40
for k in range(nb_iterations):

    SEED = np.random.randint(10000, 99999)

    # For reproductibility
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    trainloader, validloader, testloader = create_dataloader()

    # Load model
    from model import get_model
    model = get_model(prob=0.5)  #prob = prob for dropout
    my_model=Model_extented(model, epochs=4, lr=1e-3)

    # Training
    my_model.trainloop(trainloader, validloader, testloader)

    # Training loss and validation loss
    plt.plot(my_model.loss_during_training,label='Training Loss')
    plt.plot(my_model.valid_loss_during_training,label='Validation Loss')
    plt.legend()

    print("accuracy/len(trainloader), recall : ", my_model.eval_performance(trainloader))
    print("accuracy/len(validloader), recall : ", my_model.eval_performance(validloader))



    # Saliency maps
    print(my_model.saliency(testloader, index=0))

    print(my_model.visualize_predictions(train_df, 10))

    # print("auc_roc : ",my_model.auc_roc(testloader))
    auc_values.append(my_model.auc_roc(testloader))

    # See the results
    df = pd.DataFrame(train.results, columns=['True', "Predict"])
    df.to_excel('results.xlsx', index=False)

    # Save the missing data in a file for later
    # df = pd.DataFrame(utils.non_existant_file, columns=['Missing patient', "-", "ST", "SE", "IM"])
    # df.to_excel('non_existant_file.xlsx', index=False)

