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

print("accuracy/len(trainloader), recall : ", my_model.eval_performance(trainloader))
print("accuracy/len(validloader), recall : ", my_model.eval_performance(validloader))



# Saliency maps
print(my_model.saliency(testloader, index=0))

print(my_model.visualize_predictions(train_df, 10))

print("auc_roc : ",my_model.auc_roc(testloader))

# See the results
df = pd.DataFrame(train.results, columns=['True', "Predict"])
df.to_excel('results.xlsx', index=False)

# Save the missing data in a file for later
df = pd.DataFrame(utils.non_existant_file, columns=['Missing patient', "-", "ST", "SE", "IM"])
df.to_excel('non_existant_file.xlsx', index=False)






# test_df_f = test_df.copy()

# # reshaped_predictions = np.vstack(my_model.submission_predictions)
# # averaged_preds = np.average(reshaped_predictions, axis=0, weights=[2**i for i in range(len(submission_predictions))])
# # averaged_preds = np.average(reshaped_predictions, axis=0)
# # averaged_preds = averaged_preds.reshape(test_df.shape[0], test_df.shape[1])
# preds = my_model.submission_predictions
# print("submission_predictions : ",preds)
# test_df_f.iloc[:, :] = preds

# test_df_f = test_df_f.stack().reset_index()
# test_df_f.insert(loc = 0, column = 'ID', value = test_df_f['Image'].astype(str) + "_" + test_df_f['Diagnosis'])
# test_df_f = test_df_f.drop(["Image", "Diagnosis"], axis=1)
# # test_df_f.to_csv('densenet169_submission_new.csv', index = False)
# print(test_df_f)
# print("Label Min : ", test_df_f["Label"].min())
# print("Label Max : ",test_df_f["Label"].max())