import numpy as np
import matplotlib.pyplot as plt
import torch
import pandas as pd
import csv
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score, recall_score, f1_score
from albumentations import Compose, Resize, CenterCrop

from train import Model_extented
import dataloader
import utils
import train
import configs
from model import get_model, Classifier, Classifier_Many_Layers


if __name__ == "__main__":
    outputs = ['Infarction', 'Rebleeding', 'Epileptic seizure', 'VasoespasmA', 'Exitus', 'ANY Vasoespasm ', 'Hydrocephalus']
    metrics_mean = pd.DataFrame(columns=["accuracy", "recall", "f1_score"])
    metrics_max = pd.DataFrame(columns=["accuracy", "recall", "f1_score"])
    for configs.target_output in outputs :
        print(configs.target_output)

        lw = 2 #line width

        #Load data
        df = dataloader.load_data(code_data=2, idx_patient_path=configs.patient, target_output=configs.target_output)
        testloader = dataloader.create_dataloader_predict(df, target_output=configs.target_output)

        # Select the 5 best model base on val auc_roc_score
        weights = pd.read_csv(f"{configs.DIR}checkpoints/auc_roc_val_scores_{configs.target_output}.csv", sep=',')
        top_5_models = weights.sort_values(by='auc_roc_val', ascending=False).head(5)
        print("Top 5 models:", top_5_models)

        #Predict
        all_predictions = []
        labels_ref = None
        for k in range(len(top_5_models)):
            seed = top_5_models.iloc[k]['seed']
            auc_roc_val = top_5_models.iloc[k]['auc_roc_val']
            # print("LISTTTTTTTTTTE3")
            # print(list(torch.load('checkpoints/model_seed_66086_auc_0.7740_output_Exitus.pt').keys())[:30])
            model = get_model(prob=0.5, image_backbone="se_resnext50_32x4d", pretrained = "medical", classifier=Classifier_Many_Layers, metadata=True, attention =True) #prob = prob for dropout
            path = f"checkpoints/model_seed_{seed:.0f}_auc_{auc_roc_val:.4f}_output_{configs.target_output}.pt"
            state_dict = torch.load(path, map_location=configs.device)
            model.load_state_dict(state_dict)
            my_model=Model_extented(model, epochs=5, lr=1)
            labels, probs = my_model.return_outputs(testloader)

            if labels_ref is None:
                labels_ref = labels
            else:
                assert np.array_equal(labels_ref, labels)

            all_predictions.append(probs)
                
        # Take the mean of predictions -----------------------------------------------
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
        plt.title(f'Receiver Operating Characteristic Curve \n best 5-models average for {configs.target_output} ')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{configs.DIR}results_2nd_test_cohort/auc roc 5-model mean for {configs.target_output}.png") 
        plt.close()
        print(f"AUC ROC (5-model ensemble average) for {configs.target_output} = {auc_roc:.4f}")

        # Accuracy, recall
        predicted_labels = np.where(mean_predictions > 0.3, 1, 0)
        accuracy = accuracy_score(labels_ref, predicted_labels)
        recall = recall_score(labels_ref, predicted_labels)
        f1score = f1_score(labels_ref, predicted_labels)

        metrics_mean.loc[configs.target_output] = [accuracy, recall, f1score]

        print(f"Accuracy : {accuracy}, Recall : {recall}, F1-score : {f1score}")
        print("Metrics for mean")
        print(metrics_mean)

        # Take the max of predictions ------------------------------------------------------------
        max_predictions = np.max(np.stack(all_predictions), axis=0)

        # Final auc roc score
        fpr, tpr, _ = roc_curve(labels_ref, max_predictions) #false positiv rate and true positiv rate
        auc_roc = auc(fpr, tpr) 
        plt.figure()
        plt.plot(fpr, tpr, lw=lw, label=f'(AUC = {auc_roc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic Curve \n best 5-models max for {configs.target_output} ')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{configs.DIR}results_2nd_test_cohort/auc roc 5-model max for {configs.target_output}.png") 
        plt.close()
        print(f"AUC ROC (5-model ensemble average) for {configs.target_output} = {auc_roc:.4f}")

        # Accuracy, recall
        predicted_labels = np.where(max_predictions > 0.3, 1, 0)
        accuracy = accuracy_score(labels_ref, predicted_labels)
        recall = recall_score(labels_ref, predicted_labels)
        f1score = f1_score(labels_ref, predicted_labels)

        metrics_max.loc[configs.target_output] = [accuracy, recall, f1score]

        print(f"Accuracy : {accuracy}, Recall : {recall}, F1-score : {f1score}")
        print("Metrics for max")
        print(metrics_max)
        
