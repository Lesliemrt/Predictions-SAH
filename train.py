import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import torch
from torch import optim
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from torchmetrics.classification import BinaryRecall
from torch.optim.lr_scheduler import StepLR
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from dataloader import create_dataloader
import configs
import utils
from reliability_diagrams import *

# trainloader, validloader, testloader = create_dataloader()

class Model_extented(nn.Module):
    def __init__(self,model, epochs,lr):
        super().__init__()
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.optim = optim.Adam(self.model.classifier.parameters(), self.lr)
        self.loss_function = nn.BCEWithLogitsLoss()
        self.loss_during_training = []
        self.valid_loss_during_training = []
        self.accuracy_during_training = []
        self.valid_accuracy_during_training = []
        self.submission_predictions=[]
        self.device = configs.device
        self.model.to(self.device)

    def forward(self, image, meta):
        return self.model(image, meta)

    def trainloop(self, trainloader, validloader, testloader):
        self.model.train()
        scheduler = StepLR(self.optim, step_size=2, gamma=0.5) # Every 2 epochs, split lr by 2
        for epoch in range (self.epochs):
            print(f'=========== EPOCH {epoch}/{self.epochs-1}')
            start_time = time.time()
            running_loss = 0.0
            train_accuracy = 0.0
            for batch in trainloader:
                inputs = batch['image']
                meta = batch['meta']
                labels = batch['label']
                inputs, meta, labels = inputs.float().to(self.device), meta.float().to(self.device), labels.float().to(self.device) #model expect float32
                self.optim.zero_grad()
                outputs = self.forward(image=inputs, meta=meta)
                labels = labels.unsqueeze(1)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                running_loss += loss.item()
                self.optim.step()

                # Accuracy
                probs = torch.sigmoid(outputs)
                predicted_labels = (probs > 0.3).float() # Get predicted labels based on threshold
                equals = (predicted_labels == labels) # Compare predicted and actual labels
                train_accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Calculate accuracy

            self.loss_during_training.append(running_loss/len(trainloader))
            self.accuracy_during_training.append(train_accuracy/len(trainloader))

            # Validation mode
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                val_accuracy = 0.0
                for batch in validloader:
                    inputs = batch['image']
                    meta = batch['meta']
                    labels = batch['label']
                    inputs, meta, labels = inputs.float().to(self.device), meta.float().to(self.device), labels.float().to(self.device)
                    outputs = self.forward(image = inputs, meta=meta)
                    labels = labels.unsqueeze(1)
                    loss = self.loss_function(outputs, labels)
                    val_loss += loss.item()

                    # Accuracy
                    probs = torch.sigmoid(outputs)
                    predicted_labels = (probs > 0.3).float() # Get predicted labels based on threshold
                    equals = (predicted_labels == labels) # Compare predicted and actual labels
                    val_accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Calculate accuracy

                self.valid_loss_during_training.append(val_loss/len(validloader))
                self.valid_accuracy_during_training.append(val_accuracy/len(validloader))

            # scheduler.step()  # decrease lr
            current_lr = self.optim.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

            self.model.train()
            print("Training loss: %f, Validation loss: %f, Time per epoch: %f seconds"
                       %(self.loss_during_training[-1],self.valid_loss_during_training[-1],
                       (time.time() - start_time)))
        
    # eval performance on trainloader and validloader
    def eval_training_performance(self,dataloader):
        accuracy = 0
        recall_metric = BinaryRecall(threshold=0.5).to(self.device)
        self.model.eval()
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['image']
                meta = batch['meta']
                labels = batch['label']
                inputs, meta, labels = inputs.float().to(self.device), meta.float().to(self.device), labels.float().to(self.device)
                outputs = self.forward(image=inputs, meta=meta)
                probs = torch.sigmoid(outputs)
                # print("Max prob in batch:", probs.max().item())

                predicted_labels = (probs > 0.5).float() # Get predicted labels based on threshold
                labels = labels.unsqueeze(1)
                equals = (predicted_labels == labels) # Compare predicted and actual labels
                # accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Calculate accuracy
                accuracy += torch.mean(equals.float()).item()


                recall_metric.update(probs.view(-1), labels.view(-1))
            
        recall = recall_metric.compute()

        self.model.train()
        return accuracy/len(dataloader), recall
    
    # to eval performance on testloader and only run the dataloader loop once
    def return_outputs(self, dataloader):
        self.model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['image']
                meta = batch['meta']
                labels = batch['label']
                inputs, meta, labels = inputs.float().to(self.device), meta.float().to(self.device), labels.float().to(self.device)
                outputs = self.forward(image=inputs, meta=meta)
                probs = torch.sigmoid(outputs)

                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        # Concatenate all batchs
        all_labels = torch.cat(all_labels).flatten().numpy()
        all_probs = torch.cat(all_probs).flatten().numpy()
        self.model.train()
        return all_labels, all_probs        

    def visualize_predictions(self, df, num_images_to_show):
        self.model.eval()
        with torch.no_grad():
            for i in range(num_images_to_show):

                random_index = np.random.randint(0, len(df))

                img_path = df['Path'].iloc[random_index]
                label = df['ANY_Vasospasm'].iloc[random_index]

                img = utils._read(img_path)

                img = img.float().unsqueeze(0).to(self.device)
                output = self.forward(img)
                probs = torch.sigmoid(output)

                plt.figure(figsize=(10, 5))
                img = img.squeeze().permute(1, 2, 0).cpu().numpy()
                plt.imshow(img, cmap='gray' if img.shape[2] == 1 else None)
                plt.title(f"Réel: {label}, Prédiction: {probs.item():.4f}")
                plt.axis("off")
                # plt.show()
                plt.savefig(f"{configs.DATA_DIR}/results/visualize predictions.png") 
                plt.close()

        self.model.train()


    def saliency(self, dataloader, num_images_to_show):
        self.model.eval()
        for i in range(num_images_to_show):
            for batch in dataloader:
                inputs = batch['image']
                meta = batch['meta']
                labels = batch['label']
                batch_size = inputs.size(0)
                index = np.random.randint(0, batch_size-1)
                input_img = inputs[index].unsqueeze(0).float().to(self.device)
                input_img.requires_grad = True
                meta = meta[index].float().to(self.device)

                output = self.forward(input_img, meta)
                probs = torch.sigmoid(output)

                # Get the class with the highest predicted score
                score, _ = torch.max(probs, dim=1)
                score.backward()
                # Get the saliency map: max of gradients across channels
                saliency_map, _ = torch.max(torch.abs(input_img.grad[0]), dim=0)
                saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())

                # Visualization
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                img_np = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy()
                plt.imshow(img_np, cmap='gray' if img_np.shape[2] == 1 else None)
                plt.title(f"Original Image (label = {labels[index]})")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(img_np, cmap='gray' if img_np.shape[2] == 1 else None)
                plt.imshow(saliency_map.cpu(), cmap='hot', alpha=0.7)
                plt.title("Saliency Map")
                plt.axis("off")
                plt.tight_layout()
                # plt.show()
                plt.savefig(f"{configs.DATA_DIR}/results/saliency{index}.png") 
                plt.close()

        self.model.train()
    
    def gradcam(self, dataloader, index):
        self.model.eval()
        print("debut gradcam ")
        for batch in dataloader:
            inputs = batch['image']
            input_img = inputs[index].unsqueeze(0).float().to(self.device)
            input_img.requires_grad = True
            break #just for one image
        input_tensor =input_img
        img_np = input_img.detach().cpu().squeeze().permute(1, 2, 0).numpy() # to visualize initial image
        # We have to specify the target we want to generate the CAM for.
        targets = [ClassifierOutputTarget(0)]  # Visualise la "classe positive"

        layers_to_compare = [
        self.model.features.denseblock1.denselayer2.conv2,
        self.model.features.denseblock2.denselayer4.conv2,
        self.model.features.denseblock3.denselayer8.conv2,
        self.model.features[-2].denselayer32.conv2,
        ]

        _ , axs = plt.subplots(1, len(layers_to_compare), figsize=(20, 5))
        for i, layer in enumerate(layers_to_compare):
            print(f"i : {i}, layer : {layer}")
            with GradCAM(model=self.model, target_layers=[layer]) as cam:
                grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
                grayscale_cam = grayscale_cam[0, :]
                visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)
                axs[i].imshow(visualization)
                axs[i].set_title(f'Grad-CAM layer {i+1}')
                axs[i].axis('off')

        plt.tight_layout()
        # plt.show()
        plt.savefig(f"{configs.DATA_DIR}/results/grad cam.png") 
        plt.close()

    def eval_performance(self, all_labels, all_probs):
        predicted_labels = (all_probs > 0.5).astype(int)
        print("Classification report at threshold 0.5:")
        print(classification_report(all_labels, predicted_labels))

        roc_auc_score_ = roc_auc_score(all_labels, all_probs)

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs) #false positiv rate and true positiv rate
        roc_auc = auc(fpr, tpr) # same as roc_auc_score but different method

        plt.figure(figsize=(12, 6))

        # Plot the curve
        plt.subplot(1, 2, 1)
        lw = 2
        plt.plot(fpr, tpr, color='magenta',
                lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic Curve (Seed: {configs.SEED})')
        plt.legend(loc="lower right")
        # plt.show()
        plt.savefig(f"{configs.DATA_DIR}/results/auc roc curve.png") 
        plt.close()

        self.model.train()
        return roc_auc_score_
    
    def calibration_plot(self, all_labels, all_probs):
        all_preds = (all_probs >= 0.5).astype(int)

        df = pd.DataFrame({
            "true_label": all_labels,
            "pred_label": all_preds,
            "confidence": all_probs
        })
        
        y_true = df.true_label.values
        y_pred = df.pred_label.values
        y_conf = df.confidence.values   

        fig = reliability_diagram(y_true, y_pred, y_conf, num_bins=10, draw_ece=True,
                          draw_bin_importance="alpha", draw_averages=True,
                          title="Reliability diagram (top) and confidence histogram (bottom)", figsize=(6, 6), dpi=100, 
                          return_fig=True)

    
    # def auc_roc(self, dataloader):
    #     self.model.eval()
    #     all_labels = []
    #     all_probs = []
    #     with torch.no_grad():
    #         for inputs, labels in dataloader:
    #             inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
    #             outputs = self.forward(inputs)
    #             probs = torch.sigmoid(outputs)

    #             all_labels.append(labels.cpu())
    #             all_probs.append(probs.cpu())

    #     # Concatenate all batchs
    #     all_labels = torch.cat(all_labels).numpy()
    #     all_probs = torch.cat(all_probs).numpy()
    #     # results.extend(zip(all_labels, all_probs))

    #     predicted_labels = (all_probs > 0.5).astype(int)
    #     print("Classification report at threshold 0.5:")
    #     print(classification_report(all_labels, predicted_labels))

    #     roc_auc_score_ = roc_auc_score(all_labels, all_probs)

    #     fpr, tpr, thresholds = roc_curve(all_labels, all_probs) #false positiv rate and true positiv rate
    #     roc_auc = auc(fpr, tpr) # same as roc_auc_score but different method

    #     plt.figure(figsize=(12, 6))

    #     # Plot the curve
    #     plt.subplot(1, 2, 1)
    #     lw = 2
    #     plt.plot(fpr, tpr, color='magenta',
    #             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    #     plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    #     plt.xlim([0.0, 1.0])
    #     plt.ylim([0.0, 1.05])
    #     plt.xlabel('False Positive Rate')
    #     plt.ylabel('True Positive Rate')
    #     plt.title(f'Receiver Operating Characteristic Curve (Seed: {configs.SEED})')
    #     plt.legend(loc="lower right")
    #     plt.show()

    #     self.model.train()

    #     return roc_auc_score_
    
    def auc_roc_iteration(self, dataloader):
        self.model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                inputs = batch['image']
                meta = batch['meta']
                labels = batch['label']
                inputs, meta, labels = inputs.float().to(self.device), meta.float().to(self.device), labels.float().to(self.device)
                outputs = self.forward(image=inputs, meta=meta)
                probs = torch.sigmoid(outputs)

                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        # Concatenate all batchs
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        fpr, tpr, thresholds = roc_curve(all_labels, all_probs) #false positiv rate and true positiv rate
        roc_auc = auc(fpr, tpr) # same as roc_auc_score but different method

        self.model.train()

        return roc_auc, fpr, tpr




