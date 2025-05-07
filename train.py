import time
import matplotlib.pyplot as plt
import numpy as np

import torch
from torch import optim
from torch import nn
from sklearn.metrics import roc_auc_score, roc_curve, auc, classification_report
from torchmetrics.classification import BinaryRecall
from torch.optim.lr_scheduler import StepLR

from dataloader import create_dataloader
import configs
import utils

trainloader, validloader, testloader = create_dataloader()
results = []

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
        self.submission_predictions=[]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #to run on gpu if available
        self.model.to(self.device)

    def forward(self, x):
        return self.model(x)

    def trainloop(self, trainloader, validloader, testloader):
        self.model.train()
        scheduler = StepLR(self.optim, step_size=2, gamma=0.5) # Every 2 epochs, split lr by 2
        for epoch in range (self.epochs):
            print(f'=========== EPOCH {epoch}')
            start_time = time.time()
            running_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device) #model expect float32
                self.optim.zero_grad()
                outputs = self.forward(inputs)
                labels = labels.unsqueeze(1)
                loss = self.loss_function(outputs, labels)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                running_loss += loss.item()
                self.optim.step()

            self.loss_during_training.append(running_loss/len(trainloader))

            # Validation mode
            self.model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for inputs, labels in validloader:
                    inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
                    outputs = self.forward(inputs)
                    labels = labels.unsqueeze(1)
                    loss = self.loss_function(outputs, labels)
                    val_loss += loss.item()

                self.valid_loss_during_training.append(val_loss/len(validloader))

            scheduler.step()  # decrease lr
            current_lr = self.optim.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

            self.model.train()
            print("Training loss: %f, Validation loss: %f, Time per epoch: %f seconds"
                       %(self.loss_during_training[-1],self.valid_loss_during_training[-1],
                       (time.time() - start_time)))
        

    def eval_performance(self,dataloader):
        # loss = 0
        accuracy = 0
        recall_metric = BinaryRecall(threshold=0.5).to(self.device)
        self.model.eval()
        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():
            for inputs,labels in dataloader:
                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
                outputs = self.forward(inputs)
                probs = torch.sigmoid(outputs)
                # print("Max prob in batch:", probs.max().item())

                predicted_labels = (probs > 0.3).float() # Get predicted labels based on threshold
                labels = labels.unsqueeze(1)
                equals = (predicted_labels == labels) # Compare predicted and actual labels
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # Calculate accuracy

                recall_metric.update(probs.view(-1), labels.view(-1))
            
        recall = recall_metric.compute()

        self.model.train()
        return accuracy/len(dataloader), recall

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
                plt.show()
        self.model.train()


    def saliency(self, dataloader, index):
        self.model.eval()
        for inputs, _ in dataloader:
            input_img = inputs[index].unsqueeze(0).float().to(self.device)
            input_img.requires_grad = True

            output = self.forward(input_img)
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
            plt.title("Original Image")
            plt.axis("off")
            plt.subplot(1, 2, 2)
            plt.imshow(saliency_map.cpu(), cmap='hot')
            plt.title("Saliency Map")
            plt.axis("off")
            plt.tight_layout()
            plt.show()

            break  # Only one image
        self.model.train()
    
    def auc_roc(self, dataloader):
        self.model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
                outputs = self.forward(inputs)
                probs = torch.sigmoid(outputs)

                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        # Concatenate all batchs
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        predicted_labels = (all_probs > 0.5).astype(int)
        print("Classification report at threshold 0.5:")
        print(classification_report(all_labels, predicted_labels))

        roc_auc_score = roc_auc_score(all_labels, all_probs)

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
        plt.show()

        self.model.train()

        return roc_auc_score
    
    def auc_roc_iteration(self, dataloader):
        self.model.eval()
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs, labels = inputs.float().to(self.device), labels.float().to(self.device)
                outputs = self.forward(inputs)
                probs = torch.sigmoid(outputs)

                all_labels.append(labels.cpu())
                all_probs.append(probs.cpu())

        # Concatenate all batchs
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # roc_auc_score = roc_auc_score(all_labels, all_probs)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs) #false positiv rate and true positiv rate
        roc_auc = auc(fpr, tpr) # same as roc_auc_score but different method

        self.model.train()

        return roc_auc, fpr, tpr




