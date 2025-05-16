import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import cv2
# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

import configs
import utils

""" Data loader """
class TrainDataset(Dataset):
    def __init__(self, dataset, labels, batch_size, img_size = (configs.CHANNELS, configs.HEIGHT, configs.WIDTH), augment = False):
        self.dataset = dataset
        self.labels = labels
        self.ids = dataset.index.tolist()
        self.batch_size = batch_size
        self.img_size = img_size
        # self.img_dir = img_dir
        self.augment = augment
        self.transform = self.get_transforms()

    def get_transforms(self):
        base_transforms = []
        if self.augment:
            base_transforms += [
                utils.sometimes(0.50, transforms.RandomResizedCrop(size=(self.img_size[1], self.img_size[2]), scale=(0.8, 1.0))),
                utils.sometimes(0.3, transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))),  # Zoom
                utils.sometimes(0.3, transforms.RandomRotation(degrees=30)),   # Rotation
                utils.sometimes(0.3, transforms.ColorJitter(brightness=0.2)),  # Brightness variation (≈ Multiply)
                utils.sometimes(0.3, transforms.RandomErasing(scale=(0.02, 0.1))),
                utils.sometimes(0.3, transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)))

                # Other augmentations :
                # transforms.RandomHorizontalFlip(p=0.25),
                # transforms.RandomVerticalFlip(p=0.10),
                # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Translation
                # utils.sometimes(0.25, transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))),
                # transforms.RandomCrop(size=(self.img_size[1], self.img_size[2]),  # Crop
                #                     padding=(int(0.1 * self.img_size[1]), int(0.1 * self.img_size[2])),
                #                     pad_if_needed=True,
                #                     padding_mode='reflect')
            ]
        return transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_path = self.dataset['Path'].iloc[index]

        image = utils._read(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found : {image_path}")

        # resize_transform = transforms.Resize((self.img_size[1], self.img_size[2]))
        # image = resize_transform(image)

        # Augmentations for trainloader (if self.augment)
        image = self.transform(image)

        label = torch.tensor(self.labels.iloc[index], dtype=torch.float32)

        return image, label

# To predict probabilities on new data without labels (not used for now)
class TestDataset(Dataset):
    def __init__(self, dataset, labels, batch_size = 16, img_size = (configs.CHANNELS, configs.HEIGHT, configs.WIDTH), img_dir = configs.TEST_IMAGES_DIR, *args, **kwargs):
        self.dataset = dataset
        self.ids = dataset.index.tolist()
        self.labels = labels
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_path = self.df['Path'].iloc[index]

        image = utils._read(image_path)

        if image is None:
            raise FileNotFoundError(f"Image not found : {image_path}")

        # # Normalize
        # image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        # image = image.astype(np.float32) / 255.0  # Normalisation between 0 et 1
        # image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W) for PyTorch

        label = torch.zeros(6)

        return image, label


"""TRAINING VALID AND TEST DATASET"""
# Read the excel with new label
new_label_df = pd.read_excel('excel_predicciones.xlsx', sheet_name='selected_cortes')
new_label_df['Path'] = new_label_df['Identifier'].apply(utils.ajust_path)

# Create the DataFrame for the dataset
data_df = new_label_df[['ANY Vasoespasm ','Path']]
data_df = data_df.rename(columns={'ANY Vasoespasm ':'ANY_Vasospasm'})

# unique_patients = data_df['Path'].apply(lambda x: x.split('/')[patiente]).unique()
# print('unique_patients : ', unique_patients)

# Remove unexistant file/ path from data_df : 
count_0 = len(data_df[data_df["ANY_Vasospasm"] == 0])
count_1 = len(data_df[data_df["ANY_Vasospasm"] == 1])
print("data_df before removing wrong paths : ","count 1 : ", count_1, "count 0 : ", count_0)

data_df = data_df[data_df['Path'].apply(os.path.exists)]

count_0 = len(data_df[data_df["ANY_Vasospasm"] == 0])
count_1 = len(data_df[data_df["ANY_Vasospasm"] == 1])
print("data_df after : ","count 1 : ", count_1, "count 0 : ", count_0)

# np.random.shuffle(unique_patients)

# split_idx_train = int(len(unique_patients) * configs.split_train)
# split_idx_valid = int(len(unique_patients) * (configs.split_train + configs.split_valid))

# train_patients = unique_patients[:split_idx_train]
# valid_patients = unique_patients[split_idx_train:split_idx_valid]
# test_patients = unique_patients[split_idx_valid:]

# Stratified split in patients 
patiente = 8 #index of {patiente} in the path
patient_df = data_df.copy()
patient_df["ID"] = patient_df["Path"].apply(lambda x: x.split('/')[patiente])
print("1 ",patient_df.head(5))
patient_df = patient_df.groupby("ID")["ANY_Vasospasm"].max().reset_index()  # label = 1 if at least one image is positive
print("2 ",patient_df.head(5))

# Everything inside create_dataloader to be able to change the seed with main_40_iterations
def create_dataloader():
    print(patient_df["ANY_Vasospasm"].value_counts())
    train_patients, val_test_patients = train_test_split(
        patient_df,
        train_size=configs.split_train,
        stratify=patient_df["ANY_Vasospasm"],
        random_state=configs.SEED
    )
    print(val_test_patients["ANY_Vasospasm"].value_counts())
    test_size = configs.split_test/(configs.split_valid + configs.split_test)

    valid_patients, test_patients = train_test_split(
        val_test_patients,
        test_size=test_size,
        stratify=val_test_patients["ANY_Vasospasm"],
        random_state=configs.SEED
    )

    # Create DataFrames
    train_df = data_df[data_df['Path'].apply(lambda x: x.split('/')[patiente] in train_patients["ID"].values)]
    valid_df = data_df[data_df['Path'].apply(lambda x: x.split('/')[patiente] in valid_patients["ID"].values)]
    test_df = data_df[data_df['Path'].apply(lambda x: x.split('/')[patiente] in test_patients["ID"].values)]

    # Oversampling for class 1 (~ 28% of 1) only for training !!

    # Method oversampling 1 (nb class 1 = nb class 0)
    count_0 = len(train_df[train_df["ANY_Vasospasm"] == 0])
    df_oversampled = train_df[train_df["ANY_Vasospasm"] == 1].sample(count_0, replace=True, random_state=configs.SEED)
    df_balanced = pd.concat([train_df[train_df["ANY_Vasospasm"] == 0], df_oversampled])
    train_df = df_balanced.sample(frac=1, random_state=configs.SEED).reset_index(drop=True)

    # Method oversampling 2 (double class 1)
    # vasospasm_df = train_df[train_df["ANY_Vasospasm"] == 1]
    # train_oversample_df = pd.concat([train_df, vasospasm_df])
    # train_df = train_oversample_df

    count_0 = len(train_df[train_df["ANY_Vasospasm"] == 0])
    count_1 = len(train_df[train_df["ANY_Vasospasm"] == 1])
    print("count 1 train : ", count_1, "count 0 train : ", count_0)

    # Labels 'ANY_Vasospasm'
    train_labels = train_df["ANY_Vasospasm"]
    valid_labels = valid_df["ANY_Vasospasm"]
    test_labels = test_df["ANY_Vasospasm"]

    # Train Dataset
    train_dataset = TrainDataset(
        dataset=train_df,
        labels=train_labels,
        batch_size=configs.TRAIN_BATCH_SIZE,
        augment=True
    )

    # Validation Dataset
    valid_dataset = TrainDataset(
        dataset=valid_df,
        labels=valid_labels,
        batch_size=configs.VALID_BATCH_SIZE,
        augment=False
    )

    # Test Dataset
    test_dataset = TrainDataset(
        dataset=test_df,
        labels=test_labels,
        batch_size=configs.TEST_BATCH_SIZE,
        augment=False
    )

    # Visualize random images from training set before training
    utils.visualize(3, train_df)

    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=configs.TRAIN_BATCH_SIZE, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=configs.VALID_BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    return trainloader, validloader, testloader

