import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import cv2
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
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
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.10),
                utils.sometimes(0.25, transforms.RandomResizedCrop(size=(self.img_size[1], self.img_size[2]), scale=(0.8, 1.0))),
                # utils.sometimes(0.25, transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))),
                # utils.sometimes(0.25, transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)))  # Zoom

                # transforms.RandomHorizontalFlip(p=0.5),  # Fliplr
                # transforms.RandomVerticalFlip(p=0.5),    # Flipud
                # transforms.RandomRotation(degrees=20),   # Rotation
                # transforms.ColorJitter(brightness=0.2),  # Brightness variation (≈ Multiply)
                # transforms.RandomAffine(degrees=0, scale=(0.8, 1.2)),  # Zoom
                # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  # Translation
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

        image = utils._read(image_path, self.img_size)

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

        image = utils._read(image_path, self.img_size)

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


# Get patient without duplicates
patiente = 11 #index of {patiente} in the path
unique_patients = data_df['Path'].apply(lambda x: x.split('\\')[patiente]).unique()

# To remove once we have all the data
missing_files = ['HSA 77', 'HSA 79', 'HSA 80', 'HSA 82',
 'HSA 87', 'HSA 89', 'HSA 90', 'HSA 91', 'HSA 92', 'HSA 93', 'HSA 94', 'HSA 95',
 'HSA 96', 'HSA 97', 'HSA 98', 'HSA 105', 'HSA 106', 'HSA 109', 'HSA 111',
 'HSA 112', 'HSA 114', 'HSA 117', 'HSA 118', 'HSA 120', 'HSA 121', 'HSA 122',
 'HSA 123', 'HSA 126', 'HSA 127', 'HSA 129', 'HSA 130', 'HSA 131', 'HSA 132',
 'HSA 133', 'HSA 134', 'HSA 135', 'HSA 137', 'HSA 138', 'HSA 139', 'HSA 141',
 'HSA 143', 'HSA 144', 'HSA 145A', 'HSA 145B', 'HSA 146', 'HSA 149', 'HSA 150',
 'HSA 152', 'HSA 153', 'HSA 154', 'HSA 155', 'HSA 156', 'HSA 158', 'HSA 159',
 'HSA 160', 'HSA 161', 'HSA 163', 'HSA 164', 'HSA 166', 'HSA 167', 'HSA 168',
 'HSA 169', 'HSA 170', 'HSA 171', 'HSA 173', 'HSA 174', 'HSA 175']
unique_patients = [p for p in unique_patients if p not in missing_files]

print('unique_patients : ', unique_patients)

np.random.seed(configs.SEED)
np.random.shuffle(unique_patients)

split_idx_train = int(len(unique_patients) * configs.split_train)
split_idx_valid = int(len(unique_patients) * (configs.split_train + configs.split_valid))

train_patients = unique_patients[:split_idx_train]
valid_patients = unique_patients[split_idx_train:split_idx_valid]
test_patients = unique_patients[split_idx_valid:]

# Create DataFrames
train_df = data_df[data_df['Path'].apply(lambda x: x.split('\\')[patiente] in train_patients)]
valid_df = data_df[data_df['Path'].apply(lambda x: x.split('\\')[patiente] in valid_patients)]
test_df = data_df[data_df['Path'].apply(lambda x: x.split('\\')[patiente] in test_patients)]


# Oversampling for class 1 (~ 15% of 1) only for training !!

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
utils.visualize(1, train_df)


# Create DataLoaders
def create_dataloader():
    trainloader = DataLoader(train_dataset, batch_size=configs.TRAIN_BATCH_SIZE, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=configs.VALID_BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return trainloader, validloader, testloader

