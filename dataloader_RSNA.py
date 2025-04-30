import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from torch.utils.data import DataLoader

import configs
import utils

""" Dataset generators """
class TrainDataset(Dataset):
    def __init__(self, dataset, labels, batch_size, img_size = (configs.CHANNELS, configs.HEIGHT, configs.WIDTH), img_dir = configs.TRAIN_IMAGES_DIR, augment = False):
        self.dataset = dataset
        self.labels = labels
        self.ids = dataset.index.tolist()
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.augment = augment
        self.transform = self.get_transforms()

    def get_transforms(self):
        base_transforms = []
        if self.augment:
            base_transforms += [
                transforms.RandomHorizontalFlip(p=0.25),
                transforms.RandomVerticalFlip(p=0.10),
                utils.sometimes(0.25, transforms.RandomResizedCrop(size=(self.img_size[1], self.img_size[2]), scale=(0.8, 1.0)))
            ]
        return transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ID = self.ids[index]
        image_path = os.path.join(self.img_dir, ID + ".dcm")

        image = utils._read(image_path, self.img_size)

        image = self.transform(image)

        label = torch.tensor(self.labels.iloc[index].values, dtype=torch.float32)

        return image, label

class TestDataset(Dataset):
    def __init__(self, dataset, labels, batch_size = 16, img_size = (configs.CHANNELS, configs.HEIGHT, configs.WIDTH), img_dir = configs.TEST_IMAGES_DIR, *args, **kwargs):
        self.dataset = dataset
        self.ids = dataset.index.tolist()
        self.labels = labels
        self.img_size = img_size
        self.img_dir = img_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        ID = self.ids[index]
        image_path = os.path.join(self.img_dir, ID + ".dcm")

        image = utils._read(image_path, self.img_size)

        label = torch.zeros(6)

        return image, label
    

""" Import training and test datasets"""

def read_testset(filename = configs.DATA_DIR + "stage_2_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df

def read_trainset(filename = configs.DATA_DIR + "stage_2_train.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)
    duplicates_to_remove = [56346, 56347, 56348, 56349,
                            56350, 56351, 1171830, 1171831,
                            1171832, 1171833, 1171834, 1171835,
                            3705312, 3705313, 3705314, 3705315,
                            3705316, 3705317, 3842478, 3842479,
                            3842480, 3842481, 3842482, 3842483 ]
    df = df.drop(index = duplicates_to_remove)
    df = df.reset_index(drop = True)
    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)
    return df

"""Read Train and Test Datasets"""
test_df_all = read_testset()
train_df_all = read_trainset()

# Reduce the df to only the selected images
test_image_ids = os.listdir(configs.TEST_IMAGES_DIR)
test_image_ids = [id.replace('.dcm', '') for id in test_image_ids]
test_df = test_df_all[test_df_all.index.isin(test_image_ids)]

train_image_ids = os.listdir(configs.TRAIN_IMAGES_DIR)
train_image_ids = [id.replace('.dcm', '') for id in train_image_ids]
train_df = train_df_all[train_df_all.index.isin(train_image_ids)]

# Oversampling the minority class 'epidural'
epidural_df = train_df[train_df.Label['epidural'] == 1]
train_oversample_df = pd.concat([train_df, epidural_df])
train_df = train_oversample_df

# Perform stratified split
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=configs.TEST_SIZE, random_state=configs.SEED)
X = train_df.index
Y = train_df.Label.values
train_idx, valid_idx = next(msss.split(X, Y))

labels = [('Label',              'any'),
            ('Label',         'epidural'),
            ('Label', 'intraparenchymal'),
            ('Label', 'intraventricular'),
            ('Label',     'subarachnoid'),
            ('Label',         'subdural')]

# Train Dataset
train_dataset = TrainDataset(
    dataset=train_df.iloc[train_idx],
    labels=train_df.iloc[train_idx][labels],
    batch_size=configs.TRAIN_BATCH_SIZE,
    img_dir=configs.TRAIN_IMAGES_DIR,
    augment=True
)

# Validation Dataset
valid_dataset = TrainDataset(
    dataset=train_df.iloc[valid_idx],
    labels=train_df.iloc[valid_idx][labels],
    batch_size=configs.VALID_BATCH_SIZE,
    img_dir=configs.TRAIN_IMAGES_DIR,
    augment=False
)

# Test Dataset
test_dataset = TestDataset(
    dataset=test_df,
    labels=None,
    img_dir=configs.TEST_IMAGES_DIR
)

# Create DataLoaders
def create_dataloader():
    trainloader = DataLoader(train_dataset, batch_size=configs.TRAIN_BATCH_SIZE, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=configs.VALID_BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    return trainloader, validloader, testloader
