import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
import pandas as pd
import numpy as np
import pydicom
from albumentations import *
import cv2
# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn.functional as F

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
                # utils.sometimes(0.50, transforms.RandomResizedCrop(size=(self.img_size[1], self.img_size[2]), scale=(0.8, 1.0))),
                utils.sometimes(0.3, transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))),  # Zoom
                utils.sometimes(0.3, transforms.RandomRotation(degrees=30)),   # Rotation
                utils.sometimes(0.3, transforms.ColorJitter(brightness=0.2)),  # Brightness variation (≈ Multiply)
                # utils.sometimes(0.3, transforms.RandomErasing(scale=(0.02, 0.1))),
                # utils.sometimes(0.3, transforms.RandomAffine(degrees=0, translate=(0.2, 0.2))) # Translation

                # Other augmentations :
                # transforms.RandomHorizontalFlip(p=0.25),
                # transforms.RandomVerticalFlip(p=0.10),
                # transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),  
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
        # image
        image_path = self.dataset['Path'].iloc[index]

        image = utils._read(image_path) # size : torch.Size([3, 256, 256])
        # print("SIIIIIIIIIZZZZZZZZZZZZZZZZZZZZZZEEEEEEE ================ : ", image.size())

        if image is None:
            raise FileNotFoundError(f"Image not found : {image_path}")

        # Augmentations for trainloader (if self.augment)
        image = self.transform(image)

        # metadata
        age = torch.tensor([self.dataset['Age'].iloc[index]], dtype=torch.float32)
        age = utils.normalize_min_max(age, 18, 100)
        saps2 = torch.tensor([self.dataset['SAPSII'].iloc[index]], dtype=torch.float32)
        saps2 = utils.normalize_min_max(saps2, 0, 69)
        gcs = torch.tensor([self.dataset['GCS'].iloc[index]], dtype=torch.float32)
        gcs = utils.normalize_min_max_inverted(gcs, 3, 15)
        fisher = torch.tensor([self.dataset['Fisher'].iloc[index]], dtype=torch.float32)
        fisher = utils.normalize_min_max(fisher, 1, 4)
        hunthess = torch.tensor([self.dataset['HuntHess'].iloc[index]], dtype=torch.float32)
        hunthess = utils.normalize_min_max(hunthess, 1, 5)
        wfns = torch.tensor([self.dataset['WFNS'].iloc[index]], dtype=torch.float32)
        wfns = utils.normalize_min_max(wfns, 1, 5)
        sex = self.dataset['Sex'].iloc[index]
        sex = F.one_hot(torch.tensor(sex, dtype=torch.long), num_classes=2).float() # dim = 2

        meta = torch.cat([age, sex, saps2, gcs, fisher, hunthess, wfns], dim=0) # dim = 8
        # label
        label = torch.tensor(self.labels.iloc[index], dtype=torch.float32)

        return {'image':image, 'meta':meta, 'label':label}

# To predict probabilities on new data without labels (not used for now)
class TestDataset(Dataset):
    def __init__(self, dataset, batch_size, img_size = (configs.CHANNELS, configs.HEIGHT, configs.WIDTH)):
        self.dataset = dataset
        self.ids = dataset.index.tolist()
        self.img_size = img_size

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_path = self.dataset['Path'].iloc[index]

        image = utils._read(image_path)

        # if image is None:
        #     # raise FileNotFoundError(f"Image not found : {image_path}")
        #     return None

        # # Normalize
        # image = cv2.resize(image, (self.img_size[1], self.img_size[0]))
        # image = image.astype(np.float32) / 255.0  # Normalisation between 0 et 1
        # image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W) for PyTorch

        label = torch.zeros(6)

        return {'image':image, 'label':label}


"""TRAINING VALID AND TEST DATASET"""
# Read the excel with new label
new_label_df = pd.read_excel(f'{configs.DATA_DIR}excel_predicciones.xlsx', sheet_name='selected_cortes')
new_label_df['Path'] = new_label_df['Identifier'].apply(utils.ajust_path)

# Create the DataFrame for the dataset
data_df = new_label_df[['ANY Vasoespasm ','Path']]
data_df = data_df.rename(columns={'ANY Vasoespasm ':'ANY_Vasospasm'})

# Remove unexistant file/ path from data_df : 
count_0 = len(data_df[data_df["ANY_Vasospasm"] == 0])
count_1 = len(data_df[data_df["ANY_Vasospasm"] == 1])
print("data_df before removing wrong paths : ","count 1 : ", count_1, "count 0 : ", count_0)

data_df = data_df[data_df['Path'].apply(os.path.exists)]

count_0 = len(data_df[data_df["ANY_Vasospasm"] == 0])
count_1 = len(data_df[data_df["ANY_Vasospasm"] == 1])
print("data_df after : ","count 1 : ", count_1, "count 0 : ", count_0)

# No stratified split in patient
# patiente = 8 #index of {patiente} in the path
# unique_patients = data_df['Path'].apply(lambda x: x.split('/')[patiente]).unique()
# print('unique_patients : ', unique_patients)
# np.random.shuffle(unique_patients)

# split_idx_train = int(len(unique_patients) * configs.split_train)
# split_idx_valid = int(len(unique_patients) * (configs.split_train + configs.split_valid))

# train_patients = unique_patients[:split_idx_train]
# valid_patients = unique_patients[split_idx_train:split_idx_valid]
# test_patients = unique_patients[split_idx_valid:]

# Stratified split in patients 
patiente = configs.patient #index of {patiente} in the path
patient_df = data_df.copy()
patient_df["HSA"] = patient_df["Path"].apply(lambda x: x.split('/')[patiente])
patient_df = patient_df.groupby("HSA")["ANY_Vasospasm"].max().reset_index()  # patient's label = 1 if at least one image is positive


"""DATA FRAME META DATA"""
metadata_df = pd.read_excel(f'{configs.DATA_DIR}excel_predicciones.xlsx', sheet_name='datos hospital')
metadata_df = metadata_df[['HSA', 'Edad', 'Sexo', 'SAPSII', 'GCS', 'Fisher', 'HuntHess', 'WFNS']]
metadata_df = metadata_df.rename(columns={'Edad':'Age','Sexo':'Sex'})
metadata_df = metadata_df[:197] # Delete the last lines of the excel that contains totals

# Add metadata to data_df
data_df['HSA'] = data_df['Path'].apply(lambda x: x.split('/')[patiente])
data_df = pd.merge(data_df, metadata_df, on='HSA', how='left')

# Everything inside create_dataloader to be able to change the seed with main_40_iterations
def create_dataloader():
    print(patient_df["ANY_Vasospasm"].value_counts())
    train_patients, val_test_patients = train_test_split(
        patient_df,
        train_size=configs.split_train,
        stratify=patient_df["ANY_Vasospasm"],
        random_state=configs.SEED
    )
    test_size = configs.split_test/(configs.split_valid + configs.split_test)

    valid_patients, test_patients = train_test_split(
        val_test_patients,
        test_size=test_size,
        stratify=val_test_patients["ANY_Vasospasm"],
        random_state=configs.SEED
    )

    # Create DataFrames
    train_df = data_df[data_df['Path'].apply(lambda x: x.split('/')[patiente] in train_patients["HSA"].values)]
    valid_df = data_df[data_df['Path'].apply(lambda x: x.split('/')[patiente] in valid_patients["HSA"].values)]
    test_df = data_df[data_df['Path'].apply(lambda x: x.split('/')[patiente] in test_patients["HSA"].values)]

    # Oversampling for class 1 (~ 28% of 1) only for training !!

    # Method oversampling 1 (nb class 1 = nb class 0)
    # count_0 = len(train_df[train_df["ANY_Vasospasm"] == 0])
    # df_oversampled = train_df[train_df["ANY_Vasospasm"] == 1].sample(count_0, replace=True, random_state=configs.SEED)
    # df_balanced = pd.concat([train_df[train_df["ANY_Vasospasm"] == 0], df_oversampled])
    # train_df = df_balanced.sample(frac=1, random_state=configs.SEED).reset_index(drop=True)

    # Method oversampling 2 (double class 1)
    # vasospasm_df = train_df[train_df["ANY_Vasospasm"] == 1]
    # train_oversample_df = pd.concat([train_df, vasospasm_df])
    # train_df = train_oversample_df

    print("traindf", train_df.head(10))

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

    # Create DataLoaders
    trainloader = DataLoader(train_dataset, batch_size=configs.TRAIN_BATCH_SIZE, shuffle=True)
    validloader = DataLoader(valid_dataset, batch_size=configs.VALID_BATCH_SIZE, shuffle=False)
    testloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return trainloader, validloader, testloader







""" Other Data Loader (from https://github.com/okotaku/kaggle_rsna2019_3rd_solution) """

class RSNADatasetTest(Dataset):

    def __init__(self,
                 df,
                 img_size,
                 image_path,
                 crop_rate = 1.0,
                 id_colname="Image",
                 img_type=".dcm",
                 transforms=None,
                 means=[0.485, 0.456, 0.406],
                 stds=[0.229, 0.224, 0.225],
                 black_crop=False,
                 subdural_window=False,
                 three_window=False,
                 n_tta=1,
                 rescaling=False,
                 user_window=1,
                 pick_type="pre_post"
                 ):
        self.df = df
        self.img_size = img_size
        # self.image_path = image_path
        self.transforms = transforms
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.id_colname = id_colname
        self.img_type = img_type
        self.crop_rate = crop_rate
        self.black_crop = black_crop
        self.subdural_window = subdural_window
        self.three_window = three_window
        self.n_tta = n_tta
        self.transforms2 = Compose([
            #CenterCrop(512 - 50, 512 - 50, p=1.0),
            Resize(img_size, img_size, p=1)
        ])
        self.rescaling = rescaling
        self.user_window = user_window
        self.pick_type = pick_type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row[self.id_colname]
        if self.pick_type == "pre_post":
            img_id_pre = cur_idx_row[["pre1_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["post1_SOPInstanceUID"]].fillna(img_id).values[0]
        elif self.pick_type == "pre_pre":
            img_id_pre = cur_idx_row[["pre1_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["pre2_SOPInstanceUID"]].fillna(img_id_pre).values[0]
        elif self.pick_type == "post_post":
            img_id_pre = cur_idx_row[["post1_SOPInstanceUID"]].fillna(img_id).values[0]
            img_id_post = cur_idx_row[["post2_SOPInstanceUID"]].fillna(img_id_pre).values[0]
        if self.user_window == 1:
            img = self._get_img(idx, 1)
            img_pre = self._get_img(img_id_pre, 2)
            img_post = self._get_img(img_id_post, 3)
        elif self.user_window == 2:
            img_id_prepre = cur_idx_row[["pre2_SOPInstanceUID"]].fillna(img_id_pre).values[0]
            img_id_postpost = cur_idx_row[["post2_SOPInstanceUID"]].fillna(img_id_post).values[0]
            img = self._get_img(img_id, 1)
            img_pre = self._get_img(img_id_prepre, 2)
            img_post = self._get_img(img_id_postpost, 3)

        img = np.concatenate([img, img_pre, img_post], axis=2)

        if self.transforms is not None:
            augmented = self.transforms2(image=img)
            img_tta = augmented['image']
            augmented = self.transforms(image=img)
            img = augmented['image']

        imgs = []
        img = img / 255
        img -= self.means
        img /= self.stds
        img = img.transpose((2, 0, 1))
        imgs.append(torch.FloatTensor(img))
        if self.n_tta >= 2:
            flip_img = img[:, :, ::-1].copy()
            imgs.append(torch.FloatTensor(flip_img))

        if self.n_tta >= 4:
            img_tta = img_tta / 255
            img_tta -= self.means
            img_tta /= self.stds
            img_tta = img_tta.transpose((2, 0, 1))
            imgs.append(torch.FloatTensor(img_tta))
            flip_img_tta = img_tta[:, :, ::-1].copy()
            imgs.append(torch.FloatTensor(flip_img_tta))

        return imgs

    def _get_img(self, img_id, n):
        # img_path = os.path.join(self.image_path, img_id + self.img_type)
        row = self.df[self.df[self.id_colname] == img_id]
        img_path = row["Path"].values[0]
        dataset = pydicom.read_file(img_path)
        image = dataset.pixel_array

        if image.shape[0] != 512:
            image = cv2.resize(image, (512, 512))

        if self.black_crop:
            try:
                mask_img = image > np.mean(image)
                sum_channel = np.sum(mask_img, 2)
                w_cr = np.where(sum_channel.sum(0) != 0)
                h_cr = np.where(sum_channel.sum(1) != 0)
            except:
                print("pass black crop {}".format(img_id))

        if self.subdural_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            image = rescale_image(image, intercept, slope)
            image = window_image(image, 80, 200)
        elif self.three_window:
            window_center, window_width, intercept, slope = get_windowing(dataset)
            image = rescale_image(image, intercept, slope)
            if n == 1:
                image = window_image(image, 80, 200, self.rescaling)
                if not self.rescaling:
                    image = (image - (-20)) / 200
            elif n == 2:
                image = window_image(image, 40, 80, self.rescaling)
                if not self.rescaling:
                    image = (image - 0) / 80
            elif n == 3:
                image = window_image(image, 40, 300, self.rescaling)
                if not self.rescaling:
                    image = (image - (-150)) / 380

        if self.black_crop:
            try:
                image = image[np.min(h_cr):np.max(h_cr) + 1, np.min(w_cr):np.max(w_cr) + 1, :]
            except:
                print("pass black crop {}".format(img_id))

        if not self.subdural_window and not self.three_window:
            min_ = image.min()
            max_ = image.max()
            image = (image - min_) / (max_ - min_)
            image = image * 255

        image = np.expand_dims(image, axis=2)

        return image


def window_image(img, window_center, window_width, rescale=True):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
        img = img * 255

    return img


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]


def rescale_image(img, intercept, slope):
    img = (img * slope + intercept)

    return img
