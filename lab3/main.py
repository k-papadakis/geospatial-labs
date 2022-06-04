# %%
from typing import Any, Dict, Hashable, Optional, Union, Tuple
import random
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import rasterio
from rasterio.windows import Window

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn, tensor
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, RandomSampler, Subset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy, ConfusionMatrix
from torchsummary import summary
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip
import torchvision.transforms.functional as TF 

RANDOM_STATE = 42

INFO_PATH = 'hyrank_info.csv'

TRAIN_X_PATHS = [
    'HyRANK_satellite/TrainingSet/Dioni.tif',
    'HyRANK_satellite/TrainingSet/Loukia.tif',
]

TRAIN_Y_PATHS = [
    'HyRANK_satellite/TrainingSet/Dioni_GT.tif',
    'HyRANK_satellite/TrainingSet/Loukia_GT.tif'
]

VALINDATION_X_PATHS = [
    'HyRANK_satellite/ValidationSet/Erato.tif',
    'HyRANK_satellite/ValidationSet/Kirki.tif',
    'HyRANK_satellite/ValidationSet/Nefeli.tif',
]


# %%

def read_data(x_paths, y_paths):
    images = []
    labels = []
    for x_path, y_path in zip(x_paths, y_paths):
        with rasterio.open(y_path) as y_src:
            y_img = y_src.read(1)
            
        with rasterio.open(x_path) as x_src:
            x_img = x_src.read()
        max_val = np.iinfo(x_img.dtype).max
        x_img = x_img / max_val
        
        images.append(x_img)
        labels.append(y_img)
        
    return images, labels

def read_info(path) -> Dict[int, Dict]:
    with open(INFO_PATH) as f:
        reader = csv.DictReader(f)
        names = reader.fieldnames[1:]
        info = {field: [] for field in names}
        for record in reader:
            for name in names:
                info[name].append(record[name])
    return info

images, label_images = read_data(TRAIN_X_PATHS, TRAIN_Y_PATHS)
info = read_info(INFO_PATH)
cmap = ListedColormap(info['color'])

# %%
# STEP 1 - PLOT SAMPLES

def plot_rgb(images, names, rgb=(23, 11, 7)):
    fig, axs = plt.subplots(len(images), figsize=(12, 8))
    for image, name, ax in zip(images, names, axs.flat):
        image = image[rgb, ...]
        image = image.transpose((1,2,0))
        image = image / image.max()
        ax.imshow(image)
        ax.axis('off')
        ax.set_title(name)
        
    return fig, axs

# res = plot_rgb(images, train_x_paths) 
# plt.show()

# %% STEP 2 - TRAIN PIXEL-WISE
def make_pixel_dataset(images, label_images):
    x_lst = []
    y_lst = []
    for x, y in zip(images, label_images):
        mask = y != 0
        y = y[mask]
        x = x[:, mask]
        x_lst.append(x)
        y_lst.append(y)
    x = np.concatenate(x_lst, 1).T
    y = np.concatenate(y_lst, 0)
    # Remap labels 1..14 --> 0..13
    y = y - 1
    return x, y


# %% Train SVM and RF

def train_traditional():
    x_train, x_test, y_train, y_test = train_test_split(
        *make_pixel_dataset(images, label_images),
        test_size=0.2, random_state=RANDOM_STATE
    )
    models = [
        make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf')),
        RandomForestClassifier(n_estimators=10),
    ]

    for model in models:
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        print(type(model).__name__)
        print(classification_report(y_test, y_pred))
        print()
        
    return models


# %% Train an MLP

class LitMLP(pl.LightningModule):
    
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(128, dim_out),
        )
        
    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, -1)
        return pred 
    
    def training_step(self, batch,  batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch,  batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer


def train_mlp(images, label_images):
    x_train, x_test, y_train, y_test = train_test_split(
        *make_pixel_dataset(images, label_images),
        test_size=0.2, random_state=RANDOM_STATE
    )
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=16)
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.LongTensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=16)
    
    dim_in = x_train.shape[-1]
    dim_out = len(np.unique(y_train))
    mlp = LitMLP(dim_in, dim_out)
    
    callbacks = [
        # EarlyStopping(monitor='val_loss', mode='min', patience=3, min_delta=1e-2),
    ]
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu', callbacks=callbacks)
    trainer.fit(mlp, train_loader, test_loader)

    y_pred = torch.cat(trainer.predict(mlp, test_loader), 0)
    print(classification_report(y_test, y_pred))
    
    return mlp


# %% STEP 3 - TRAIN PATCH WISE

class PatchDataset(Dataset):
    
    def __init__(self, image, label_image, patch_size, channels=None, transform=None):
        super().__init__()
        
        self.image = image
        self.channels = channels if channels is not None else slice(None)
            
        # Keep only the pixels that are labelled and whose patch lies inside the image 
        r = patch_size // 2
        is_inner = np.full_like(label_image, False, bool)
        is_inner[r:-r, r:-r] = True
        self.indices = np.nonzero((label_image != 0) & is_inner)

        self.labels = label_image[self.indices]
        # Remap labels 1..14 --> 0..13
        self.labels = self.labels - 1
            
        self.patch_size = patch_size
        self.transform = transform
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        i, j = self.indices[0][idx], self.indices[1][idx]
        r = self.patch_size // 2
        x = self.image[self.channels, i-r : i+r+1, j-r : j+r+1]
        y = self.labels[idx]
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        
        if self.transform is not None:
            x = self.transform(x)
            
        return x, y
    
class AugmentedDataset(Dataset):
    """"Wraps a dataset to include a transform"""
    
    def __init__(self, dataset, transform, apply_on_target=False):
        self.dataset = dataset
        self.transform = transform
        self.apply_on_target = apply_on_target
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if self.apply_on_target:
            x, y = self.transform(x, y)
        else:
            x = self.transform(x)
        return x, y


def display_patch(dataset, idx, ax=None, rgb=(23, 11, 7)):
    if ax is None:
        fig, ax = plt.subplots()
    patch, label = dataset[idx]
    img = patch[rgb, ...]
    img = img / img.max()
    img = torch.permute(img, (1, 2, 0))
    ax.imshow(img)
    ax.set_axis_off()
    return ax

# %%
def split_dataset(dataset, train_size, test_size=0., seed=None):
    if not 0 <= train_size + test_size <= 1:
        raise ValueError('Invalid train/test sizes')
    n = len(dataset)
    n_train = int(train_size * n)
    n_test = int(test_size * n)
    n_val = n - (n_train + n_test)
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    dataset_train, dataset_val, dataset_test = random_split(
        dataset, [n_train, n_val, n_test], generator)
    return dataset_train, dataset_val, dataset_test


def flip_and_rotate(*tensors):
    # Use this transform only when height == width!
    
    tensors = list(tensors)
    # Expand 2 dimensional tensors so that the transforms work
    expanded = []
    for i in range(len(tensors)):
        if tensors[i].ndim == 2:
            tensors[i] = torch.unsqueeze(tensors[i], 0)
            expanded.append(i)
    
    # Flip vertically
    if random.random() > 0.5:
        tensors = [TF.vflip(t) for t in tensors]
    # Rotate by 0, 1, 2, or 3 right angles 
    if (k := random.randint(0, 4)) != 0:
        tensors = [TF.rotate(t, k * 90) for t in tensors]
    
    # Undo the expansion
    for idx in expanded:
        tensors[idx] = torch.squeeze(tensors[i], 0)

    return tuple(tensors) if len(tensors) > 1 else tensors[0]


patch_size = 15
patch_dataset = ConcatDataset([
    PatchDataset(image, label_image, patch_size)
    for image, label_image in zip(images, label_images)
])

patch_dataset_train, patch_dataset_val, patch_dataset_test = split_dataset(
    patch_dataset, train_size=0.8, test_size=0.1, seed=RANDOM_STATE
)
patch_dataset_train_aug = AugmentedDataset(patch_dataset_train, flip_and_rotate)

# fig, axs = plt.subplots(5, 5, figsize=(12, 12))
# for ax in axs.flat:
#     display_patch(patch_dataset_train_aug, 200, ax=ax)
#     ax.set_axis_off()


patch_loader_train = DataLoader(patch_dataset_train, batch_size=64, shuffle=True, num_workers=8)
patch_loader_train_aug = DataLoader(patch_dataset_train_aug, batch_size=64, shuffle=True, num_workers=8)
patch_loader_val = DataLoader(patch_dataset_val, batch_size=64, num_workers=8)
patch_loader_test = DataLoader(patch_dataset_test, batch_size=64, num_workers=8)


# %% 
class LitCNN(pl.LightningModule):
    
    def __init__(self, channels_in, n_classes, lr=1e-3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*4*64, 128),  # adaptive pool (4, 4)
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        
        self.lr = lr
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.save_hyperparameters()

    def forward(self, x):
        x = self.stem(x)
        x = x + self.block1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) 
        x = x + self.block2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) 
        x = x + self.block3(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) 
        x = x + self.block4(x)
        x = F.adaptive_avg_pool2d(x, output_size=(4, 4))
        x = self.classifier(x)
        return x
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, -1)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/train', loss)

        self.train_accuracy(logits, y)
        self.log('accuracy/train', self.train_accuracy)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/val', loss)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    


def train_cnn():
    callbacks = [
        EarlyStopping(monitor='accuracy/val', mode='max', patience=3),
        ModelCheckpoint(monitor='accuracy/val', mode='max', save_last=True)
    ]
    model = LitCNN(176, 14, lr=1e-3)
    trainer = pl.Trainer(
        accelerator='gpu', 
        max_epochs=20,
        callbacks=callbacks,
        default_root_dir='cnn_results'
    )
    trainer.fit(model, train_dataloaders=patch_loader_train_aug, val_dataloaders=patch_loader_val)

    y_true = torch.cat([y for x, y in patch_loader_test], 0)
    y_pred = torch.cat(trainer.predict(dataloaders=patch_loader_test), 0)
    print(classification_report(y_true, y_pred))
    
    return trainer


# %%
rgb_dataset = ConcatDataset([
    PatchDataset(image, label_image, patch_size, channels=(23, 11, 7))
    for image, label_image in zip(images, label_images)
])
rgb_dataset_train, rgb_dataset_val, rgb_dataset_test = split_dataset(
    rgb_dataset, train_size=0.8, test_size=0.1, seed=RANDOM_STATE
)
rgb_dataset_train_aug = AugmentedDataset(rgb_dataset_train, flip_and_rotate)

rgb_loader_train = DataLoader(rgb_dataset_train, batch_size=64, shuffle=True, num_workers=8)
rgb_loader_train_aug = DataLoader(rgb_dataset_train_aug, batch_size=64, shuffle=True, num_workers=8)
rgb_loader_val = DataLoader(rgb_dataset_val, batch_size=64, num_workers=8)
rgb_loader_test = DataLoader(rgb_dataset_test, batch_size=64, num_workers=8)


# %%

class CroppedDataset(Dataset):
    
    def __init__(self, image, labels_image,
                 crop_size: Union[int, Tuple[int, int]],
                 channels: Optional[Tuple[int, ...]] = None,
                 transform=None,
    ):
        super().__init__()
        
        self.image = image
        self.labels = labels_image
        
        if isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        elif isinstance(crop_size, tuple):
            self.crop_h, self.crop_w = crop_size
        else:
            raise ValueError('Invalid crop_size.')
        
        if self.crop_h > self.img_h or self.crop_w > self.img_w:
            raise ValueError('crops_size is bigger than image size.')
        
        self.channels = channels
        self.transform = transform
        
    def __len__(self):
        return self.n_rows * self.n_cols
    
    def __getitem__(self, idx):
        min_i, min_j, max_i, max_j = self.index_to_bounds(idx)
        channels = self.channels if self.channels is not None else slice(None)
        x = self.image[channels, min_i : max_i, min_j : max_j]
        y = self.labels[min_i : max_i, min_j : max_j]
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        
        if self.transform:
            x, y = self.transform(x, y)
            
        return x, y
    
    def index_to_bounds(self, idx):
        min_i, min_j = divmod(idx, self.n_cols)
        # Max bound is exclusive
        max_i = min_i + self.crop_h
        max_j = min_j + self.crop_w
        return min_i, min_j, max_i, max_j
        
    @property
    def img_h(self):
        return self.image.shape[-2]
    
    @property
    def img_w(self):
        return self.image.shape[-1]
    
    @property
    def n_rows(self):
        return self.img_h - self.crop_h + 1
    
    @property
    def n_cols(self):
        return self.img_w - self.crop_w + 1


# %%
# One image has height 249 while the other has 250
# We will use crop_height 249 for both,
# and ignore the last row of pixels of
# the image with height 250.
# The crop width will be equal to crop_height = 249.
# We will slide the crop window on the first row with
# step crop_width // 4 so that there is some overlap.
crop_size = 249
cropped_datasets_lst = []
for image, label_image in zip(images, label_images):
    ds = CroppedDataset(image, label_image, crop_size)
    indices = range(0, ds.n_cols, ds.crop_w // 4)
    ds = Subset(ds, indices)
    cropped_datasets_lst.append(ds)
cropped_dataset = ConcatDataset(cropped_datasets_lst)
# display_patch(cropped_dataset, 0)
# display_patch(cropped_dataset, 1)

cropped_dataset_train, cropped_dataset_val, cropped_dataset_test = split_dataset(
    cropped_dataset, train_size=0.7, test_size=0.2, seed=RANDOM_STATE
)
    
cropped_dataset_train_aug = AugmentedDataset(
    cropped_dataset_train, transform=flip_and_rotate, apply_on_target=True
)
# print(*map(len, (cropped_dataset_train_aug, cropped_dataset_val, cropped_dataset_test)))

cropped_loader_train_aug = DataLoader(cropped_dataset_train_aug, batch_size=1, shuffle=True, num_workers=1)
cropped_loader_val = DataLoader(cropped_dataset_val, batch_size=1, num_workers=1)
cropped_loader_test = DataLoader(cropped_dataset_test, batch_size=1, num_workers=1)
    
# %%
def display_segmentation(dataset, idx, rgb=(23, 11, 7), cmap=None, axs=None):
    if axs is None:
        fig, axs = plt.subplots(ncols=2)
        
    img, label = dataset[idx]
    img = img[rgb, ...]
    img = img / img.max()
    img = torch.permute(img, (1, 2, 0))
    
    axs[0].imshow(img)
    axs[1].imshow(label, cmap=cmap)
    
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    
    return axs

# fig, axs = plt.subplots(4, 2, figsize=(4, 8))
# for ax1, ax2 in axs:
#     display_segmentation(cropped_dataset_train_aug, 0, cmap=cmap, axs=(ax1, ax2))

# %%
