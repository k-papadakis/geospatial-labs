# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy, ConfusionMatrix
from torchsummary import summary

RANDOM_STATE = 42

# %%
train_x_paths = [
    'HyRANK_satellite/TrainingSet/Dioni.tif',
    'HyRANK_satellite/TrainingSet/Loukia.tif',
]

train_y_paths = [
    'HyRANK_satellite/TrainingSet/Dioni_GT.tif',
    'HyRANK_satellite/TrainingSet/Loukia_GT.tif'
]

validation_x_paths = [
    'HyRANK_satellite/ValidationSet/Erato.tif',
    'HyRANK_satellite/ValidationSet/Kirki.tif',
    'HyRANK_satellite/ValidationSet/Nefeli.tif',
]


# %%
# STEP 1 - PLOT SAMPLES

def plot_rgb(*paths, rgb=(23, 11, 7)):
    fig, axs = plt.subplots(len(paths), figsize=(12, 12))

    for p, ax in zip(paths, axs.flat):
        p = Path(p)
        with rasterio.open(p) as src:
            img = src.read(rgb).transpose((1,2,0))
            img = img / img.max()
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(p)
        
    return fig, axs


# plot_rgb(*train_x_paths, *validation_x_paths) 
# plt.show()

# %% STEP 2 - TRAIN PIXEL-WISE

def read_flattened_pixels(x_paths, y_paths):
    x_lst = []
    y_lst = []

    for x_path, y_path in zip(x_paths, y_paths):
        with rasterio.open(y_path) as y_src:
            y_img = y_src.read(1)
        with rasterio.open(x_path) as x_src:
            x_img = x_src.read()
        
        mask = y_img != 0
        y = y_img[mask]
        x = x_img[:, mask]

        x_lst.append(x)
        y_lst.append(y)

    x = np.concatenate(x_lst, 1).T
    # Map all values in [0, 1]
    max_val = np.iinfo(x.dtype).max
    x = x / max_val
    y = np.concatenate(y_lst, 0)
    # Remap labels 1..14 --> 0..13
    y = y - 1

    return x, y


x_train, x_test, y_train, y_test = train_test_split(
    *read_flattened_pixels(train_x_paths, train_y_paths),
    test_size=0.2
)

# %% Train SVM and RF

def train_traditional(x_train, y_train, x_test, y_test):
    models = [
        make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf')),
        RandomForestClassifier(n_estimators=10),
    ]

    for model in models:
        model.fit(x_train, y_train)
        
    for model in models:
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


def train_mlp(x_train, y_train, x_test, y_test):
    train_dataset = TensorDataset(torch.Tensor(x_train), torch.LongTensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=16)
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
    
    def __init__(self, image_path, labels_path, patch_size, transform=None):
        super().__init__()
        
        self.image_path = image_path
        self.labels_path = labels_path
        self.patch_size = patch_size
        self.transform = transform
        
        with rasterio.open(image_path) as src:
            image = src.read()
        
        # Map all values in [0, 1]
        max_val = np.iinfo(image.dtype).max
        self.image = image / max_val
            
        with rasterio.open(labels_path) as src:
            labels = src.read(1)
        
        # Keep only the pixels that are labelled and whose patch lies inside the image 
        r = patch_size // 2
        is_inner = np.full_like(labels, False, bool)
        is_inner[r:-r, r:-r] = True
        self.indices = np.nonzero((labels != 0) & is_inner)
        
        self.labels = labels[self.indices]
        # Remap labels 1..14 --> 0..13
        self.labels -= 1
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        i, j = self.indices[0][idx], self.indices[1][idx]
        r = self.patch_size // 2
        x = self.image[:, i-r : i+r+1, j-r : j+r+1]
        y = self.labels[idx]
        
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int64)
        
        if self.transform is not None:
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
    return ax


def split_dataset(dataset, train_size, test_size=0., seed=RANDOM_STATE):
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


patch_size = 15

dataset = ConcatDataset(
    [PatchDataset(image_path, labels_path, patch_size)
     for image_path, labels_path in zip(train_x_paths, train_y_paths)]
)

dataset_train, dataset_val, dataset_test = split_dataset(
    dataset, train_size=0.8, test_size=0.1
)

loader_train = DataLoader(dataset_train, batch_size=64, num_workers=8)
loader_val = DataLoader(dataset_val, batch_size=64, num_workers=8)
loader_test = DataLoader(dataset_test, batch_size=64, num_workers=8)

# TODO: ADD TRANSFORMS

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
    model = LitCNN(176, 14)
    trainer = pl.Trainer(
        accelerator='gpu', 
        max_epochs=20,
        callbacks=callbacks,
        default_root_dir='cnn_results'
    )
    trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)

    y_true = torch.cat([y for x, y in loader_test], 0)
    y_pred = torch.cat(trainer.predict(dataloaders=loader_test), 0)
    print(classification_report(y_true, y_pred))
    
    return trainer
    

# %%
