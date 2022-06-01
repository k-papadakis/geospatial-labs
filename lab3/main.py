# %%
from cProfile import label
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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

    x = np.concatenate(x_lst, axis=1).T
    y = np.concatenate(y_lst, axis=0)
    y = y - 1  # remap labels 1..14 --> 0..13

    return x, y


x_train, x_test, y_train, y_test = train_test_split(
    *read_flattened_pixels(train_x_paths, train_y_paths),
    test_size=0.2
)

# TODO: X ARE INT. FIX THIS.

# %% Train SVM and RF

def train_traditional(x_train, y_train, x_test, y_test):
    models = [
        SVC(C=1.0, kernel='rbf'),
        RandomForestClassifier(n_estimators=10),
        MLPClassifier(
            hidden_layer_sizes=(32, 64),
            learning_rate_init=0.001,
            alpha=0.0001
        )
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
        self.linear1 = nn.Linear(dim_in, 128)
        self.linear2 = nn.Linear(128, 128)
        self.linear3 = nn.Linear(128, dim_out)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.dropout1(x)
        x = F.relu(x)
        
        x = self.linear2(x)
        x = self.dropout2(x)
        x = F.relu(x)
        
        x = self.linear3(x)
        
        return x
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, -1)
        return pred 
    
    def training_step(self, batch,  batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch,  batch_idx):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)
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
        # EarlyStopping(monitor="val_loss", mode="min", patience=3, min_delta=1e-2),
    ]
    trainer = pl.Trainer(max_epochs=10, accelerator="gpu", callbacks=callbacks)
    trainer.fit(mlp, train_loader, test_loader)

    y_pred = torch.cat(trainer.predict(mlp, test_loader), 0)
    print(classification_report(y_test, y_pred))
    
    return mlp
    

# %% STEP 3 - TRAIN PATCH WISE

class PatchDataset(Dataset):
    
    def __init__(self, image_path, labels_path, patch_size=7):
        super().__init__()
        
        self.image_path = image_path
        self.labels_path = labels_path
        self.patch_size = patch_size
        
        with rasterio.open(image_path) as src:
            self.height = src.height
            self.width = src.width
            
        with rasterio.open(labels_path) as src:
            labels = src.read(1)
        r = patch_size // 2
        
        is_inner = np.full_like(labels, False, bool)
        is_inner[r:-r, r:-r] = True
        
        self.indices = np.nonzero((labels != 0) & is_inner)  # (i_array, j_array)
        
        self.labels = labels[self.indices]
            
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        i, j = self.indices[0][idx], self.indices[1][idx]
        r = self.patch_size // 2
        window = Window.from_slices((i-r, i+r+1), (j-r, j+r+1))
        with rasterio.open(self.image_path) as src:
            x = src.read(window=window)
        
        y = self.labels[idx]
        
        x = torch.tensor(x, dtype=torch.float32)  # TODO: THIS IS INT. FIX IT
        y = torch.tensor(y, dtype=torch.int64)
        return x, y
    

dataset = ConcatDataset(
    [PatchDataset(image_path, labels_path) for image_path, labels_path in zip(train_x_paths, train_y_paths)]
)

dataloader = DataLoader(dataset, batch_size=64, num_workers=16)

x, y = next(iter(dataloader))