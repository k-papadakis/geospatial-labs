# %%
from typing import Dict, List, Optional, Union, Tuple
import warnings
from pathlib import Path
import itertools
import random
import csv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import rasterio

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, ConcatDataset, Subset, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchmetrics
from torchsummary import summary
import torchvision.transforms.functional as TF 
from torchvision.models import resnet18


##########################################################
# PYTORCH DATASETS
##########################################################

class PatchDatasetNoPad(Dataset):
    """Patches of specified size around the pixels whose label is non zero.
    No use of padding.
    Instead, ignore pixels whose patch doesn't lie inside the image.
    """
    
    def __init__(self, image, label_image, patch_size, channels=None, transform=None):
        super().__init__()
        
        self.image = image
        self.channels = channels if channels is not None else slice(None)
            
        # Keep only the pixels that are labelled and whose patch lies inside the image 
        r = patch_size // 2
        is_inner = np.full(image.shape[-2:], False, bool)
        is_inner[r:-r, r:-r] = True
        if label_image is not None:
            self.indices = np.nonzero((label_image != 0) & is_inner)
            self.labels = label_image[self.indices]
            # Remap labels 1..14 --> 0..13
            self.labels = self.labels - 1
        else:
            self.indices = np.nonzero(is_inner)
            self.labels = None
            
        self.patch_size = patch_size
        self.transform = transform
        
    def __len__(self):
        return len(self.indices[0])
    
    def __getitem__(self, idx):
        i, j = self.indices[0][idx], self.indices[1][idx]
        r = self.patch_size // 2
        x = self.image[self.channels, i-r : i+r+1, j-r : j+r+1]
        
        x = torch.tensor(x, dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        
        if self.labels is not None:
            y = self.labels[idx]
            y = torch.tensor(y, dtype=torch.int64)
            return x, y
        else:
            return x


class PatchDatasetPrePad(Dataset):
    """Patches of specified size around the pixels whose label is non zero.
    Pads the entire image before slicing. Results in uneven padding
    """
    
    def __init__(self, image, label_image, patch_size, channels=None, transform=None):
        super().__init__()
        
        self.transform = transform
        self.channels = channels if channels is not None else slice(None)
        self.patch_size = patch_size
        r = patch_size // 2
        self.padded_image = np.pad(image, ((0,0), (r,r), (r,r)))
        
        if label_image is not None:
            # Keep only the pixels that are labelled
            self.indices = np.nonzero(label_image)
            self.labels = label_image[self.indices]
            # Remap labels 1..14 --> 0..13
            self.labels = self.labels - 1
        else:
           self.indices = tuple(map(np.ravel, np.indices(image.shape[-2:])))
           self.labels = None
        
    def __len__(self):
        return len(self.indices[0])
    
    def __getitem__(self, idx):
        i, j = self.indices[0][idx], self.indices[1][idx]
        r = self.patch_size // 2
        x = self.padded_image[self.channels, i : 1+i+2*r, j : 1+j+2*r]
        
        x = torch.tensor(x, dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        
        if self.labels is not None:    
            y = self.labels[idx]
            y = torch.tensor(y, dtype=torch.int64)
            return x, y
        else:
            return x


class PatchDatasetPostPad(Dataset):
    """Patches of specified size around the pixels whose label is non zero.
    Slices a patch and then pads it equally on each side.
    """
    
    def __init__(self, image, label_image, patch_size, channels=None, transform=None):
        super().__init__()
        
        self.transform = transform
        self.channels = channels if channels is not None else slice(None)
        self.patch_size = patch_size
        self.image = image
        
        if label_image is not None:
            # Keep only the pixels that are labelled
            self.indices = np.nonzero(label_image)
            self.labels = label_image[self.indices]
            # Remap labels 1..14 --> 0..13
            self.labels = self.labels - 1
        else:
           self.indices = tuple(map(np.ravel, np.indices(image.shape[-2:])))
           self.labels = None
        
    def __len__(self):
        return len(self.indices[0])
    
    def __getitem__(self, idx):
        i, j = self.indices[0][idx], self.indices[1][idx]
        r = self.patch_size // 2
        
        i_min = max(0, i-r)
        i_max = min(i+r+1, self.image.shape[-2] - 1)
        j_min = max(0, j-r)
        j_max = min(j+r+1, self.image.shape[-1] - 1)
        x = self.image[self.channels, i_min : i_max, j_min : j_max]
        
        h, w = x.shape[-2:]
        h_pad = self.patch_size - h
        w_pad = self.patch_size - w
        padding = (
            (0, 0),
            (h_pad // 2, h_pad - h_pad//2),
            (w_pad // 2, w_pad - w_pad//2)
        )
        x = np.pad(x, padding)
        
        x = torch.tensor(x, dtype=torch.float32)
        if self.transform is not None:
            x = self.transform(x)
        
        if self.labels is not None:    
            y = self.labels[idx]
            y = torch.tensor(y, dtype=torch.int64)
            return x, y
        else:
            return x
        
        
class CroppedDataset(Dataset):
    """Sliding window over an image and its label"""
    
    def __init__(self, image, label_image,
                 crop_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1,
                 channels: Optional[Tuple[int, ...]] = None,
                 transform=None,
    ):
        super().__init__()
        
        self.image = image
        self.img_h = self.image.shape[-2]
        self.img_w = self.image.shape[-1]
        
        if isinstance(crop_size, int):
            self.crop_h, self.crop_w = crop_size, crop_size
        elif isinstance(crop_size, tuple):
            self.crop_h, self.crop_w = crop_size
        else:
            raise ValueError('Invalid crop_size.')
        
        if self.crop_h > self.img_h or self.crop_w > self.img_w:
            raise ValueError('crops_size is bigger than image size.')
        
        if isinstance(stride, int):
            self.stride_h, self.stride_w = stride, stride
        elif isinstance(stride, tuple):
            self.stride_h, self.stride_w = stride
        else:
            raise ValueError('Invalid stride.')
        
        self.n_rows = 1 + (self.img_h - self.crop_h)//self.stride_h
        self.n_cols = 1 + (self.img_w - self.crop_w)//self.stride_w
        self.labels = label_image
        
        self.channels = channels if channels is not None else slice(None)
        self.transform = transform
        
    def __len__(self):
        return self.n_rows * self.n_cols
    
    def __getitem__(self, idx):
        min_i, min_j, max_i, max_j = self.get_bounds(idx)
        
        x = self.image[self.channels, min_i : max_i, min_j : max_j]
        x = torch.tensor(x, dtype=torch.float32)
        
        if self.labels is not None:
          y = self.labels[min_i : max_i, min_j : max_j]
          y = torch.tensor(y, dtype=torch.int64)
          if self.transform:
            x, y = self.transform(x, y)
          return x, y
        else:
          if self.transform:
            x = self.transform(x)
          return x

    def get_bounds(self, idx):
        r, c = divmod(idx, self.n_cols)
        min_i, min_j = r * self.stride_h, c * self.stride_w
        max_i, max_j = min_i + self.crop_h, min_j + self.crop_w
        return min_i, min_j, max_i, max_j


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


##########################################################
# PYTORCH MODELS
##########################################################

class LitMLP(pl.LightningModule):
    """Simple 4-layer MLP with Dropout and L2 normalization"""
    
    def __init__(self, dim_in, dim_out, lr=1e-3, weight_decay=0, p_dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.Dropout(p_dropout),
            nn.ReLU(),
            nn.Linear(128, dim_out),
        )
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        
        self.train_dice = torchmetrics.Dice()
        self.val_dice = torchmetrics.Dice()
        self.test_dice = torchmetrics.Dice()
        
        self.train_confusion_matrix = torchmetrics.ConfusionMatrix(dim_out)
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(dim_out)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(dim_out)
        
        self.save_hyperparameters()
        
    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, 1)
        return pred 
    
    def training_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/train', loss, on_epoch=True, on_step=False)
        
        self.train_accuracy(logits, y)
        self.log('accuracy/train', self.train_accuracy, on_epoch=True, on_step=False)
        
        self.train_dice(logits, y)
        self.log('dice/train', self.train_dice, on_epoch=True, on_step=False)
        
        self.train_confusion_matrix(logits, y)
        
        return loss
    
    def validation_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/val', loss, on_epoch=True, on_step=False)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy, on_epoch=True, on_step=False)
        
        self.val_dice(logits, y)
        self.log('dice/val', self.val_dice, on_epoch=True, on_step=False)
        
        self.val_confusion_matrix(logits, y)
        
    def test_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        self.test_accuracy(logits, y)
        self.test_dice(logits, y)
        self.test_confusion_matrix(logits, y)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer
    

class LitCNN(pl.LightningModule):
    """Deep CNN with skip connections between convolutions that are two steps away
    Expects a 15 by 15 image.
    """
    
    def __init__(self, channels_in, n_classes, lr=1e-3):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(channels_in, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3*3*64, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes),
        )
        
        self.lr = lr
        
        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        
        self.train_dice = torchmetrics.Dice()
        self.val_dice = torchmetrics.Dice()
        self.test_dice = torchmetrics.Dice()
        
        self.train_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        
        self.save_hyperparameters()

    def forward(self, x):
        x = self.stem(x)
        x = x + self.block1(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) 
        x = x + self.block2(x)
        x = F.max_pool2d(x, kernel_size=2, stride=2) 
        x = x + self.block3(x)
        x = self.classifier(x)
        return x
    
    def predict_step(self, batch, batch_idx):
        x = batch
        logits = self(x)
        pred = torch.argmax(logits, -1)
        return pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/train', loss, on_epoch=True, on_step=False)

        self.train_accuracy(logits, y)
        self.log('accuracy/train', self.train_accuracy, on_epoch=True, on_step=False)
        
        self.train_dice(logits, y)
        self.log('dice/train', self.train_dice, on_epoch=True, on_step=False)
        
        self.train_confusion_matrix(logits, y)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/val', loss, on_epoch=True, on_step=False)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy, on_epoch=True, on_step=False)
        
        self.val_dice(logits, y)
        self.log('dice/val', self.val_dice, on_epoch=True, on_step=False)
        
        self.val_confusion_matrix(logits, y)
        
    def test_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        self.test_accuracy(logits, y)
        self.test_dice(logits, y)
        self.test_confusion_matrix(logits, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# %%
# U-Net Construction
class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.model(x)
    

class DownBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_channels, out_channels)
        )
        
    def forward(self, x):
        return self.model(x)
    
    
class UpBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, 2, 2)
        self.conv = ConvBlock(in_channels, out_channels)  # in_channels because of concat
        
    
    def forward(self, x, x_res):
        x = self.convt(x)
        dh = x_res.shape[-2] - x.shape[-2]
        dw = x_res.shape[-1] - x.shape[-1]
        assert dh >= 0 and dw >= 0
        padding = dh//2, dh - dh//2, dw//2, dw - dw//2
        x = F.pad(x, padding)
        x_cat = torch.cat((x, x_res), -3)
        return self.conv(x_cat)
    

class UNet(nn.Module):
    
    def __init__(self, n_channels, n_classes):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        self.entrance = ConvBlock(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)
        self.up1 = UpBlock(1024, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        self.exit = nn.Conv2d(64, n_classes, 1)
        
    def forward(self, x):
        a1 = self.entrance(x)
        a2 = self.down1(a1)
        a3 = self.down2(a2)
        a4 = self.down3(a3)
        t = self.down4(a4)
        b1 = self.up1(t, a4)
        b2 = self.up2(b1, a3)
        b3 = self.up3(b2, a2)
        b4 = self.up4(b3, a1)
        logits = self.exit(b4)
        return logits
    

class LitUNet(pl.LightningModule):
    
    def __init__(self, n_channels, n_classes, ignore_index=None, lr=1e-4):
        super().__init__()
        self.unet = UNet(n_channels, n_classes)
        self.lr = lr
        
        # TODO: Dice loss is consistently about 1/10 lower than in the other models. Bug?
        
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        self.train_accuracy = torchmetrics.Accuracy(ignore_index=ignore_index, mdmc_average='global')
        self.val_accuracy = torchmetrics.Accuracy(ignore_index=ignore_index, mdmc_average='global')
        self.test_accuracy = torchmetrics.Accuracy(ignore_index=ignore_index, mdmc_average='global')
        
        self.train_dice = torchmetrics.Dice(ignore_index=ignore_index, mdmc_average='global')
        self.val_dice = torchmetrics.Dice(ignore_index=ignore_index, mdmc_average='global')
        self.test_dice = torchmetrics.Dice(ignore_index=ignore_index, mdmc_average='global')
        
        self.train_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.unet(x)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, 1)
        return pred
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        self.log('loss/train', loss, on_epoch=True, on_step=False)
        
        self.train_accuracy(logits, y)
        self.log('accuracy/train', self.train_accuracy, on_epoch=True, on_step=False)
        
        self.train_dice(logits, y)
        self.log('dice/train', self.train_dice, on_epoch=True, on_step=False)
        
        self.train_confusion_matrix(logits, y)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        self.log('loss/val', loss, on_epoch=True, on_step=False)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy, on_epoch=True, on_step=False)
        
        self.val_dice(logits, y)
        self.log('dice/val', self.val_dice, on_epoch=True, on_step=False)
        
        self.val_confusion_matrix(logits, y)
                
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        self.test_accuracy(logits, y)
        self.test_dice(logits, y)
        self.test_confusion_matrix(logits, y)


class TransferResNet(pl.LightningModule):
    """ResNet with the last layer replaced so that its output matches n_classes"""
    
    def __init__(self, n_classes, freeze_head=False, lr=1e-3):
        super().__init__()
        
        self.model = resnet18(pretrained=True)
        self.freeze_head = freeze_head
        if self.freeze_head:
            for param in self.model.parameters():
                param.requires_grad = False
                
        self.model.fc = nn.Linear(512, n_classes)

        self.lr = lr

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()
        
        self.train_dice = torchmetrics.Dice()
        self.val_dice = torchmetrics.Dice()
        self.test_dice = torchmetrics.Dice()
        
        self.train_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        self.val_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        self.test_confusion_matrix = torchmetrics.ConfusionMatrix(n_classes)
        
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        pred = torch.argmax(logits, -1)
        return pred 
    
    def training_step(self, batch,  batch_idx, optimizer_idx=None):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/train', loss, on_epoch=True, on_step=False)
        
        self.train_accuracy(logits, y)
        self.log('accuracy/train', self.train_accuracy, on_epoch=True, on_step=False)
        
        self.train_dice(logits, y)
        self.log('dice/train', self.train_dice, on_epoch=True, on_step=False)
        
        self.train_confusion_matrix(logits, y)
        
        return loss
    
    def validation_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/val', loss, on_epoch=True, on_step=False)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy, on_epoch=True, on_step=False)
        
        self.val_dice(logits, y)
        self.log('dice/val', self.val_dice, on_epoch=True, on_step=False)
        
        self.val_confusion_matrix(logits, y)
        
    def test_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        self.test_accuracy(logits, y)
        self.test_dice(logits, y)
        self.test_confusion_matrix(logits, y)

    def configure_optimizers(self):
        children = list(self.model.children())
        
        tail_params = children[-1].parameters()
        tail_optim = torch.optim.Adam(tail_params, lr=self.lr)
        
        if self.freeze_head:
            return tail_optim
        else:
            head_params = itertools.chain.from_iterable(
                child.parameters() for child in children[:-1]
            )
            head_optim = torch.optim.Adam(head_params, lr=self.lr / 200)
            return head_optim, tail_optim
        
        
##########################################################
# PYTORCH TRANSFORMS
##########################################################

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
    k = random.randint(0, 4)
    if k != 0:
        tensors = [TF.rotate(t, k * 90) for t in tensors]
    
    # Undo the expansion
    for idx in expanded:
        tensors[idx] = torch.squeeze(tensors[idx], 0)

    return tuple(tensors) if len(tensors) > 1 else tensors[0]


##########################################################
# PLOTTING FUNCTIONS
##########################################################
def display_image_and_label(
    image, label, cmap, classnames, rgb,
    title=None, axs=None,
):
    if axs is None:
        fig, axs = plt.subplots(nrows=2, figsize=(16, 6))
    image = image[rgb, ...].transpose(1, 2, 0)
    image = image / image.max()
    axs[0].imshow(image)
    axs[1].imshow(label, cmap=cmap)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    
    norm = mpl.colors.Normalize(0, cmap.N)
    cbar = plt.gcf().colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axs, shrink=0.9, location='right'
    )
    cbar.set_ticks(0.5 + np.arange(cmap.N))
    cbar.set_ticklabels(classnames)
    plt.suptitle(title)
    
    return axs


def display_patch(dataset, idx, rgb, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    patch, label = dataset[idx]
    img = patch[rgb, ...]
    img = img / img.max()
    img = torch.permute(img, (1, 2, 0))
    ax.imshow(img)
    ax.set_axis_off()
    return ax


def display_segmentation(dataset, idx, cmap, names, rgb, n_repeats=1, axs=None):
    if axs is None:
        fig, axgrid = plt.subplots(nrows=n_repeats, ncols=2, figsize= (4, n_repeats))
        
    img, label = dataset[idx]
    img = img[rgb, ...]
    img = img / img.max()
    img = torch.permute(img, (1, 2, 0))
    
    for ax1, ax2 in axgrid:
        ax1.imshow(img)
        ax2.imshow(label, cmap=cmap)
        ax1.set_axis_off()
        ax2.set_axis_off()
    
    norm = mpl.colors.Normalize(0, cmap.N)
    cbar = plt.gcf().colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axgrid, shrink=0.9, location='right'
    )
    cbar.set_ticks(0.5 + np.arange(cmap.N))
    cbar.set_ticklabels(names)
    
    return axgrid


##########################################################
# UTILITY FUNCTIONS
##########################################################

def load_image(src_path):
    with rasterio.open(src_path) as src:
            image = src.read()
    # Map integers to floats in [0, 1]    
    max_val = np.iinfo(image.dtype).max
    image = image / max_val
    return image


def load_label(src_path):
    with rasterio.open(src_path) as src:
        label = src.read(1)
    return label
    
    
def read_data(image_paths, label_paths):
    """Loads the images and labels in two separate lists."""
    images = []
    labels = []
    for x_path, y_path in zip(image_paths, label_paths):
        x_img = load_image(x_path)
        y_img = load_label(y_path)
        images.append(x_img)
        labels.append(y_img)
        
    return images, labels


def read_info(path) -> Dict[str, List]:
    """Read class colors and names"""
    with open(path) as f:
        reader = csv.DictReader(f)
        info = {field: [] for field in reader.fieldnames}
        for record in reader:
            for name in reader.fieldnames:
                info[name].append(record[name])
    return info


def split_dataset(dataset, train_size, test_size=0., seed=None):
    """Splits a PyTorch Dataset into train, validation and test Subsets."""
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


def make_pixel_dataset(images, label_images):
    """Select pixels with non-zero labels from a list of images.
    Return the pixel values and their labels.
    """
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


##########################################################
# MODEL TRAINING FUNCTIONS (Hardcoded)
##########################################################

def evaluate_model(trainer, dataset_test, names, ignore_index=None):
    """Test a model and display the accuracy, the dice loss and the confusion matrix.
    This function assumes a very specific structure of the input.
    """
    best_model = trainer.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(best_model, dataloaders=dataset_test)
    confusion_matrix = best_model.test_confusion_matrix.compute().cpu().numpy()
    if ignore_index is not None:
        if np.any(confusion_matrix[:, ignore_index]):
            warnings.warn('`confusion_matrix` has predictions with label `ignore_index`')
        else:
            del names[ignore_index]
            confusion_matrix = np.delete(confusion_matrix, ignore_index, 0)
            confusion_matrix = np.delete(confusion_matrix, ignore_index, 1)
    
    title = type(best_model).__name__
    print(title)
    print(f'Test Accuracy: {best_model.test_accuracy.compute(): .2%}')
    print(f'Test Dice Loss: {best_model.test_dice.compute(): .4f}')
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=names,
    ).plot(xticks_rotation=90, ax=ax, colorbar=False)
    ax.set_title(title)
    plt.savefig(f'{title}.png')
    return ax
    

def train_traditional(images, label_images, names, random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(
        *make_pixel_dataset(images, label_images),
        test_size=0.3, random_state=random_state
    )
    models = [
        make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf')),
        RandomForestClassifier(n_estimators=10),
    ]
    titles = ['RandomForestClassifier', 'SVM(rbf)']
    
    fig, axs = plt.subplots(ncols=len(models), figsize=(9 * len(models), 9))
    
    for model, ax, title in zip(models, axs.flat, titles):
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        print(title)
        print(classification_report(y_test, y_pred))

        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=names,
        ).plot(xticks_rotation=90, ax=ax, colorbar=False)
        ax.set_title(title)
        
        print('-'*70)
    
    axs[1].set_yticks([])
    
    plt.savefig('traditional.png')
    return axs
    
    
def train_mlp(images, label_images, names, epochs=50, random_state=None):
    xs, ys = make_pixel_dataset(images, label_images)
    xs = torch.tensor(xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.int64)
    pixel_dataset = TensorDataset(xs, ys)
    
    pixel_dataset_train, pixel_dataset_val, pixel_dataset_test = split_dataset(
        pixel_dataset, train_size=0.7, test_size=0.15, seed=random_state
    )
    pixel_loader_train = DataLoader(pixel_dataset_train, batch_size=512, num_workers=2, shuffle=True)
    pixel_loader_val = DataLoader(pixel_dataset_val, batch_size=512, num_workers=2)
    pixel_loader_test = DataLoader(pixel_dataset_test, batch_size=512, num_workers=2)
    
    mlp = LitMLP(176, 14, lr=1e-3, weight_decay=1e-5)
    
    callbacks = [
        # EarlyStopping(monitor='accuracy/val', mode='max', patience=10),
        ModelCheckpoint(monitor='accuracy/val', mode='max', save_last=True),
    ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir='mlp_results'
    )
    trainer.fit(mlp, pixel_loader_train, pixel_loader_val)
    evaluate_model(trainer, pixel_loader_test, names)
    
    
def train_cnn(loader_train, loader_val, loader_test, names, epochs=50):
    # WARNING! YOU WILL GET GOOD RESULTS BECAUSE THE IMAGES IN TRAIN AND TEST OVERLAP!
    # The labels don't, but because the labels are locally continuous, we have information leakage via the images!
    cnn = LitCNN(176, 14, lr=1e-3)
    callbacks = [
        # EarlyStopping(monitor='accuracy/val', mode='max', patience=10),
        ModelCheckpoint(monitor='accuracy/val', mode='max', save_last=True)
    ]
    trainer = pl.Trainer(
        accelerator='gpu', 
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir='cnn_results'
    )
    trainer.fit(cnn, train_dataloaders=loader_train, val_dataloaders=loader_val)

    evaluate_model(trainer, loader_test, names)


def train_resnet(loader_train, loader_val, loader_test, names, freeze_head=False, epochs=50):
    resnet = TransferResNet(14, freeze_head=freeze_head, lr=1e-4)
    
    callbacks = [
        # EarlyStopping(monitor='accuracy/val', mode='max', patience=10),
        ModelCheckpoint(monitor='accuracy/val', mode='max', save_last=True),
    ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator='gpu',
        callbacks=callbacks,
        default_root_dir='resnet_results'
    )
    trainer.fit(resnet, loader_train, loader_val)
    evaluate_model(trainer, loader_test, names)
     
    
def train_unet(loader_train, loader_val, loader_test, names, epochs):
    model = LitUNet(176, 15, ignore_index=0, lr=1e-4)
    callbacks = [
        # Too much variance due sample size variability due to label exclusion
        # EarlyStopping(monitor='accuracy/val', mode='max', patience=10),
        ModelCheckpoint(monitor='accuracy/val', mode='max', save_last=False),
    ]
    trainer = pl.Trainer(
        accelerator='gpu', 
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir='unet_results'
    )
    trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)

    evaluate_model(trainer, loader_test, names, ignore_index=0)


def train_unet_overlap(
    crop_size, stride, images, label_images, batch_size, epochs
):
    dataset = ConcatDataset([
        CroppedDataset(image, label_image, stride=stride, crop_size=crop_size)
        for image, label_image in zip(images, label_images)
    ])

    # Print max bounds and number of images
    for ds in dataset.datasets:
        print(ds.img_h, ds.img_w, ds.get_bounds(len(ds) - 1)[-2:])
        print(len(ds))

    dataset = AugmentedDataset(
        dataset, transform=flip_and_rotate, apply_on_target=True
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LitUNet(176, 15, ignore_index=0, lr=1e-4)
    callbacks = [
        ModelCheckpoint(monitor='loss/train', mode='min', save_last=False),
    ]
    trainer = pl.Trainer(
        accelerator='gpu',
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir='unet_results_overlap'
    )
    trainer.fit(model, train_dataloaders=loader)
    
    return trainer

    
##########################################################
# IMAGE PREDICTION FUNCTIONS
##########################################################

def predict_image_unet(src_path, dst_path, size, model):
    # Load from disk
    image = load_image(src_path)
    dataset = CroppedDataset(image, None, size, size)
    loader = DataLoader(dataset)
    
    # Write to disk
    with rasterio.open(
        dst_path, 'w',
        height=dataset.image.shape[-2],
        width=dataset.image.shape[-1],
        count=1,
        dtype=rasterio.uint8,
        driver='Gtiff',
    ) as dst:
        for idx, x in enumerate(loader):
            logits = model(x)
            preds = torch.argmax(logits, 1)
            preds = preds.numpy().astype(rasterio.uint8)
            min_i, min_j, max_i, max_j = dataset.get_bounds(idx)
            window = rasterio.windows.Window.from_slices(
                (min_i, max_i), (min_j, max_j)
            )
            dst.write(preds, window=window)


def predict_all_images_unet(
    paths, sizes, model,
    cmap, classnames, rgb, suffix='_unet'
):
    
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    for src_path, size in zip(paths, sizes):
        src_path = Path(src_path)
        tiff_dst_path = output_dir / Path(src_path.stem + f'_pred{suffix}.tif')
        png_dst_path = output_dir / Path(src_path.stem + f'_pred{suffix}.png')
        
        predict_image_unet(src_path, tiff_dst_path, size, model)
        
        image = load_image(src_path)
        label = load_label(tiff_dst_path)
        display_image_and_label(
            image=image, label=label, cmap=cmap, classnames=classnames,
            rgb=rgb, title=src_path.stem + suffix
        )
        
        plt.savefig(png_dst_path)
        

def predict_image_cnn(src_path, dst_path, model, batch_size, patch_size=15):
    # Load from disk
    image = load_image(src_path)
    dataset = PatchDatasetPostPad(image, None, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Predict
    trainer = pl.Trainer(accelerator='gpu')
    preds = trainer.predict(model, loader)
    preds = torch.cat(preds)
    preds = preds.reshape(dataset.image.shape[-2:])
    
    # Write to disk
    with rasterio.open(
        dst_path, 'w',
        height=dataset.image.shape[-2],
        width=dataset.image.shape[-1],
        count=1,
        dtype=rasterio.uint8,
        driver='Gtiff',
    ) as dst:
        dst.write(preds.numpy().astype(rasterio.uint8)[np.newaxis, ...])
        
        
def predict_all_images_cnn(
    paths, model, batch_size,
    cmap, classnames, rgb, suffix='_cnn'
):
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    
    for src_path in paths:
        src_path = Path(src_path)
        tiff_dst_path = output_dir / Path(src_path.stem + f'_pred{suffix}.tif')
        png_dst_path = output_dir / Path(src_path.stem + f'_pred{suffix}.png')
        
        predict_image_cnn(src_path, tiff_dst_path, model, batch_size)
        
        image = load_image(src_path)
        label = load_label(tiff_dst_path)
        display_image_and_label(
            image=image, label=label, cmap=cmap, classnames=classnames,
            rgb=rgb, title=src_path.stem + suffix
        )
        
        plt.savefig(png_dst_path)

# %%
# if __name__ == '__main__':
random_state = 42
rgb = (23, 11, 7)
info_path = 'hyrank_info.csv'
train_x_paths = (
    'HyRANK_satellite/TrainingSet/Dioni.tif',
    'HyRANK_satellite/TrainingSet/Loukia.tif',
)
train_y_paths = (
    'HyRANK_satellite/TrainingSet/Dioni_GT.tif',
    'HyRANK_satellite/TrainingSet/Loukia_GT.tif'
)
validation_x_paths = (
    'HyRANK_satellite/ValidationSet/Erato.tif',
    'HyRANK_satellite/ValidationSet/Kirki.tif',
    'HyRANK_satellite/ValidationSet/Nefeli.tif',
)

# %%
# LOAD THE IMAGES AND THEIR INFO
images, label_images = read_data(train_x_paths, train_y_paths)
info = read_info(info_path)
names = info['name']
colors = info['color']
cmap = mpl.colors.ListedColormap(info['color'])

# Display the training data
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)
for image, label, path in zip(images, label_images, train_x_paths):
    path = Path(path)
    display_image_and_label(image, label, cmap, names, rgb, path.stem)
    plt.savefig(output_dir / path.with_suffix('.png').name)

# %%
# CREATE DATASETS AND DATALOADERS

# Patch Dataset All Channels
patch_dataset = ConcatDataset([
    PatchDatasetPostPad(image, label_image, patch_size=15)
    for image, label_image in zip(images, label_images)
])
patch_dataset_train, patch_dataset_val, patch_dataset_test = split_dataset(
    patch_dataset, train_size=0.7, test_size=0.15, seed=random_state
)
patch_dataset_train_aug = AugmentedDataset(patch_dataset_train, flip_and_rotate)

patch_loader_train = DataLoader(patch_dataset_train, batch_size=64, shuffle=True, num_workers=2)
patch_loader_train_aug = DataLoader(patch_dataset_train_aug, batch_size=64, shuffle=True, num_workers=2)
patch_loader_val = DataLoader(patch_dataset_val, batch_size=64, num_workers=2)
patch_loader_test = DataLoader(patch_dataset_test, batch_size=64, num_workers=2)

fig, axs = plt.subplots(nrows=5, figsize=(4, 5))
for ax in axs.flat:
    display_patch(patch_dataset_train_aug, 200, rgb=rgb, ax=ax)
    ax.set_axis_off()
plt.savefig(output_dir / 'display_patch.png')

# Patch Dataset RGB Only   
rgb_dataset = ConcatDataset([
    PatchDatasetPostPad(image, label_image, patch_size=15, channels=rgb)
    for image, label_image in zip(images, label_images)
])
rgb_dataset_train, rgb_dataset_val, rgb_dataset_test = split_dataset(
    rgb_dataset, train_size=0.7, test_size=0.15, seed=random_state
)
rgb_dataset_train_aug = AugmentedDataset(rgb_dataset_train, flip_and_rotate)

rgb_loader_train = DataLoader(rgb_dataset_train, batch_size=256, shuffle=True, num_workers=2)
rgb_loader_train_aug = DataLoader(rgb_dataset_train_aug, batch_size=256, shuffle=True, num_workers=2)
rgb_loader_val = DataLoader(rgb_dataset_val, batch_size=256, num_workers=2)
rgb_loader_test = DataLoader(rgb_dataset_test, batch_size=256, num_workers=2)


# Cropped Dataset
# 62*4 = 248 < 249 and 250, the heights of the images
stride = 62
crop_size = 62
cropped_dataset = ConcatDataset([
    CroppedDataset(image, label_image, crop_size=crop_size, stride=stride)  
    for image, label_image in zip(images, label_images)
])
cropped_dataset_train, cropped_dataset_val, cropped_dataset_test = split_dataset(
    cropped_dataset, train_size=0.7, test_size=0.15, seed=random_state
)
cropped_dataset_train_aug = AugmentedDataset(
    cropped_dataset_train, transform=flip_and_rotate, apply_on_target=True
)

# Use batch_size=2 when running locally
cropped_loader_train_aug = DataLoader(cropped_dataset_train_aug, batch_size=32, shuffle=True, num_workers=2)
cropped_loader_val = DataLoader(cropped_dataset_val, batch_size=32, num_workers=2)
cropped_loader_test = DataLoader(cropped_dataset_test, batch_size=32, num_workers=2)

display_segmentation(cropped_dataset_train_aug, 80, cmap=cmap, names=names, rgb=rgb, n_repeats=5)

    
print('\nTraining Random Forest and SVM...')
train_traditional(images, label_images, names=names[1:], random_state=random_state)
print('\nTraining MLP...')
train_mlp(images, label_images, names=names[1:], random_state=random_state, epochs=200)
print('\nTraining CNN...')
train_cnn(patch_loader_train_aug, patch_loader_val, patch_loader_test, names=names[1:], epochs=200)
print('\nTraining ResNet...')
train_resnet(rgb_loader_train_aug, rgb_loader_val, rgb_loader_test, names=names[1:], freeze_head=False, epochs=200)
print('\nTraining U-Net...')
train_unet(cropped_loader_train_aug, cropped_loader_val, cropped_loader_test, names=names, epochs=100)
print('\nFinished!')

# # Train U-Net with overlap on the entire dataset
# # to use it for the final estimation.
# train_unet_overlap(
#     crop_size=62, stride=15, images=images,
#     label_images=label_images, epochs=300, batch_size=32
# )

ckpt_path_cnn = (
    'cnn_results/lightning_logs/version_0/checkpoints/epoch=1-step=5868.ckpt'
)
model = LitCNN.load_from_checkpoint(ckpt_path_cnn)
 
predict_all_images_cnn(
    paths=validation_x_paths, model=model, batch_size=64,
    cmap=cmap, classnames=names, rgb=rgb,
)

# Evaluate on the unlabelled dataset
ckpt_path_unet = (
    'unet_results_overlap/lightning_logs/'
    'version_0/checkpoints/epoch=280-step=16860.ckpt'
)
model = LitUNet.load_from_checkpoint(ckpt_path_unet)
predict_all_images_unet(
    paths=validation_x_paths, sizes=(60, 60, 62),
    model=model, cmap=cmap, classnames=names, rgb=rgb
)

##########################################################
# COMMENTARY
##########################################################

# TRANSFORMS WILL GIVE YOU "WORSE" RESULTS IN THIS DATASET!
# If for example there's forest at the top of the (entire) image,
# and below the forest, it's all urban,
# then rotating a patch of it will produce an image that doesn't exist in the dataset
# and it would not help us predict other patches from the Same image.
# It's all because the train and the test set are very dependent on one another

# Using sliding window with stride equal to window size
# so that no part from the training images exist in the validation images.
# If overlap is allowed both train and validation accuracy reach ~100%,
# but without overlap, validation accuracy drops to about 80%!
# For example consider a test image with a city in its center,
# and that city appearing in the corners of some training images.

# Patch wise models do well because of overlap. Two neighboring pixels
# produce essentially indentical patches, while they are also very likely
# to have the same labels die to local "continuity" of the image.
# If one of these two pixel is in the train set and the other is in the test set, then we have huge overlap.

# Element wise models also perform really well because of local image "continuity".

# Of all the models, only the segmentation model with no overlap
# is the one that it test somewhat realistically.

