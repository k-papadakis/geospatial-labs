import itertools

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet18
import pytorch_lightning as pl
import torchmetrics


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
         
        return loss
    
    def validation_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/val', loss, on_epoch=True, on_step=False)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy, on_epoch=True, on_step=False)
        
    def test_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        self.test_accuracy(logits, y)
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

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/val', loss, on_epoch=True, on_step=False)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy, on_epoch=True, on_step=False)
        
    def test_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        self.test_accuracy(logits, y)
        self.test_confusion_matrix(logits, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


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
    
    def __init__(self, n_channels, n_classes, ignore_index=-1, lr=1e-4):
        super().__init__()
        self.unet = UNet(n_channels, n_classes)
        self.lr = lr
        self.ignore_index = ignore_index
        
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        self.train_accuracy = torchmetrics.Accuracy(ignore_index=ignore_index, mdmc_average='global')
        self.val_accuracy = torchmetrics.Accuracy(ignore_index=ignore_index, mdmc_average='global')
        self.test_accuracy = torchmetrics.Accuracy(ignore_index=ignore_index, mdmc_average='global')
        
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
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        self.log('loss/val', loss, on_epoch=True, on_step=False)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy, on_epoch=True, on_step=False)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = self.cross_entropy(logits, y)
        self.test_accuracy(logits, y)
        
        mask = y != self.ignore_index
        self.test_confusion_matrix(
            torch.transpose(logits, 1, 0)[:, mask].T,
            y[mask]
        )


class LitTransferResNet(pl.LightningModule):
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
        
        return loss
    
    def validation_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        
        loss = F.cross_entropy(logits, y)
        self.log('loss/val', loss, on_epoch=True, on_step=False)
        
        self.val_accuracy(logits, y)
        self.log('accuracy/val', self.val_accuracy, on_epoch=True, on_step=False)
        
    def test_step(self, batch,  batch_idx):
        x, y = batch
        logits = self(x)
        self.test_accuracy(logits, y)
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