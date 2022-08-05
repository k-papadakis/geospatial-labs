# %%
from typing import Callable, List
import os
import pickle

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_video
from torchvision.models import resnet18
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import ConvertImageDtype, Resize, Normalize

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy


VTransform = Callable[[torch.Tensor], torch.Tensor]


class UCF101(Dataset):
    def __init__(self, data_root, mode="train", video_transforms: List[VTransform] = [], use_precomputed=True):
        '''
        Return a UCF101 Dataset instance
        '''
        super().__init__()
        assert mode in ["train", "test"]
        
        self.root = data_root
        self.mode = mode
        self.v_transforms = video_transforms
        
        # Build database of samples
        self._build_db()
        
        # Features precomute functionality
        self.pre = use_precomputed
        self.pre_root = os.path.join(self.root, "precomp")
        if self.pre and not os.path.exists(self.pre_root):
            os.makedirs(self.pre_root)
        
    def _build_db(self):
        '''
        Parse train/test csv containing paths to videos and corresponding labels.
        Also, assign a unique index to each category
        '''
        csv_file = os.path.join(self.root, self.mode + ".csv")
        self.db: np.ndarray = pd.read_csv(csv_file, header=0).values
        
        unique_categories = np.sort(np.unique(self.db.T[1]))
        self.categories = {c_name: c_idx for c_idx, c_name in enumerate(unique_categories)}
        
    def compute_sample(self, video_name, category):
        '''
        For a specific video, read data into memory, permute data to NumFrames x Channels x Height x Width format.
        Also, transform data according to list of transforms
        '''
        
        # Load video
        V, *_ = read_video(os.path.join(self.root, self.mode, video_name))
        # Permute data to NxCxHxW from NxHxWxC
        V = V.permute(0,3,1,2)
        
        for T in self.v_transforms:
            V = T(V)
        
        return V, self.categories[category]
        
    def __getitem__(self, index):
        '''
        Retrieve a specific sample from the dataset
        '''
        video_name, category = self.db[index]
        
        hval = "_".join([
            self.mode,
            video_name
        ])
        
        if os.path.exists(os.path.join(self.pre_root, f"{hval}.tmp")):
            with open(os.path.join(self.pre_root, f"{hval}.tmp"), "rb") as f:
                sample =  pickle.load(f)
        else:
            sample = self.compute_sample(video_name, category)
            # Save tmp
            with open(os.path.join(self.pre_root, f"{hval}.tmp"), "wb") as f:
                pickle.dump(sample, f)
            
        return sample
            
    def __len__(self):
        '''
        Returns the number of samples in the dataset
        '''
        return self.db.shape[0]


class Seq2Vec(pl.LightningModule):
    def __init__(self, features_in, num_classes, learning_rate=1e-3):
        '''
        Returns a Seq2Vec RNN model
        '''
        super().__init__()
        
        self.rnn_encoder = nn.GRU(
            input_size=features_in,
            hidden_size=32,
            num_layers=2,
            batch_first=False,
            dropout=0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)
        )
        
        self.lr = learning_rate

        self.val_accuracy = Accuracy()        
        self.save_hyperparameters()
        
    def forward(self, x):
        '''
        Forward-pass
        '''
        rnn_out, h_n = self.rnn_encoder(x)
        #  rnn_out: L, B, 32
        return self.classifier(rnn_out[-1])

    def training_step(self, batch, batch_idx):
        '''
        Training logic
        '''
        X, y = batch

        logits = self(X)

        loss = F.nll_loss(torch.log_softmax(logits, dim=-1), y)
        self.log("loss/train", loss, on_epoch=True, on_step=False, batch_size=X.size()[1])

        return loss
    
    def validation_step(self, batch, batch_idx):
        '''
        Validation logic
        '''
        X, y = batch

        logits = self(X)

        loss = F.nll_loss(torch.log_softmax(logits, dim=-1), y)
        self.log("loss/val", loss, on_epoch=True, on_step=False, batch_size=X.size()[1])

        self.val_accuracy(logits, y)
        self.log("accuracy/val", self.val_accuracy, on_epoch=True, on_step=False, batch_size=X.size()[1])
    
    def configure_optimizers(self):
        '''
        Setup Adam optimizer
        '''
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
def compute_features() -> VTransform:
    '''
    Returns a VTransform object that uses a pretrained CNN to extract features
    '''
    # Instantiate a CNN for feature extraction
    encoder = resnet18(pretrained=True, progress=False)
    # model = nn.Sequential(*list(encoder.children())[:-1], nn.Flatten())
    model = create_feature_extractor(encoder, ["avgpool"])
    model.eval()
    
    def apply(v: torch.Tensor) -> torch.Tensor:    
        # return model(v)
        with torch.no_grad():
            feats = torch.flatten(model(v)["avgpool"], 1)
        return feats
    
    return apply


def pad_sequences_collate_fn(samples: List[tuple]) -> tuple:
    '''
    Zero-pad (in front) each sample to enable batching. The longest sequence defines the sequence length for the batch
    '''
    
    labels = torch.stack([torch.tensor(v[1]) for v in samples])
    data = pad_sequence([v[0] for v in samples])
    
    return data, labels
    

# %%
DATA_ROOT = "data/"

train_dset = UCF101(
    DATA_ROOT,
    "train",
    video_transforms=[
        ConvertImageDtype(torch.float32),
        Resize((224, 224)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        compute_features(),
    ]
)

val_dset = UCF101(
    DATA_ROOT,
    "test",
    video_transforms=[
        ConvertImageDtype(torch.float32),
        Resize((224, 224)),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        compute_features(),
    ]
)

train_dloader = DataLoader(train_dset, batch_size=16, collate_fn=pad_sequences_collate_fn, shuffle=True, num_workers=4)
val_dloader = DataLoader(val_dset, batch_size=16, collate_fn=pad_sequences_collate_fn, shuffle=False, num_workers=4)

callbacks = [
    EarlyStopping(monitor="accuracy/val", mode="max", patience=50),
    ModelCheckpoint(monitor="accuracy/val", mode="max", save_last=True)
]

model = Seq2Vec(512, len(train_dset.categories), learning_rate=1e-4)
trainer = pl.Trainer(
    accelerator="gpu", 
    devices=1,
    min_epochs=300,
    max_epochs=1000,
    callbacks=callbacks,
    default_root_dir="seq2vec_gru"
)

trainer.fit(model, train_dataloaders=train_dloader, val_dataloaders=val_dloader)

# # %%
# for x, y in train_dset:
#     pass
# for x, y in val_dset:
#     pass