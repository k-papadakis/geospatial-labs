# %%
from pathlib import Path

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_video
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.transforms import Compose, ConvertImageDtype, Resize, Normalize

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy

# %%
class UCF101(Dataset):
    
    def __init__(
        self, root_dir, training: bool, transform=None,
        cache_fetched=False, cache_exist_ok=False,
    ):
        self.root_dir = Path(root_dir)
        self.training = training
        self.mode = 'train' if self.training else 'test'
        self.transform = transform
        self.cache_fetched = cache_fetched
        
        # Create a directory to cache fetched videos, if appropriate
        if self.cache_fetched:
            self.cache_dir = self.root_dir / f'{self.__class__.__name__}_cache'
            (self.cache_dir / self.mode).mkdir(parents=True, exist_ok=cache_exist_ok)
        else:
            self.cache_dir = None
        
        # Load the video filenames, and their respective labels
        csv_path = self.root_dir/ f'{self.mode}.csv'
        csv_ = np.genfromtxt(csv_path, delimiter=',', names=True, dtype=None, encoding=None)
        self.filenames = csv_['video_name']
        self.class_names, self.ys = np.unique(csv_['tag'], return_inverse=True)
        
        if isinstance(self.transform, nn.Module):
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.transform.to(self.device)
        else:
            self.device = None
            
        
    def __getitem__(self, idx):
        file_name = Path(self.filenames[idx])
        y = self.ys[idx]
        
        file_path = self.root_dir / self.mode / file_name
        
        if self.cache_fetched:
            cache_path = self.cache_dir / self.mode / file_name.with_suffix('.npy')
            if cache_path.exists():
                x = np.load(cache_path)
            else:
                x = self.read_transform_video(file_path)
                np.save(cache_path, x)
        else:
            x = self.read_transform_video(file_path)
        
        return x, y
    
    def __len__(self):
        return len(self.filenames)
    
    def read_transform_video(self, file_path):
        x, *_ = read_video(str(file_path), output_format='TCHW')
        if self.transform is not None:
            if isinstance(self.transform, nn.Module):
                x = x.to(self.device)
            x = self.transform(x).cpu()
        return x
    
    
class ResNet50ExtractorVideo(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.weights = ResNet18_Weights.IMAGENET1K_V1
        self.preprocessor = self.weights.transforms()
        model = resnet18(weights=self.weights)
        self.extractor = create_feature_extractor(model, ['flatten'])
        self.extractor.eval()
    
    def forward(self, vid):
        need_squeeze = False
        if vid.ndim < 5:
            vid = vid.unsqueeze(dim=0)
            need_squeeze = True
        n, t, c, h, w = vid.shape
        vid = vid.view(-1, c, h, w)

        vid = self.preprocessor(vid)
        vid = self.extractor(vid)['flatten']

        vid = vid.view(n, t, -1)
        if need_squeeze:
            vid = vid.squeeze(dim=0)
        return vid
    
          
dataset = UCF101(
    root_dir='data', training=True, transform=ResNet50ExtractorVideo(),
    cache_fetched=False, cache_exist_ok=False,
)
x, y = dataset[10]
# %%
