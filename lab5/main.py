# %%

from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import convert_image_dtype
from torchvision.io import read_image


class ODDataset(Dataset):
    
    def __init__(self, root_dir, mode: str):
        assert mode in {'train', 'val', 'test'}
        super().__init__()
        
        self.root_dir = Path(root_dir)
        self.mode = mode
        
        self.class_names = ['battery', 'dice', 'toycar', 'candle', 'highlighter', 'spoon']
        self._class_name_to_int = {name: i for i, name in enumerate(self.class_names)}
        self.class_name_to_int = np.vectorize(self._class_name_to_int.get, otypes=[np.int64])
        
        self.image_names = sorted(p.stem for p in (self.root_dir/self.mode/'images').iterdir())
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        name = self.image_names[idx]
        image_path = self.root_dir/self.mode/'images'/f'{name}.jpg'
        label_path = self.root_dir/self.mode/'labels'/f'{name}.txt'

        image = read_image(str(image_path))
        image = convert_image_dtype(image, torch.float32)
        
        boxes = np.loadtxt(label_path, usecols=range(4, 8), delimiter=' ', dtype=np.int64)
        boxes = torch.from_numpy(boxes) 
        
        labels = np.loadtxt(label_path, usecols=0, delimiter=' ', dtype=str, encoding=None)    
        labels = self.class_name_to_int(labels)
        labels = torch.from_numpy(labels)
        
        targets = {'boxes': boxes, 'labels': labels}
        return image, targets
        

def collate_fn(batch):
    return tuple(zip(*batch))


data_root_dir = 'data'
dataset_train = ODDataset(data_root_dir, 'train')
dataset_val = ODDataset(data_root_dir, 'val')
dataset_test = ODDataset(data_root_dir, 'test')
# dataloader_val = DataLoader(dataset_val, batch_size=8, collate_fn=collate_fn, shuffle=False)
# images, targetss = next(iter(dataloader_val))
# targetss[3]


# %%