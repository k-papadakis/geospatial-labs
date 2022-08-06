# %%
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_video
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy 

# %%
class Ucf101(Dataset):
    
    def __init__(
        self, root_dir, training: bool, transform=None,
        cache_fetched=False, cache_exist_ok=False, cuda=False,
    ):
        # If isinstance(transform, nn.Module) and cuda=True,
        #  then multiprocessing will fail (e.g. DataLoader).
        # Generally, avoid using a DataLoader if not everything is cached,
        #  because workers might crash for unknown reasons.
        # Instead, make a call to `cache_all` first,
        #  and use the dataset with transform=None.
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
        self.class_names, ys = np.unique(csv_['tag'], return_inverse=True)
        self.ys = torch.as_tensor(ys)
        
        # Move the transform to gpu, if it's Module.
        if isinstance(self.transform, nn.Module):
            self.device = torch.device('cuda:0' if cuda else 'cpu')
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
        
        x = torch.as_tensor(x)
        return x, y
    
    def __len__(self):
        return len(self.filenames)
    
    def read_transform_video(self, file_path):
        x, *_ = read_video(str(file_path), output_format='TCHW', pts_unit='sec')
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
    
    @torch.inference_mode()
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
    
    
def cache_all(source_root_dir):
    """Call this first to avoid errors in data loading"""
    cuda = torch.cuda.is_available()
    transform = ResNet50ExtractorVideo()
    dataset_train = Ucf101(
        root_dir=source_root_dir, training=True, transform=transform,
        cache_fetched=True, cache_exist_ok=True, cuda=cuda
    )
    dataset_test = Ucf101(
        root_dir=source_root_dir, training=False, transform=transform,
        cache_fetched=True, cache_exist_ok=True, cuda=cuda
    )

    for dataset in dataset_train, dataset_test:
        for x, y in dataset:
            pass


def plot_transformed(video_path):
    vid, *_ = read_video(video_path, output_format='TCHW', pts_unit='sec')
    preprocessor = ResNet18_Weights.IMAGENET1K_V1.transforms()
    preprocessed = preprocessor(vid)
    
    vid = np.transpose(vid, (0, 2, 3, 1))
    preprocessed = np.transpose(preprocessed, (0, 2, 3, 1))
    
    fig, axs = plt.subplots(nrows=10, ncols=2, figsize=(8,16))
    fig.set_tight_layout(True)
    for i in range(10):
        axs[i, 0].imshow(vid[i])
        axs[i, 1].imshow(preprocessed[i])
        axs[i, 0].set_axis_off()
        axs[i, 1].set_axis_off()
        
        
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


def rnn_collate_fn(batch):
    seqs, y = map(list, zip(*batch))
    padded = pad_sequence(seqs, batch_first=True)
    y = torch.stack(y)
    return padded, y


# %%
class GRUClassifier(pl.LightningModule):
    
    def __init__(
        self, input_size, rnn_size, dense_size, num_classes,
        num_rnn_layers=2, rnn_dropout=0.2, lr=1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=rnn_size,
            num_layers=num_rnn_layers,
            dropout=rnn_dropout,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(rnn_size, dense_size),
            nn.ReLU(),
            nn.Linear(dense_size, num_classes),
        )
        self.lr = lr
        self.train_accuracy = Accuracy(num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)
        self.test_accuracy = Accuracy(num_classes=num_classes)
        
    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.classifier(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.train_accuracy(logits, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': self.train_accuracy})
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.val_accuracy(logits, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': self.val_accuracy})
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.test_accuracy(logits, y)
        self.log_dict({'loss/test': loss, 'accuracy/test': self.test_accuracy})
        
    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        pred = torch.argmax(logits, dim=1)
        return pred
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


seed = 7
root_dir = 'data'
# cache_all(root_dir)
transform = None  # Hack to save some memory

dataset_train = Ucf101(
    root_dir=root_dir, training=True, transform=transform,
    cache_fetched=True, cache_exist_ok=True,
)
dataset_test_original = Ucf101(
    root_dir=root_dir, training=False, transform=transform,
    cache_fetched=True, cache_exist_ok=True,
)
dataset_val, dataset_test, _ = split_dataset(dataset_test_original, 0.5, seed=seed)

loader_train_rnn = DataLoader(
    dataset_train, shuffle=True, collate_fn=rnn_collate_fn,
    batch_size=16, num_workers=2,
)
loader_val_rnn = DataLoader(
    dataset_val, shuffle=False, collate_fn=rnn_collate_fn,
    batch_size=16, num_workers=2,
)
loader_test_rnn = DataLoader(
    dataset_test, shuffle=False, collate_fn=rnn_collate_fn,
    batch_size=16, num_workers=2,
)

rnn = GRUClassifier(
    input_size=512,
    rnn_size=32,
    dense_size=16,
    num_classes=5,
    num_rnn_layers=2,
    lr=1e-3,
)
callbacks = [
    EarlyStopping(monitor='loss/val', mode='min', patience=20),
    ModelCheckpoint(monitor='loss/val', mode='min', save_last=True)
]
trainer = pl.Trainer(
    accelerator='gpu',
    min_epochs=1,
    max_epochs=100,
    callbacks=callbacks,
    default_root_dir='rnn_results'
)

# %%
trainer.fit(rnn, train_dataloaders=loader_train_rnn, val_dataloaders=loader_val_rnn)

# %%
best_rnn = trainer.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)

trainer.test(best_rnn, loader_test_rnn)

# %%
class_names = dataset_train.class_names
y_true = torch.cat([y for _, y in loader_test_rnn])
y_pred = torch.cat(trainer.predict(best_rnn, loader_test_rnn))
print(classification_report(y_true, y_pred, target_names=class_names))
ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=class_names, xticks_rotation='vertical')
plt.show()

# %%