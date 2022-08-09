from pathlib import Path
import re
import json

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay, classification_report

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from torchvision.io import read_video
from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models.feature_extraction import create_feature_extractor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchmetrics import Accuracy


# _________________________________ DATASETS _________________________________

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
        
        # Move the transform to gpu, if it is a Module.
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
        N, T, C, H, W = vid.shape
        vid = vid.view(-1, C, H, W)

        vid = self.preprocessor(vid)
        vid = self.extractor(vid)['flatten']

        vid = vid.view(N, T, -1)
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
        for _ in dataset:
            pass


def plot_transformed(video_path, nframes=10):
    vid, *_ = read_video(video_path, output_format='TCHW', pts_unit='sec')
    preprocessor = ResNet18_Weights.IMAGENET1K_V1.transforms()
    preprocessed = preprocessor(vid)
    
    vid = np.transpose(vid, (0, 2, 3, 1))
    preprocessed = np.transpose(preprocessed, (0, 2, 3, 1))
    
    fig, axs = plt.subplots(nrows=nframes, ncols=2, figsize=(8, 16))
    fig.set_tight_layout(True)
    for i in range(nframes):
        axs[i, 0].imshow(vid[i])
        axs[i, 1].imshow(preprocessed[i])
        axs[i, 0].set_axis_off()
        axs[i, 1].set_axis_off()


def test_val_split(dataset_test_original, n_groups_test):
    """Splits test UCF101 dataset into test and validation datasets.
    
    Every class in the test UCF101 directory has 7 video groups,
    and every video group has 7 videos,
    which are cuts of a single original video.
    The first `n_groups_test` video groups of each class will comprise our test set,
    and the rest `7 - n_groups` video groups will comprise our validation set.
    """
    p = re.compile(r'v_[a-zA-Z]*_g(\d+)_')
    test_indices, val_indices = [], []
    for i, s in enumerate(dataset_test_original.filenames):
        video_group_str = p.match(s).group(1)
        video_group = int(video_group_str)
        if video_group <= n_groups_test:
            test_indices.append(i)
        else:
            val_indices.append(i)

    dataset_test = Subset(dataset_test_original, test_indices)
    dataset_val = Subset(dataset_test_original, val_indices)
    
    return dataset_test, dataset_val


def rnn_collate_fn(batch):
    seqs, y = map(list, zip(*batch))
    padded = pad_sequence(seqs, batch_first=True)
    y = torch.stack(y)
    return padded, y


def transformer_collate_fn(batch):
    seqs, y = map(list, zip(*batch))
    
    src = pad_sequence(seqs, batch_first=True)
    
    y = torch.stack(y)
    
    lengths = list(map(len, seqs))
    N = len(lengths)
    T = max(lengths)
    src_key_padding_mask = torch.full((N, T), False, dtype=torch.bool)
    for i in range(N):
        src_key_padding_mask[i, lengths[i]:] = True
    
    return src, src_key_padding_mask, y


# __________________________________ MODELS __________________________________

class LightningClassifier(pl.LightningModule):
    
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.train_accuracy = Accuracy(num_classes=num_classes)
        self.val_accuracy = Accuracy(num_classes=num_classes)
        self.test_accuracy = Accuracy(num_classes=num_classes)
    
    def _get_logits_and_target(self, batch):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        logits, y = self._get_logits_and_target(batch)
        loss = F.cross_entropy(logits, y)
        self.train_accuracy(logits, y)
        self.log_dict({'loss/train': loss, 'accuracy/train': self.train_accuracy})
        return loss
    
    def validation_step(self, batch, batch_idx):
        logits, y = self._get_logits_and_target(batch)
        loss = F.cross_entropy(logits, y)
        self.val_accuracy(logits, y)
        self.log_dict({'loss/val': loss, 'accuracy/val': self.val_accuracy})
        
    def test_step(self, batch, batch_idx):
        logits, y = self._get_logits_and_target(batch)
        loss = F.cross_entropy(logits, y)
        self.test_accuracy(logits, y)
        self.log_dict({'loss/test': loss, 'accuracy/test': self.test_accuracy})
        
    def predict_step(self, batch, batch_idx):
        logits, _ = self._get_logits_and_target(batch)
        pred = torch.argmax(logits, dim=1)
        return pred
    
    
class GRUClassifier(LightningClassifier):
    
    def __init__(
        self, input_size, rnn_size, num_classes,
        num_rnn_layers=2, dropout=0.1, lr=1e-3,
    ):
        super().__init__(num_classes)
        self.save_hyperparameters()
        
        self.dropout = dropout
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=rnn_size,
            num_layers=num_rnn_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.linear = nn.Linear(rnn_size, self.num_classes)
        self.lr = lr
        
    def forward(self, x):
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = F.dropout(x, self.dropout)
        x = self.linear(x)
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def _get_logits_and_target(self, batch):
        x, y = batch
        logits = self(x)
        return logits, y
    

class PositionalEncoder(nn.Module):
    
    def __init__(self, d_model: int, max_len):
        super().__init__()
        
        pos = torch.arange(max_len).unsqueeze(1)
        exponents = torch.arange(0, d_model, 2) / (-d_model)
        angle_rates = torch.pow(10_000.0, exponents)
        angles = pos * angle_rates
        encoding = torch.empty(1, max_len, d_model)
        encoding[0, :, 0::2] = torch.sin(angles)
        encoding[0, :, 1::2] = torch.cos(angles)
        
        self.register_buffer('encoding', encoding)
        
    def forward(self, x):
        x = x + self.encoding[:, :x.shape[1], :]
        return x


class TransformerClassifier(LightningClassifier):
    
    def __init__(
        self, d_model, nhead, dim_feedforward, num_layers, num_classes,
        dropout=0.1, max_len=1_000, lr=1e-3,
    ):
        super().__init__(num_classes)
        self.save_hyperparameters()
        
        self.dropout = dropout
        self.positional_encoder = PositionalEncoder(d_model, max_len=max_len)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout=dropout, batch_first=True
            ),
            num_layers=num_layers,
        )
        self.linear = nn.Linear(d_model, self.num_classes)
        self.lr = lr
        
    def forward(self, src, mask=None, src_key_padding_mask=None):
        # src.shape = B, T, D
        x = self.positional_encoder(src)
        x = F.dropout(x, p=self.dropout)
        x = self.transformer_encoder(x, mask=mask, src_key_padding_mask=src_key_padding_mask) 
        x = torch.amax(x, 1) # B, D
        x = F.dropout(x, p=self.dropout)
        x = self.linear(x) # B, C
        return x
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    def _get_logits_and_target(self, batch):
        x, src_key_padding_mask, y = batch
        logits = self(x, src_key_padding_mask=src_key_padding_mask)
        return logits, y


# _________________________________ TRAINING _________________________________

def train_evaluate(trainer, model, loader_train, loader_val, loader_test, class_names, output_dir):
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)
    
    best_model = trainer.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(best_model, loader_test)

    y_true = torch.cat([batch[-1] for batch in loader_test])  # Assuming that target == batch[-1] 
    y_pred = torch.cat(trainer.predict(best_model, loader_test))

    d = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    with open(output_dir/'classification_report.json', 'w') as f:
        json.dump(d, f, indent=4)
    
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=class_names, xticks_rotation=90, ax=ax
    )
    ax.set_title(best_model.__class__.__name__)
    fig.savefig(output_dir/'confusion_matrix.png', facecolor='white', bbox_inches='tight')
    
    
def train_evaluate_rnn(
    dataset_train, dataset_val, dataset_test,
    max_epochs=300, patience=50, lr=1e-3, output_dir='output/rnn'
):
    loader_train_rnn = DataLoader(
        dataset_train, shuffle=True, collate_fn=rnn_collate_fn,
        batch_size=32, num_workers=2,
    )
    loader_val_rnn = DataLoader(
        dataset_val, shuffle=False, collate_fn=rnn_collate_fn,
        batch_size=32, num_workers=2,
    )
    loader_test_rnn = DataLoader(
        dataset_test, shuffle=False, collate_fn=rnn_collate_fn,
        batch_size=32, num_workers=2,
    )

    rnn = GRUClassifier(
        input_size=512,
        rnn_size=512,
        num_classes=5,
        num_rnn_layers=2,
        lr=lr,
    )
    callbacks = [
        EarlyStopping(monitor='loss/val', mode='min', patience=patience),
        ModelCheckpoint(monitor='loss/val', mode='min', save_last=True)
    ]
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        min_epochs=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        default_root_dir=output_dir,
    )
    
    train_evaluate(
        trainer=trainer, model=rnn, loader_train=loader_train_rnn, loader_val=loader_val_rnn,
        loader_test=loader_test_rnn, class_names=dataset_train.class_names, output_dir=output_dir,
    )


def train_evaluate_transformer(
    dataset_train, dataset_val, dataset_test,
    max_epochs=300, patience=50, lr=1e-3, output_dir='output/transformer'
):
    loader_train_transformer = DataLoader(
        dataset_train, shuffle=True, collate_fn=transformer_collate_fn,
        batch_size=32, num_workers=2,
    )
    loader_val_transformer = DataLoader(
        dataset_val, shuffle=False, collate_fn=transformer_collate_fn,
        batch_size=32, num_workers=2,
    )
    loader_test_transformer = DataLoader(
        dataset_test, shuffle=False, collate_fn=transformer_collate_fn,
        batch_size=32, num_workers=2,
    )

    transformer = TransformerClassifier(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_layers=2,
        num_classes=5,
        lr=lr,
    )
    
    callbacks = [
        EarlyStopping(monitor='loss/val', mode='min', patience=patience),
        ModelCheckpoint(monitor='loss/val', mode='min', save_last=True)
    ]
    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        min_epochs=1,
        max_epochs=max_epochs,
        callbacks=callbacks,
        log_every_n_steps=10,
        default_root_dir=output_dir,
    )
    
    train_evaluate(
        trainer=trainer, model=transformer, loader_train=loader_train_transformer, loader_val=loader_val_transformer,
        loader_test=loader_test_transformer, class_names=dataset_train.class_names, output_dir=output_dir,
    )
    

def main():
    pl.utilities.seed.seed_everything(42)
    data_root_dir = 'data'
    
    cache_all(data_root_dir)
    transform = None  # Hack to save some memory. Requires cache_all(root_dir),

    dataset_train = Ucf101(
        root_dir=data_root_dir, training=True, transform=transform,
        cache_fetched=True, cache_exist_ok=True,
    )  # 594 samples
    dataset_test_original = Ucf101(
        root_dir=data_root_dir, training=False, transform=transform,
        cache_fetched=True, cache_exist_ok=True,
    )  # 224 samples
    dataset_test, dataset_val = test_val_split(dataset_test_original, 4)  # 121, 103 samples
    # max(x.shape[0] for x, _ in itertools.chain(dataset_train, dataset_test_original)) == 528

    train_evaluate_rnn(dataset_train, dataset_val, dataset_test)
    train_evaluate_transformer(dataset_train, dataset_val, dataset_test)
    
    
if __name__ == '__main__':
    main()
