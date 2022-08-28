# %%
import itertools
from collections import OrderedDict
from operator import itemgetter
from os import PathLike
from pathlib import Path
import re
from typing import (Any, Callable, List, Literal, Optional, Tuple, Dict, TypeVar, TypedDict)


from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.detection import (
    FasterRCNN, fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone, BackboneWithFPN
)
from torchvision.models.detection.image_list import ImageList
from torchvision.io import read_image, ImageReadMode
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# IMPORTANT: Install torchmetrics via
#  pip install -e git+https://github.com/k-papadakis/metrics.git#egg=torchmetrics
#  otherwise MeanAveragePrecision won't work.
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class EmptyDict(TypedDict):
    pass


class TargetsDict(TypedDict):
    labels: Tensor
    boxes: Tensor


class LossesDict(TypedDict):
    # RPN losses
    loss_objectness: Tensor
    loss_rpn_box_reg: Tensor
    # Detector losses
    loss_classifier: Tensor
    loss_box_reg: Tensor
    
    
class DetectionsDict(TypedDict):
    labels: Tensor
    boxes: Tensor
    scores: Tensor


class ObjectsDataset(Dataset):

    def __init__(
        self, root_dir: str | PathLike, mode: Literal['train', 'val', 'test']
    ) -> None:
        super().__init__()
        self.root_dir = Path(root_dir)
        self.mode = mode
        self.image_names = sorted(
            p.stem for p in (self.root_dir / self.mode / 'images').iterdir()
        )
        self.class_names = [
            'battery', 'dice', 'toycar', 'candle', 'highlighter', 'spoon'
        ]
        self.class_name_to_int = {
            name: i for i, name in enumerate(self.class_names)
        }

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> Tuple[Tensor, TargetsDict]:
        name = self.image_names[idx]
        image_path = self.root_dir / self.mode / 'images' / f'{name}.jpg'
        targets_path = self.root_dir / self.mode / 'labels' / f'{name}.txt'

        image = read_image(str(image_path), ImageReadMode.RGB)
        image = convert_image_dtype(image, torch.float32)

        targets_tensor = torch.from_numpy(
            np.loadtxt(
                targets_path,
                usecols=(0, *range(4, 8)),
                dtype=np.int64,
                encoding=None,
                converters={0: self.class_name_to_int.get},
                ndmin=2,
            )
        )
        targets: TargetsDict = {
            'labels': targets_tensor[:, 0],
            'boxes': targets_tensor[:, 1:],
        }

        return image, targets


def transpose(batch):
    return tuple(map(list, zip(*batch)))


def create_dataloaders(
    dataset_train: Dataset,
    dataset_val: Dataset,
    dataset_test: Dataset,
    batch_size: int,
    collate_fn: Optional[Callable] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    loader_train = DataLoader(
        dataset_train,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    loader_val = DataLoader(
        dataset_val,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    loader_test = DataLoader(
        dataset_test,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return loader_train, loader_val, loader_test


def get_resnet50_fpn_backbone() -> BackboneWithFPN:
    return resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=ResNet50_Weights.DEFAULT,
        trainable_layers=3
    )


class LitFasterRCNN(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        phase: Literal[1, 2, 3, 4] = 1
        ) -> None:
        # num_classes (int): number of output classes of the model (including the background).
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.phase = phase
        backbone = get_resnet50_fpn_backbone()
        self.model = FasterRCNN(backbone, num_classes)
        self.val_mean_ap = MeanAveragePrecision()
        self.test_mean_ap = MeanAveragePrecision()

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[TargetsDict]] = None
    ) -> List[DetectionsDict] | LossesDict:
        """Return losses if training else return detections."""
        return self.model(images, targets)

    def training_step(
        self, batch: Tuple[List[Tensor], List[TargetsDict]], batch_idx: int
    ) -> Tensor:
        images, targets = batch
        losses: LossesDict = self.model(images, targets)
        
        match self.phase:
            case 1 | 3:
                loss_sum = losses['loss_objectness'] + losses['loss_rpn_box_reg']
                self.log_dict(
                    (
                        {f'train_{k}': losses[k] for k in ('loss_objectness', 'loss_rpn_box_reg')}
                        | {'train_loss_sum': loss_sum}
                    ),
                    batch_size=len(images)
                )
                return loss_sum
            case 2 | 4:
                loss_sum = losses['loss_classifier'] + losses['loss_box_reg']
                self.log_dict(
                    (
                        {f'train_{k}': losses[k] for k in ('loss_classifier', 'loss_box_reg')}
                        | {'train_loss_sum': loss_sum}
                    ),
                    batch_size=len(images)
                )
                return loss_sum
            case _:
                raise ValueError(f'Invalid phase value {self.phase}')
        
    def configure_optimizers(self) -> Optimizer:
        model = self.model
        
        match self.phase:
            case 1:
                params = itertools.chain(
                    model.backbone.parameters(), model.rpn.parameters()
                )
            case 2:
                params = itertools.chain(
                    model.backbone.parameters(), model.roi_heads.parameters()
                )
            case 3:
                params = model.rpn.parameters()
            case 4:
                params = model.roi_heads.parameters()
            case _:
                raise ValueError(f'Invalid phase value {self.phase}')
            
        return Adam(params, lr=self.lr)
    
    def validation_step(
        self, batch: Tuple[List[Tensor], List[TargetsDict]], batch_idx: int
    ) -> None:
        images, targets = batch
        detections: DetectionsDict = self(images)
        self.val_mean_ap.update(detections, targets)  # type: ignore
        
    def test_step(
        self, batch: Tuple[List[Tensor], List[TargetsDict]], batch_idx: int
    ) -> None:
        images, targets = batch
        detections: DetectionsDict = self(images)
        self.test_mean_ap.update(detections, targets)  # type: ignore
    
    def validation_epoch_end(self, outputs):
        self.log_dict(
            {f'val_{k}': v for k, v in self.val_mean_ap.compute().items()}
        )
        self.val_mean_ap.reset()
        
    def test_epoch_end(self, outputs):
        self.log_dict(
            {f'test_{k}': v for k, v in self.test_mean_ap.compute().items()}
        )
        self.test_mean_ap.reset()
        
        
def train_phase1(
    *,
    output_dir: str | PathLike,
    loader_train: DataLoader,
    num_classes: int,
    lr: float = 5e-5,
    max_epochs: int = 10,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    log_every_n_steps: int = 10
) -> str:
    # No validation or test since we don't tune the detector
    output_dir = Path(output_dir)
    
    model = LitFasterRCNN(num_classes, lr=lr, phase=1)
    
    trainer = pl.Trainer(
        default_root_dir=str(output_dir / 'phase1'),
        callbacks = [
            ModelCheckpoint(monitor='train_loss_sum', mode='min', save_weights_only=True),
        ],
        max_epochs=max_epochs,
        accelerator=accelerator,
        log_every_n_steps=log_every_n_steps,
    )
    trainer.fit(model, loader_train)
    
    ckpt: str = trainer.checkpoint_callback.best_model_path  # type: ignore
    return ckpt

def train_phase2(
    *,
    phase1_ckpt: str,
    output_dir: str | PathLike,
    loader_train: DataLoader,
    loader_val: DataLoader,
    loader_test: DataLoader,
    lr: float = 5e-5,
    max_epochs: int = 10,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    log_every_n_steps: int = 10
) -> str:
    output_dir = Path(output_dir)
    
    model: LitFasterRCNN = LitFasterRCNN.load_from_checkpoint(
        phase1_ckpt, lr=lr, phase=2
    )
    model.model.backbone = get_resnet50_fpn_backbone()
    
    trainer = pl.Trainer(
        default_root_dir=str(output_dir / 'phase2'),
        callbacks=[
            EarlyStopping(monitor='val_map', mode='max', patience=3),
            ModelCheckpoint(monitor='val_map', mode='max', save_weights_only=True),
        ],
        max_epochs=max_epochs,
        accelerator=accelerator,
        log_every_n_steps=log_every_n_steps,
    )
    trainer.fit(model, loader_train, loader_val)
    trainer.test(ckpt_path='best', dataloaders=loader_test)
    
    ckpt: str = trainer.checkpoint_callback.best_model_path  # type: ignore
    return ckpt
    

# Setup
data_dir = Path('data_small')
output_dir = Path('output')
faster_rcnn_dir = output_dir / 'faster_rcnn'
retinanet_dir = output_dir / 'retinanet'

dataset_train = ObjectsDataset(data_dir, mode='train')
dataset_val = ObjectsDataset(data_dir, mode='val')
dataset_test = ObjectsDataset(data_dir, mode='test')
num_classes = 1 + len(dataset_train.class_names)  # Including background


# --- Faster RCNN ---
loader_train, loader_test, loader_val = create_dataloaders(
    dataset_train, dataset_val, dataset_test, batch_size=4, collate_fn=transpose, num_workers=2
)
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

phase1_ckpt = train_phase1(
    output_dir=faster_rcnn_dir,
    loader_train=loader_train,
    num_classes=num_classes,
    accelerator=accelerator,
    max_epochs=2,
)
phase2_ckpt = train_phase2(
    phase1_ckpt=phase1_ckpt,
    output_dir=faster_rcnn_dir,
    loader_train=loader_train,
    loader_val=loader_val,
    loader_test=loader_test,
    accelerator=accelerator,
    max_epochs=2,
)







# %%
