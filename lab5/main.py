# %%
import itertools
from os import PathLike
from pathlib import Path
from typing import (Callable, List, Literal, Optional, Tuple, TypedDict)

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.detection import FasterRCNN
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone, BackboneWithFPN
)
from torchvision.io import read_image, ImageReadMode
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# IMPORTANT: Install torchmetrics via
# pip install -e git+https://github.com/k-papadakis/metrics.git#egg=torchmetrics
# otherwise MeanAveragePrecision might not work.
# See this https://github.com/Lightning-AI/metrics/issues/1147
from torchmetrics.detection.mean_ap import MeanAveragePrecision


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
        trainable_layers=2
    )


class LitFasterRCNN(pl.LightningModule):

    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        phase: Literal[1, 2, 3, 4] = 1
    ) -> None:
        # num_classes includes the background!
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.phase = phase
        backbone = get_resnet50_fpn_backbone()
        self.model = FasterRCNN(backbone, num_classes)
        self.val_mean_ap = MeanAveragePrecision()
        self.test_mean_ap = MeanAveragePrecision(class_metrics=True)

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[TargetsDict]] = None
    ) -> List[DetectionsDict] | LossesDict:
        """Returns losses if training else returns detections."""
        return self.model(images, targets)

    def training_step(
        self, batch: Tuple[List[Tensor], List[TargetsDict]], batch_idx: int
    ) -> Tensor:
        images, targets = batch
        losses: LossesDict = self.model(images, targets)

        match self.phase:
            case 1 | 3:
                a, b = 'loss_objectness', 'loss_rpn_box_reg'
                loss_sum = losses[a] + losses[b]
                d = {
                    f'train_{k}': losses[k] for k in (a, b)
                } | {
                    'train_loss_sum': loss_sum
                }
                self.log_dict(d, batch_size=len(images))
                return loss_sum
            case 2 | 4:
                a, b = 'loss_classifier', 'loss_box_reg'
                loss_sum = losses[a] + losses[b]
                d = {
                    f'train_{k}': losses[k] for k in (a, b)
                } | {
                    'train_loss_sum': loss_sum
                }
                self.log_dict(d, batch_size=len(images))
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
        metrics = self.val_mean_ap.compute()
        self.log_dict({f'val_{k}': v for k, v in metrics.items()})
        self.val_mean_ap.reset()

    def test_epoch_end(self, outputs):
        metrics = self.test_mean_ap.compute()

        map_per_class = metrics.pop('map_per_class')
        mar_100_per_class = metrics.pop('mar_100_per_class')

        metrics.update(
            (f'map_class_{i}', x) for i, x in enumerate(map_per_class)
        )
        metrics.update(
            (f'mar_100_class_{i}', x) for i, x in enumerate(mar_100_per_class)
        )

        self.log_dict({f'test_{k}': v for k, v in metrics.items()})
        self.test_mean_ap.reset()


def _train_faster_rcnn_phase(
    phase: Literal[1, 2, 3, 4],
    *,
    loader_train: DataLoader,
    output_dir: str | PathLike,
    prev_ckpt: Optional[str] = None,
    num_classes: Optional[int] = None,
    loader_val: Optional[DataLoader] = None,
    loader_test: Optional[DataLoader] = None,
    lr: float = 1e-4,
    max_epochs: int = 10,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    log_every_n_steps: int = 10,
    patience: int = 5,
) -> str:
    """Train a Faster RCNN phase.
    
    `num_classes` is used only for phase 1.
    `ckpt` is used for phases 2, 3 and 4.
    `loader_val` and `loader_test` are used only for phases 2 and 4.
    """

    print(f'Training Faster R-CNN Phase {phase}')

    # Model loading
    model: LitFasterRCNN

    match phase:
        case 1:
            assert num_classes is not None
            model = LitFasterRCNN(num_classes=num_classes, lr=lr, phase=phase)
        case 2 | 3 | 4:
            assert prev_ckpt is not None
            model = LitFasterRCNN.load_from_checkpoint(
                prev_ckpt, lr=lr, phase=phase-1,
            )
            if phase == 2:
                model.model.backbone = get_resnet50_fpn_backbone()

    # Model Training
    default_root_dir = str(Path(output_dir, f'phase{phase}'))

    match phase:
        case 1 | 3:
            trainer = pl.Trainer(
                default_root_dir=default_root_dir,
                callbacks=[
                    ModelCheckpoint(
                        monitor='train_loss_sum',
                        mode='min',
                        save_weights_only=True
                    ),
                ],
                max_epochs=max_epochs,
                accelerator=accelerator,
                log_every_n_steps=log_every_n_steps,
            )
            trainer.fit(model, loader_train)
        case 2 | 4:
            assert loader_train is not None and loader_val is not None
            trainer = pl.Trainer(
                default_root_dir=default_root_dir,
                callbacks=[
                    EarlyStopping(
                        monitor='val_map', mode='max', patience=patience
                    ),
                    ModelCheckpoint(
                        monitor='val_map', mode='max', save_weights_only=True
                    ),
                ],
                max_epochs=max_epochs,
                accelerator=accelerator,
                log_every_n_steps=log_every_n_steps,
            )
            trainer.fit(model, loader_train, loader_val)
            trainer.test(ckpt_path='best', dataloaders=loader_test)

    new_ckpt: str = trainer.checkpoint_callback.best_model_path  # type: ignore
    return new_ckpt


def train_faster_rcnn(
    dataset_train: ObjectsDataset, dataset_val: ObjectsDataset,
    dataset_test: ObjectsDataset, accelerator: Literal['cpu', 'gpu']
):
    faster_rcnn_dir = output_dir / 'faster_rcnn'
    num_classes = 1 + len(dataset_train.class_names)  # Including background
    loader_train, loader_test, loader_val = create_dataloaders(
        dataset_train,
        dataset_val,
        dataset_test,
        batch_size=8,
        collate_fn=transpose,
        num_workers=2
    )

    ckpt1 = _train_faster_rcnn_phase(
        phase=1,
        output_dir=faster_rcnn_dir,
        loader_train=loader_train,
        num_classes=num_classes,
        accelerator=accelerator,
        lr=5e-5,
        max_epochs=10,
    )
    ckpt2 = _train_faster_rcnn_phase(
        phase=2,
        output_dir=faster_rcnn_dir,
        prev_ckpt=ckpt1,
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test,
        accelerator=accelerator,
        lr=5e-5,
        max_epochs=10,
    )
    ckpt3 = _train_faster_rcnn_phase(
        phase=3,
        output_dir=faster_rcnn_dir,
        prev_ckpt=ckpt2,
        loader_train=loader_train,
        accelerator=accelerator,
        lr=1e-4,
        max_epochs=10,
    )
    ckpt4 = _train_faster_rcnn_phase(
        phase=4,
        output_dir=faster_rcnn_dir,
        prev_ckpt=ckpt3,
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test,
        accelerator=accelerator,
        lr=1e-3,
        max_epochs=100,
    )
    return ckpt4


# Setup
pl.seed_everything(42)
data_dir = Path('data')
output_dir = Path('output')
# retinanet_dir = output_dir / 'retinanet'

dataset_train = ObjectsDataset(data_dir, mode='train')
dataset_val = ObjectsDataset(data_dir, mode='val')
dataset_test = ObjectsDataset(data_dir, mode='test')
accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

faster_rcnn_ckpt = train_faster_rcnn(
    dataset_train, dataset_val, dataset_test, accelerator
)

# %%
