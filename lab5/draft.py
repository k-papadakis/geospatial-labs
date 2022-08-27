# %%
import itertools
from collections import OrderedDict
from operator import itemgetter
from os import PathLike
from pathlib import Path
import re
from typing import (Callable, List, Literal, Optional, Tuple, Dict, TypedDict)

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
from pytorch_lightning.callbacks import ModelCheckpoint


class EmptyDict(TypedDict):
    pass


class TargetsDict(TypedDict):
    labels: Tensor
    boxes: Tensor


class RPNLossesDict(TypedDict):
    loss_objectness: Tensor
    loss_rpn_box_reg: Tensor


class DetectionsDict(TypedDict):
    labels: Tensor
    boxes: Tensor
    scores: Tensor


class DetectorLossesDict(TypedDict):
    loss_classifier: Tensor
    loss_box_reg: Tensor


class Phase3CacheDict(TypedDict):
    features: Dict[str, Tensor]
    targets: TargetsDict
    original_image_size: Tuple[int, int]
    image_size: Tuple[int, int]


class ODDataset(Dataset):

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


def get_resnet50_fpn_backbone() -> BackboneWithFPN:
    return resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=ResNet50_Weights.DEFAULT,
        trainable_layers=3
    )


class LitFasterRCNNPhase1(pl.LightningModule):
    # Not implementing val_step, test_step,
    # because the rpn doesn't output loss if rpn.training.
    # The rpn has batchnorm layers, so it is hard to disable them manually.

    def __init__(self, num_classes: int, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FasterRCNN(get_resnet50_fpn_backbone(), num_classes)
        self.lr = lr

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[TargetsDict] = None
    ) -> Tuple[List[Tensor], RPNLossesDict | EmptyDict]:
        image_list: ImageList
        features: Dict[str, Tensor]
        proposals: List[Tensor]
        proposal_losses: RPNLossesDict | EmptyDict  # empty if not training

        image_list, targets = self.model.transform(images, targets)
        features = self.model.backbone(image_list.tensors)
        proposals, proposal_losses = self.model.rpn(
            image_list, features, targets
        )
        return proposals, proposal_losses

    def training_step(
        self,
        batch: Tuple[List[Tensor], List[TargetsDict]],
        batch_idx: int,
        optimizer_idx: Optional[int] = None
    ) -> Tensor:
        images, targets = batch
        _, proposal_losses = self(images, targets)
        loss_rpn = (
            proposal_losses['loss_objectness'] +
            proposal_losses['loss_rpn_box_reg']
        )
        self.log_dict({**proposal_losses, 'loss_rpn': loss_rpn})
        return loss_rpn

    def configure_optimizers(self) -> Optimizer:
        params = itertools.chain(
            self.model.backbone.parameters(), self.model.rpn.parameters()
        )
        return Adam(params, lr=self.lr)

    # def predict_step(
    #     self,
    #     batch: Tuple[List[Tensor], List[TargetsDict]],
    #     batch_idx: int,
    # ) -> List[Tensor]:
    #     images, _ = batch
    #     proposals, _ = self(images)
    #     return proposals


@torch.inference_mode()
def _cache_phase1(
    p1: LitFasterRCNNPhase1,
    dataset: ODDataset,
    cache_dir: str | PathLike,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    batch_size=8,
):
    images: List[Tensor]
    proposals: List[Tensor]  # Tensor shape: (1000, 4)

    print('Caching Phase 1')

    # Setup
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=transpose
    )
    device = torch.device(accelerator if accelerator != 'gpu' else 'cuda')

    was_training = p1.training
    p1.train(False)
    p1.to(device)

    # Caching
    names = iter(dataset.image_names)
    for images, _ in tqdm(loader):
        images = [img.to(device) for img in images]
        proposals, _ = p1(images)
        for proposal in proposals:
            torch.save(proposal.cpu(), cache_dir / f'{next(names)}.pt')

    # Teardown
    p1.train(was_training)


def train_phase1(
    data_root_dir: str | PathLike,
    *,
    output_dir: str | PathLike,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    max_epochs: int = 10,
    batch_size: int = 8,
    num_workers: int = 2,
) -> Tuple[str, str]:

    dataset = ODDataset(data_root_dir, 'train')
    num_classes = len(dataset.class_names)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=transpose,
        num_workers=num_workers,
    )

    # Train the RPN and the Backbone
    p1 = LitFasterRCNNPhase1(num_classes)
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        default_root_dir=str(output_dir),
        callbacks=[
            ModelCheckpoint(
                monitor='loss_rpn',
                mode='min',
                save_weights_only=True,
            ),
        ]
    )

    # Cache the RPN proposals for Phase 2.
    trainer.fit(p1, train_dataloaders=loader)
    ckpt: str = trainer.checkpoint_callback.model_path  # type: ignore
    p1.load_from_checkpoint(ckpt)
    cache_dir = str(Path(trainer.log_dir) / 'rpn_cache')  # type: ignore
    _cache_phase1(
        p1,
        dataset,
        cache_dir,
        accelerator=accelerator,
        batch_size=batch_size,
    )

    return ckpt, cache_dir


class Phase2Dataset(Dataset):

    def __init__(
        self, root_dir: str | PathLike, mode: Literal['train', 'val', 'test'],
        proposals_dir: str | PathLike
    ) -> None:
        self.dataset = ODDataset(root_dir, mode)
        self.proposals_dir = Path(proposals_dir)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, TargetsDict]:
        image, targets = self.dataset[idx]
        proposals = torch.load(
            self.proposals_dir / f'{self.dataset.image_names[idx]}.pt'
        )
        return image, proposals, targets


class LitFasterRCNNPhase2(pl.LightningModule):

    def __init__(self, num_classes: int, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FasterRCNN(get_resnet50_fpn_backbone(), num_classes)
        self.lr = lr

    def forward(
        self,
        images: List[Tensor],
        proposals: List[Tensor],
        targets: Optional[TargetsDict] = None,
    ) -> Tuple[List[DetectionsDict], DetectorLossesDict | EmptyDict]:
        image_list: ImageList
        features: Dict[str, Tensor]  # Keys: '0', '1', '2', '3', 'pool'
        original_image_sizes: List[Tuple[int, int]]
        detections: List[DetectionsDict]
        detector_losses: DetectorLossesDict | EmptyDict  # empty if not training

        original_image_sizes = [tuple(img.shape[-2:]) for img in images]
        image_list, targets = self.model.transform(images, targets)
        features = self.model.backbone(image_list.tensors)
        detections, detector_losses = self.model.roi_heads(
            features, proposals, image_list.image_sizes, targets
        )
        detections = self.model.transform.postprocess(  # type: ignore
            detections, image_list.image_sizes, original_image_sizes
        )
        return detections, detector_losses

    def training_step(
        self,
        batch: Tuple[List[Tensor], List[Tensor], List[TargetsDict]],
        batch_idx: int,
        optimizer_idx: Optional[int] = None
    ) -> Tensor:
        images, proposals, targets = batch

        _, detector_losses = self(images, proposals, targets)
        loss_detector = (
            detector_losses['loss_classifier'] + detector_losses['loss_box_reg']
        )
        self.log_dict({**detector_losses, 'loss_detector': loss_detector})
        return loss_detector

    def configure_optimizers(self) -> Optimizer:
        params = itertools.chain(
            self.model.backbone.parameters(), self.model.roi_heads.parameters()
        )
        return Adam(params, lr=self.lr)

    # def predict_step(
    #     self,
    #     batch: Tuple[List[Tensor], List[Tensor], List[TargetsDict]],
    #     batch_idx: int,
    # ) -> List[Tensor]:
    #     images, proposals, _ = batch
    #     detections, _ = self(images, proposals)
    #     return detections


@torch.inference_mode()
def _cache_phase2(
    p2: LitFasterRCNNPhase2,
    dataset: Phase2Dataset,
    cache_dir: str | PathLike,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    batch_size=8,
):
    images: List[Tensor]
    targets: List[TargetsDict]
    image_list: ImageList
    features: Dict[str, Tensor]

    print('Caching Phase 2')

    # Setup
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=transpose
    )
    device = torch.device(accelerator if accelerator != 'gpu' else 'cuda')

    was_training = p2.training
    p2.train(False)
    p2.to(device)

    # Caching
    names = iter(dataset.dataset.image_names)
    for images, _, targets in tqdm(loader):
        images = [img.to(device) for img in images]
        image_list, targets = p2.model.transform(images, targets)
        features = p2.model.backbone(image_list.tensors)

        for i in range(len(images)):
            d: Phase3CacheDict = {
                'features': {k: v[i].cpu() for k, v in features.items()},
                'targets': targets[i],
                'original_image_size': tuple(images[i].shape[-2:]),
                'image_size': image_list.image_sizes[i],
            }
            torch.save(d, cache_dir / f'{next(names)}.pt')

    # Teardown
    p2.train(was_training)


def train_phase_2(
    data_root_dir: str | PathLike,
    *,
    output_dir: str | PathLike,
    ckpt_p1: str,
    rpn_cache_dir: str,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    max_epochs: int = 10,
    batch_size: int = 8,
    num_workers: int = 2,
) -> Tuple[str, str]:
    dataset = Phase2Dataset(data_root_dir, 'train', proposals_dir=rpn_cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=transpose,
        num_workers=num_workers,
    )

    p2: LitFasterRCNNPhase2
    p2 = LitFasterRCNNPhase2.load_from_checkpoint(ckpt_p1)
    p2.model.backbone = get_resnet50_fpn_backbone()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=accelerator,
        default_root_dir=str(output_dir),
        callbacks=[
            ModelCheckpoint(
                monitor='loss_detector',
                mode='min',
                save_weights_only=True,
            ),
        ]
    )
    trainer.fit(p2, train_dataloaders=loader)
    ckpt_p2: str = trainer.checkpoint_callback.model_path  # type: ignore
    p2.load_from_checkpoint(ckpt_p2)
    cache_dir = str(Path(trainer.log_dir) / 'features_cache')  # type: ignore
    _cache_phase2(
        p2,
        dataset,
        cache_dir,
        accelerator=accelerator,
        batch_size=batch_size,
    )

    return ckpt_p2, cache_dir


class Phase34Dataset(Dataset):

    def __init__(self, features_dir: str | PathLike) -> None:
        self.features_dir = Path(features_dir)
        self.names = sorted(p.name for p in self.features_dir.iterdir())

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx) -> Phase3CacheDict:
        name = self.names[idx]
        return torch.load(self.features_dir / name)


def phase34_collate_fn(batch):
    feature_names = list(batch[0]['features'])
    features_dict_list = {k: [] for k in feature_names}
    for t in batch:
        for k, v in t['features'].items():
            features_dict_list[k].append(v)
    features_collated = {
        k: torch.stack(v) for k, v in features_dict_list.items()
    }
    targets_collated = [d['targets'] for d in batch]
    image_sizes_collated = [d['image_size'] for d in batch]
    original_image_sizes_collated = [d['original_image_size'] for d in batch]
    collated = {
        'features': features_collated,  # Dict[str: Tensor]
        'image_sizes': image_sizes_collated,  # List[Tuple[int, int]]
        'original_image_sizes':
            original_image_sizes_collated,  # List[Tuple[int, int]]
        'targets': targets_collated,  # List[TargetsDict]
    }
    return collated


class LitFasterRCNNPhase3(pl.LightningDataModule):

    def __init__(self, num_classes: int, lr: float = 1e-4) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = FasterRCNN(get_resnet50_fpn_backbone(), num_classes)
        self.lr = lr

    def forward(
        self,
        features: Dict[str, Tensor],
        image_sizes: List[Tuple[int, int]],
        original_image_sizes: List[Tuple[int, int]],
        targets: List[TargetsDict],
    ):
        pass        


def train_phase3(
    data_root_dir: str | PathLike,
    *,
    output_dir: str | PathLike,
    ckpt_p2: str,
    features_cache_dir: str,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    max_epochs: int = 10,
    batch_size: int = 8,
    num_workers: int = 2,
):  # TODO: typed return
    dataset = Phase34Dataset(features_cache_dir)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=phase34_collate_fn,
        num_workers=num_workers,
    )

    # p3: LitFasterRCNNPhase3
    # p3 = LitFasterRCNNPhase3.load_from_checkpoint()


def main() -> None:
    pl.seed_everything(42)
    data_root_dir = Path('data')
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True, parents=True)
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    ckpt_p1, rpn_cache_dir = train_phase1(
        data_root_dir,
        output_dir=output_dir / 'faster_rcnn' / 'phase1',
        accelerator=accelerator,
        max_epochs=1,
    )

    ckpt_p2, cache_dir = train_phase_2(
        data_root_dir,
        output_dir=output_dir / 'faster_rcnn' / 'phase2',
        ckpt_p1=ckpt_p1,
        rpn_cache_dir=rpn_cache_dir,
        accelerator=accelerator,
        max_epochs=1,
    )


# main()
# %%

# dataset = Phase3Dataset(
#     'output/faster_rcnn/phase2/lightning_logs/version_0/features_cache'
# )
# loader = DataLoader(dataset, batch_size=4, collate_fn=phase3_collate_fn, shuffle=False)
# res = next(iter(loader))