import itertools
from os import PathLike
from pathlib import Path
from typing import (
    Callable, Dict, List, Literal, Optional, Tuple, TypedDict, Union
)

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
from torchvision.models.detection.transform import (
    GeneralizedRCNNTransform, ImageList
)
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import RegionProposalNetwork
from torchvision.io import read_image, ImageReadMode
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm

# IMPORTANT: Install torchmetrics via
# pip install -e git+https://github.com/k-papadakis/metrics.git#egg=torchmetrics
# otherwise MeanAveragePrecision might not work.
# See this https://github.com/Lightning-AI/metrics/issues/1147
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class EmptyDict(TypedDict):
    pass


class TargetsDict(TypedDict):
    labels: Tensor
    boxes: Tensor


class RPNLossesDict(TypedDict):
    loss_objectness: Tensor
    loss_rpn_box_reg: Tensor


class DetectorLossesDict(TypedDict):
    loss_classifier: Tensor
    loss_box_reg: Tensor


class AllLossesDict(TypedDict):
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
        """"Object Detection: Batteries, Dice, and Toy Cars" from Kaggle.
        See https://www.kaggle.com/datasets/markcsizmadia/object-detection-batteries-dices-and-toy-cars?select=dataset
        """
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
            name: i for i, name in enumerate(self.class_names, 1)
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
        """Faster R-CNN 4-Step Alternating Training.
        
        Model structure
            - Phases 1 and 3: The model consists of only the Backbone (Resnet50) and the RPN.
            - Phase 2: The model consists of only the Backbone and the ROI Heads,
              and the proposals are part of the input (cached proposals from phase 1).
            - Phase 4: The model is a complete Faster R-CNN.
            
        Model forward
            - Phases 1 and 3: Returns the proposal losses if training, else returns the proposals.
              The combined loss is the sum of the proposal losses.
            - Phases 2 and 4: Returns the detection losses if training else returns the detections.
              The combined loss is the sum of the detection losses.
        
        Model training (see also `train_faster_rcnn`)
            - Phase 1: Train all layers (cache the proposals with `cache_phase1_proposals`).
            - Phase 2: Train all layers (with a new backbone, using the proposals from from phase 1).
            - Phase 3: Train only the RPN.
            - Phase 4: Train only the ROI heads.
            
        Model evaluation and testing
            - Phases 1, 2 and 3: Log only the proposal or detection losses and the combined loss.
            - Phase 4: Log the detection loss and the combined loss.
              Also, after every epoch compute and log detailed mAP and mAR results.
              Stop early if the the mAP doesn't improve.
        """
        # num_classes includes the background!
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.phase = phase
        backbone = get_resnet50_fpn_backbone()
        self.model = FasterRCNN(backbone, num_classes)
        self.val_mean_ap = MeanAveragePrecision()
        self.test_mean_ap = MeanAveragePrecision(class_metrics=True)

        match self.phase:
            case 1:
                self.model.roi_heads.requires_grad_(False)  # Not used
            case 2:
                self.model.rpn.requires_grad_(False)  # Not used
            case 3:
                self.model.backbone.requires_grad_(False)
                self.model.roi_heads.requires_grad_(False)  # Not used
            case 4:
                self.model.backbone.requires_grad_(False)
                self.model.rpn.requires_grad_(False)  # Not used
            case _:
                raise ValueError(f'Invalid phase value {self.phase}')

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[TargetsDict]] = None,
        cached_proposals: Optional[List[Tensor]] = None,
    ) -> RPNLossesDict | AllLossesDict | List[Tensor] | List[DetectionsDict]:
        """See also torchvision.models.detection.generalized_rcnn.GeneralizedRCNN.forward()"""
        image_list: ImageList
        original_image_sizes: List[Tuple[int, int]]
        features: Dict[str, Tensor]
        proposals: List[Tensor]
        detections: List[DetectionsDict]
        proposal_losses: RPNLossesDict | EmptyDict  # Empty if not training.
        detector_losses: DetectorLossesDict | EmptyDict  # Empty if not training.

        transform: GeneralizedRCNNTransform = self.model.transform  # type: ignore
        backbone: BackboneWithFPN = self.model.backbone  # type: ignore
        rpn: RegionProposalNetwork = self.model.rpn  # type: ignore
        roi_heads: RoIHeads = self.model.roi_heads  # type: ignore

        # Forward
        image_list, targets = transform(images, targets)
        features = backbone(image_list.tensors)

        match self.phase:
            case 1 | 3 | 4:
                proposals, proposal_losses = rpn(image_list, features, targets)
            case 2:
                assert cached_proposals is not None
                proposals = cached_proposals
            case _:
                raise ValueError(f'Invalid phase value {self.phase}')

        match self.phase:
            case 1 | 3:
                pass
            case 2 | 4:
                detections, detector_losses = roi_heads(
                    features, proposals, image_list.image_sizes, targets
                )
            case _:
                raise ValueError(f'Invalid phase value {self.phase}')

        match self.phase, self.training:
            case (1 | 3), True:
                return proposal_losses  # type: ignore
            case (2 | 4), True:
                return detector_losses  # type: ignore
            case (1 | 3), False:
                return proposals  # type: ignore
            case (2 | 4), False:
                original_image_sizes = [tuple(img.shape[-2:]) for img in images]
                detections = transform.postprocess(
                    detections, image_list.image_sizes, original_image_sizes  # type: ignore
                )
                return detections
            case _:
                raise ValueError('Invalid (phase, training) combination {self.phase}, {self.training}')

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

    def training_step(
        self,
        batch: Union[Tuple[List[Tensor], List[TargetsDict]],
                     Tuple[List[Tensor], List[TargetsDict], List[Tensor]]],
        batch_idx: int,
    ) -> Tensor:
        losses: RPNLossesDict | AllLossesDict

        match self.phase:
            case 1 | 3 | 4:
                assert len(batch) == 2
                images, targets = batch
                losses = self(images, targets)
            case 2:
                assert len(batch) == 3
                images, targets, cached_proposals = batch
                losses = self(images, targets, cached_proposals)
            case _:
                raise ValueError(f'Invalid phase value {self.phase}')

        loss = sum(losses.values())  # type: ignore
        self.log_dict(
            {f'train_{k}': v for k, v in losses.items()} |  # type: ignore
            {'train_loss': loss}
        )

        return loss

    def _shared_val_test_step(
        self,
        batch: Union[Tuple[List[Tensor], List[TargetsDict]],
                     Tuple[List[Tensor], List[TargetsDict], List[Tensor]]],
    ) -> Tuple[List[DetectionsDict], List[TargetsDict]]:
        detections: List[DetectionsDict]

        match self.phase:
            case 1 | 3:
                raise ValueError(
                    f'No validation or test step step for phase {self.phase}'
                )
            case 4:
                assert len(batch) == 2
                images, targets = batch
                detections = self(images)
            case 2:
                assert len(batch) == 3
                images, targets, cached_proposals = batch
                detections = self(images, cached_proposals=cached_proposals)
            case _:
                raise ValueError(f'Invalid phase value {self.phase}')

        return detections, targets

    def validation_step(
        self,
        batch: Union[Tuple[List[Tensor], List[TargetsDict]],
                     Tuple[List[Tensor], List[TargetsDict], List[Tensor]]],
        batch_idx: int,
    ) -> None:
        detections, targets = self._shared_val_test_step(batch)
        self.val_mean_ap.update(detections, targets)  # type: ignore

    def test_step(
        self,
        batch: Union[Tuple[List[Tensor], List[TargetsDict]],
                     Tuple[List[Tensor], List[TargetsDict], List[Tensor]]],
        batch_idx: int
    ) -> None:
        detections, targets = self._shared_val_test_step(batch)
        self.test_mean_ap.update(detections, targets)  # type: ignore

    def validation_epoch_end(self, outputs) -> None:
        metrics = self.val_mean_ap.compute()
        self.log_dict({f'val_{k}': v for k, v in metrics.items()})
        self.val_mean_ap.reset()

    def test_epoch_end(self, outputs) -> None:
        metrics = self.test_mean_ap.compute()

        map_per_class = metrics.pop('map_per_class')
        mar_100_per_class = metrics.pop('mar_100_per_class')

        metrics.update(
            (f'map_class_{i}', x) for i, x in enumerate(map_per_class, 1)
        )
        metrics.update(
            (f'mar_100_class_{i}', x) for i, x in enumerate(mar_100_per_class, 1)
        )

        self.log_dict({f'test_{k}': v for k, v in metrics.items()})
        self.test_mean_ap.reset()


def cache_phase1_proposals(
    ckpt: str,
    dataset: ObjectsDataset,
    cache_dir: str | PathLike,
    accelerator: Literal['cpu', 'gpu'] = 'cpu',
    batch_size=8,
) -> None:
    images: List[Tensor]
    proposals: List[Tensor]

    model: LitFasterRCNN
    model = LitFasterRCNN.load_from_checkpoint(ckpt)
    if model.phase != 1:
        raise ValueError(f'Model phase = {model.phase} != 1')

    print('Caching Faster R-CNN Phase 1 Proposals')

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True, parents=True)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=transpose
    )
    device = torch.device(accelerator if accelerator != 'gpu' else 'cuda')

    was_training = model.training
    model.train(False)
    model.to(device)

    # Caching
    names = iter(dataset.image_names)
    for images, _ in tqdm(loader):
        images = [img.to(device) for img in images]
        with torch.inference_mode():
            proposals = model(images)
        for proposal in proposals:
            torch.save(proposal.cpu(), cache_dir / f'{next(names)}.pt')

    model.train(was_training)


class Phase2Dataset(ObjectsDataset):

    def __init__(
        self, root_dir: str | PathLike, mode: Literal['train', 'val', 'test'],
        proposals_dir: str | PathLike
    ) -> None:
        """ObjectsDataset with proposals"""
        super().__init__(root_dir, mode)
        self.proposals_dir = Path(proposals_dir)

        names = sorted(p.stem for p in self.proposals_dir.iterdir())
        if names != self.image_names:
            raise ValueError('Non matching proposal and image names.')

    def __getitem__(self, idx) -> Tuple[Tensor, TargetsDict, Tensor]:
        image, targets = super().__getitem__(idx)
        proposals = torch.load(
            self.proposals_dir / f'{self.image_names[idx]}.pt'
        )
        return image, targets, proposals


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
    `loader_val` and `loader_test` are used only for phase and 4.
    
    Returns the path to the best model checkpoint saved during training.
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
                prev_ckpt, lr=lr, phase=phase,
            )
            if phase == 2:
                model.model.backbone = get_resnet50_fpn_backbone()

    # Model Training
    default_root_dir = str(Path(output_dir, f'phase{phase}'))

    match phase:
        case 1 | 2 | 3:
            trainer = pl.Trainer(
                default_root_dir=default_root_dir,
                callbacks=[
                    ModelCheckpoint(
                        monitor='train_loss',
                        mode='min',
                        save_weights_only=True
                    ),
                ],
                max_epochs=max_epochs,
                accelerator=accelerator,
                log_every_n_steps=log_every_n_steps,
            )
            trainer.fit(model, loader_train)
        case 4:
            assert loader_train is not None and loader_val is not None
            trainer = pl.Trainer(
                default_root_dir=default_root_dir,
                callbacks=[
                    EarlyStopping(
                        monitor='val_map', mode='max', patience=patience, verbose=True
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
    data_dir: str | PathLike,
    output_dir: str | PathLike,
    batch_size: int,
    accelerator: Literal['cpu', 'gpu'],
    num_workers: int,
):
    # Set up the datasets and dataloaders that are used in phases 1, 3 and 4.
    dataset_train_p134 = ObjectsDataset(data_dir, mode='train')
    dataset_val_p134 = ObjectsDataset(data_dir, mode='val')
    dataset_test_p134 = ObjectsDataset(data_dir, mode='test')
    loader_train_p134, loader_test_p134, loader_val_p134 = create_dataloaders(
        dataset_train_p134,
        dataset_val_p134,
        dataset_test_p134,
        batch_size=batch_size,
        collate_fn=transpose,
        num_workers=num_workers,
    )
    faster_rcnn_dir = Path(output_dir, 'faster_rcnn')
    num_classes = 1 + len(dataset_train_p134.class_names)  # Including background

    # Train phase 1 and cache the proposals.
    ckpt1 = _train_faster_rcnn_phase(
        phase=1,
        output_dir=faster_rcnn_dir,
        loader_train=loader_train_p134,
        num_classes=num_classes,
        accelerator=accelerator,
        lr=5e-5,
        max_epochs=10,
    )
    proposals_dir = Path(ckpt1).parent.parent / 'proposals_cache'  # version_[i] / 'proposals_cache'
    cache_phase1_proposals(
        ckpt1,
        dataset_train_p134,
        proposals_dir,
        accelerator=accelerator,
        batch_size=batch_size
    )

    # Set up the dataset and loader that is used in phase 2.
    dataset_train_p2 = Phase2Dataset(
        data_dir, mode='train', proposals_dir=proposals_dir
    )
    loader_train_p2 = DataLoader(
        dataset_train_p2,
        batch_size=batch_size,
        collate_fn=transpose,
        num_workers=num_workers
    )

    # Train phases 2, 3 and 4 and return the best phase 4 model checkpoint path.
    ckpt2 = _train_faster_rcnn_phase(
        phase=2,
        output_dir=faster_rcnn_dir,
        prev_ckpt=ckpt1,
        loader_train=loader_train_p2,
        accelerator=accelerator,
        lr=5e-5,
        max_epochs=10,
    )
    ckpt3 = _train_faster_rcnn_phase(
        phase=3,
        output_dir=faster_rcnn_dir,
        prev_ckpt=ckpt2,
        loader_train=loader_train_p134,
        accelerator=accelerator,
        lr=1e-4,
        max_epochs=10,
    )
    ckpt4 = _train_faster_rcnn_phase(
        phase=4,
        output_dir=faster_rcnn_dir,
        prev_ckpt=ckpt3,
        loader_train=loader_train_p134,
        loader_val=loader_val_p134,
        loader_test=loader_test_p134,
        accelerator=accelerator,
        lr=1e-4,
        max_epochs=30,
    )
    return ckpt4


def main():
    # Setup
    pl.seed_everything(42)
    data_dir = Path('data')
    output_dir = Path('output')
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    dataset = ObjectsDataset(data_dir, 'train')

    faster_rcnn_ckpt = train_faster_rcnn(
    data_dir, output_dir, batch_size=8, accelerator=accelerator, num_workers=2
    )


if __name__ == '__main__':
    main()
