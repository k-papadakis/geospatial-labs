# %%
from os import PathLike
from pathlib import Path
from typing import (Callable, List, Literal, Optional, Tuple, Dict, TypedDict)

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam, Optimizer
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import convert_image_dtype
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import (
    resnet_fpn_backbone, BackboneWithFPN
)
from torchvision.models.detection.image_list import ImageList
from torchvision.io import read_image
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class TargetsDict(TypedDict):
    labels: Tensor
    boxes: Tensor


class RPNLossesDict(TypedDict):
    loss_objectness: Tensor
    loss_rpn_box_reg: Tensor


class EmptyDict(TypedDict):
    pass


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

        image = read_image(str(image_path))
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


# %%
def get_resnet50_fpn_backbone() -> BackboneWithFPN:
    return resnet_fpn_backbone(
        backbone_name='resnet50',
        weights=ResNet50_Weights.DEFAULT,
        trainable_layers=3
    )


class LitFasterRCNNPhase1(pl.LightningModule):
    # Not implementing val_step, test_step,
    # because the rpn doesn't output loss if rpn.training.
    # The rpn has batchnorm layers, so it hard to disable them manually.

    def __init__(
        self,
        model: FasterRCNN,
        rpn_lr: float = 1e-3,
        backbone_lr: float = 1e-5
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.rpn_lr = rpn_lr
        self.backbone_lr = backbone_lr

    def forward(
        self,
        images: List[Tensor],
        targets: Optional[List[Dict[str, Tensor]]] = None
    ) -> Tuple[List[Tensor], RPNLossesDict | EmptyDict]:
        image_list: ImageList
        features: Dict[str, Tensor]
        proposals: List[Tensor]
        proposal_losses: RPNLossesDict | EmptyDict  # is empty if model.training

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
        loss_combined = (
            proposal_losses['loss_objectness'] +
            proposal_losses['loss_rpn_box_reg']
        )
        self.log_dict(proposal_losses | {'loss_combined': loss_combined})
        return loss_combined

    def configure_optimizers(self) -> Tuple[Optimizer, Optimizer]:
        backbone_optim = Adam(
            self.model.backbone.parameters(), lr=self.backbone_lr
        )
        rpn_optim = Adam(self.model.rpn.parameters(), lr=self.rpn_lr)
        return backbone_optim, rpn_optim

    def predict_step(
        self,
        batch: Tuple[List[Tensor], List[TargetsDict]],
        batch_idx: int,
    ) -> List[Tensor]:
        images, _ = batch
        proposals, _ = self(images)
        return proposals


data_root_dir = Path('data')
output_dir = Path('output')
output_dir.mkdir(exist_ok=True)

dataset_train = ODDataset(data_root_dir, 'train')
dataset_val = ODDataset(data_root_dir, 'val')
dataset_test = ODDataset(data_root_dir, 'test')

loader_train = DataLoader(
    dataset_train, batch_size=8, shuffle=True, collate_fn=transpose
)

model_entire = FasterRCNN(get_resnet50_fpn_backbone(), 7)
p1 = LitFasterRCNNPhase1(model_entire)
callbacks = ModelCheckpoint(monitor='loss_combined', mode='min')
trainer = pl.Trainer(
    max_epochs=2,
    accelerator='gpu',
    callbacks=callbacks,
    default_root_dir=str(output_dir / 'faster_rccn' / 'phase_1')
)
trainer.fit(p1, train_dataloaders=loader_train)

# %%
# TODO?: jit
# TODO: Make 4 separate Lightning Modules, one for each phase.
#  On __init__ use a pre-initialized Faster R-CNN
# TODO: When phase 1 finishes, use inference and cache all the rpn preds.
#  Zip (somehow) the original and the cached and create a DataLoader.
#  Use this new DataLoader to perform phase 2 train.
#
# loader_train_unshuffled = DataLoader(
#     dataset_train, batch_size=8, shuffle=True, collate_fn=transpose
# )