# %%
from pathlib import Path

from lab3.datasets import PatchDatasetPostPad, CroppedDataset, AugmentedDataset
from lab3.datasets import make_pixel_dataset, flip_and_rotate, split_dataset
from lab3.models import LitMLP, LitCNN, LitTransferResNet, LitUNet
from lab3.utils import read_data, read_info, evaluate_model
from lab3.utils import display_image_and_label, display_patch, display_segmentation
from lab3.utils import predict_all_images_cnn, predict_all_images_unet

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

##########################################################
# MODEL TRAINING FUNCTIONS (Hardcoded)
##########################################################

def train_evaluate_traditional(
    images, label_images, names,
    random_state=None, output_dir='output',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    x_train, x_test, y_train, y_test = train_test_split(
        *make_pixel_dataset(images, label_images),
        test_size=0.3, random_state=random_state
    )
    models = [
        ('SVM', make_pipeline(StandardScaler(), SVC(C=1.0, kernel='rbf'))),
        ('RandomForest', RandomForestClassifier(n_estimators=100)),
    ]
    
    for name, model in models:
        fig, ax = plt.subplots(figsize=(9, 9))
        
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        acc = accuracy_score(y_test, y_pred)
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred,
            display_labels=names,
        ).plot(xticks_rotation=90, ax=ax, colorbar=False)
        ax.set_title(f'{name} (acc={acc : .2%})')
        fig.savefig(output_dir / f'{name}_confmat.png')
    
    
def train_mlp(
    images, label_images, names, epochs=50,
    accelerator='cpu', output_dir='output', random_state=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    xs, ys = make_pixel_dataset(images, label_images)
    xs = torch.tensor(xs, dtype=torch.float32)
    ys = torch.tensor(ys, dtype=torch.int64)
    pixel_dataset = TensorDataset(xs, ys)
    
    pixel_dataset_train, pixel_dataset_val, pixel_dataset_test = split_dataset(
        pixel_dataset, train_size=0.7, test_size=0.15, seed=random_state
    )
    pixel_loader_train = DataLoader(pixel_dataset_train, batch_size=512, num_workers=2, shuffle=True)
    pixel_loader_val = DataLoader(pixel_dataset_val, batch_size=512, num_workers=2)
    pixel_loader_test = DataLoader(pixel_dataset_test, batch_size=512, num_workers=2)
    
    mlp = LitMLP(176, 14, lr=1e-3, weight_decay=1e-5)
    
    callbacks = [
        # EarlyStopping(monitor='accuracy/val', mode='max', patience=10),
        ModelCheckpoint(monitor='loss/val', mode='min', save_last=True),
    ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        callbacks=callbacks,
        default_root_dir=output_dir / 'mlp_results'
    )
    trainer.fit(mlp, pixel_loader_train, pixel_loader_val)
    evaluate_model(trainer, pixel_loader_test, names, output_dir=output_dir)
    
    
def train_cnn(
    loader_train, loader_val, loader_test, names,
    epochs=50, accelerator='cpu', output_dir='output'
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # WARNING! YOU WILL GET GOOD RESULTS BECAUSE THE IMAGES IN TRAIN AND TEST OVERLAP!
    # The labels don't, but because the labels are locally continuous, we have information leakage via the images!
    cnn = LitCNN(176, 14, lr=1e-3)
    callbacks = [
        # EarlyStopping(monitor='accuracy/val', mode='max', patience=10),
        ModelCheckpoint(monitor='loss/val', mode='min', save_last=True)
    ]
    trainer = pl.Trainer(
        accelerator=accelerator, 
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir=output_dir / ('cnn_results' if loader_test is not None else 'cnn_results_full')
    )
    trainer.fit(cnn, train_dataloaders=loader_train, val_dataloaders=loader_val)

    if loader_test is not None:
        evaluate_model(trainer, loader_test, names, output_dir=output_dir)


def train_resnet(
    loader_train, loader_val, loader_test, names, freeze_head=False, epochs=50,
    accelerator='cpu', output_dir='output',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    resnet = LitTransferResNet(14, freeze_head=freeze_head, lr=1e-4)
    
    callbacks = [
        # EarlyStopping(monitor='accuracy/val', mode='max', patience=10),
        ModelCheckpoint(monitor='loss/val', mode='min', save_last=True),
    ]
    trainer = pl.Trainer(
        max_epochs=epochs,
        accelerator=accelerator,
        callbacks=callbacks,
        default_root_dir=output_dir / 'resnet_results'
    )
    trainer.fit(resnet, loader_train, loader_val)
    evaluate_model(trainer, loader_test, names, output_dir=output_dir)

    
def train_unet(
    loader_train, loader_val, loader_test, names, epochs,
    accelerator='cpu', output_dir='output',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model = LitUNet(176, 14, ignore_index=-1, lr=1e-4)
    callbacks = [
        # Too much variance due sample size variability due to label exclusion
        # EarlyStopping(monitor='accuracy/val', mode='max', patience=10),
        ModelCheckpoint(monitor='loss/val', mode='min', save_last=False),
    ]
    trainer = pl.Trainer(
        accelerator=accelerator, 
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir=output_dir / 'unet_results'
    )
    trainer.fit(model, train_dataloaders=loader_train, val_dataloaders=loader_val)

    evaluate_model(
        trainer, loader_test, names, output_dir=output_dir
    )


def train_unet_overlap(
    crop_size, stride, images, label_images, batch_size, epochs,
    accelerator='cpu', output_dir='output',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    dataset = ConcatDataset([
        CroppedDataset(
            image, label_image, stride=stride,
            crop_size=crop_size, use_padding=True,
        )
        for image, label_image in zip(images, label_images)
    ])

    dataset = AugmentedDataset(
        dataset, transform=flip_and_rotate, apply_on_target=True
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LitUNet(176, 14, ignore_index=-1, lr=1e-4)
    callbacks = [
        ModelCheckpoint(monitor='loss/train', mode='min', save_last=False),
    ]
    trainer = pl.Trainer(
        accelerator=accelerator,
        max_epochs=epochs,
        callbacks=callbacks,
        default_root_dir=output_dir / 'unet_results_overlap'
    )
    trainer.fit(model, train_dataloaders=loader)
    
    return trainer


def main():
    random_state = 42
    rgb = (23, 11, 7)
    info_path = 'hyrank_info.csv'
    train_x_paths = (
        'HyRANK_satellite/TrainingSet/Dioni.tif',
        'HyRANK_satellite/TrainingSet/Loukia.tif',
    )
    train_y_paths = (
        'HyRANK_satellite/TrainingSet/Dioni_GT.tif',
        'HyRANK_satellite/TrainingSet/Loukia_GT.tif'
    )
    validation_x_paths = (
        'HyRANK_satellite/ValidationSet/Erato.tif',
        'HyRANK_satellite/ValidationSet/Kirki.tif',
        'HyRANK_satellite/ValidationSet/Nefeli.tif',
    )
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    DEVICE = 'gpu' if torch.cuda.is_available() else 'cpu'
    
    # LOAD THE IMAGES AND THEIR INFO
    images, label_images = read_data(train_x_paths, train_y_paths)
    info = read_info(info_path)
    classnames = info['name']
    colors = info['color']
    cmap = mpl.colors.ListedColormap(info['color'])
    norm = mpl.colors.BoundaryNorm(np.arange(-1, 14+1), ncolors=15)

    # Display the training data
    for image, label, path in zip(images, label_images, train_x_paths):
        path = Path(path)
        display_image_and_label(
            image=image, label=label,
            cmap=cmap, norm=norm, classnames=classnames, rgb=rgb, title=path.stem,
        )
        plt.savefig(output_dir / path.with_suffix('.png').name)

    # CREATE DATASETS AND DATALOADERS
    # Patch Dataset All Channels
    patch_dataset = ConcatDataset([
        PatchDatasetPostPad(image, label_image, patch_size=15)
        for image, label_image in zip(images, label_images)
    ])
    patch_dataset_train, patch_dataset_val, patch_dataset_test = split_dataset(
        patch_dataset, train_size=0.7, test_size=0.15, seed=random_state
    )
    patch_dataset_train_aug = AugmentedDataset(patch_dataset_train, flip_and_rotate)

    patch_loader_full = DataLoader(AugmentedDataset(patch_dataset, flip_and_rotate), batch_size=64, shuffle=True, num_workers=2)
    patch_loader_train = DataLoader(patch_dataset_train, batch_size=64, shuffle=True, num_workers=2)
    patch_loader_train_aug = DataLoader(patch_dataset_train_aug, batch_size=64, shuffle=True, num_workers=2)
    patch_loader_val = DataLoader(patch_dataset_val, batch_size=64, num_workers=2)
    patch_loader_test = DataLoader(patch_dataset_test, batch_size=64, num_workers=2)

    # Patch Dataset RGB Only   
    rgb_dataset = ConcatDataset([
        PatchDatasetPostPad(image, label_image, patch_size=15, channels=rgb)
        for image, label_image in zip(images, label_images)
    ])
    rgb_dataset_train, rgb_dataset_val, rgb_dataset_test = split_dataset(
        rgb_dataset, train_size=0.7, test_size=0.15, seed=random_state
    )
    rgb_dataset_train_aug = AugmentedDataset(rgb_dataset_train, flip_and_rotate)

    rgb_loader_train = DataLoader(rgb_dataset_train, batch_size=256, shuffle=True, num_workers=2)
    rgb_loader_train_aug = DataLoader(rgb_dataset_train_aug, batch_size=256, shuffle=True, num_workers=2)
    rgb_loader_val = DataLoader(rgb_dataset_val, batch_size=256, num_workers=2)
    rgb_loader_test = DataLoader(rgb_dataset_test, batch_size=256, num_workers=2)

    # Cropped Dataset
    stride = 64
    crop_size = 64
    cropped_dataset = ConcatDataset([
        CroppedDataset(
            image, label_image, crop_size=crop_size,
            stride=stride, use_padding=True,
        )  
        for image, label_image in zip(images, label_images)
    ])
    cropped_dataset_train, cropped_dataset_val, cropped_dataset_test = split_dataset(
        cropped_dataset, train_size=0.7, test_size=0.15, seed=random_state
    )
    cropped_dataset_train_aug = AugmentedDataset(
        cropped_dataset_train, transform=flip_and_rotate, apply_on_target=True
    )

    # Use batch_size=2 when running locally
    cropped_loader_train_aug = DataLoader(cropped_dataset_train_aug, batch_size=16, shuffle=True, num_workers=2)
    cropped_loader_val = DataLoader(cropped_dataset_val, batch_size=16, num_workers=2)
    cropped_loader_test = DataLoader(cropped_dataset_test, batch_size=16, num_workers=2)

    display_segmentation(cropped_dataset_train_aug, 80, cmap=cmap, norm=norm, names=classnames, rgb=rgb, n_repeats=5)    

    print('\nTraining Random Forest and SVM...')
    train_evaluate_traditional(images, label_images, names=classnames[1:], random_state=random_state)
    print('\nTraining MLP...')
    train_mlp(images, label_images, names=classnames[1:], random_state=random_state, epochs=2_000, accelerator=DEVICE, output_dir=output_dir)
    print('\nTraining CNN...')
    train_cnn(patch_loader_train_aug, patch_loader_val, patch_loader_test, names=classnames[1:], epochs=2_000, accelerator=DEVICE, output_dir=output_dir)
    print('\nTraining ResNet...')
    train_resnet(rgb_loader_train_aug, rgb_loader_val, rgb_loader_test, names=classnames[1:], freeze_head=False, epochs=2_000, accelerator=DEVICE, output_dir=output_dir)
    print('\nTraining U-Net...')
    train_unet(cropped_loader_train_aug, cropped_loader_val, cropped_loader_test, names=classnames[1:], epochs=2_000, accelerator=DEVICE, output_dir=output_dir)

    print('\nTraining CNN on all the data')
    train_cnn(patch_loader_full, None, None, names=classnames[1:], epochs=2_000, accelerator=DEVICE, output_dir=output_dir)
    print('\nTraining U-Net on all the data')
    train_unet_overlap(crop_size=64, stride=32, images=images, label_images=label_images, epochs=2_000, batch_size=16, accelerator=DEVICE)

    print('\nPredicting unlabelled data with CNN')
    ckpt_path_cnn = list((output_dir / 'cnn_results_full').glob('**/*.ckpt'))[-1]
    model = LitCNN.load_from_checkpoint(ckpt_path_cnn)
    model.eval()
    predict_all_images_cnn(
        paths=validation_x_paths, model=model, batch_size=64,
        cmap=cmap, norm=norm, classnames=classnames, rgb=rgb,
        accelerator=DEVICE, output_dir=output_dir
    )

    print('\nPredicting unlabelled data with UNet')
    ckpt_path_unet = list((output_dir / 'unet_results_overlap').glob('**/*.ckpt'))[-1]
    model = LitUNet.load_from_checkpoint(ckpt_path_unet)
    model.eval()
    predict_all_images_unet(
        paths=validation_x_paths, size=64,
        model=model, cmap=cmap, norm=norm, classnames=classnames, rgb=rgb,
        output_dir=output_dir
    )


if __name__ == '__main__':
    main()
    

##########################################################
# COMMENTARY
##########################################################

# TRANSFORMS WILL GIVE YOU "WORSE" RESULTS IN THIS DATASET!
# If for example there's forest at the top of the (entire) image,
# and below the forest, it's all urban,
# then rotating a patch of it will produce an image that doesn't exist in the dataset
# and it would not help us predict other patches from the Same image.
# It's all because the train and the test set are very dependent on one another

# Using sliding window with stride equal to window size
# so that no part from the training images exist in the validation images.
# If overlap is allowed both train and validation accuracy reach ~100%,
# but without overlap, validation accuracy drops to about 80%!
# For example consider a test image with a city in its center,
# and that city appearing in the corners of some training images.

# Patch wise models do well because of overlap. Two neighboring pixels
# produce essentially indentical patches, while they are also very likely
# to have the same labels die to local "continuity" of the image.
# If one of these two pixel is in the train set and the other is in the test set, then we have huge overlap.

# Element wise models also perform really well because of local image "continuity".

# Of all the models, only the segmentation model with no overlap
# is the one that it test somewhat realistically.

# %%
