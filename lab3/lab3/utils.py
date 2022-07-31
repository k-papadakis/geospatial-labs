from typing import Dict, List
from pathlib import Path
import csv

from .datasets import PatchDatasetPostPad, CroppedDataset

import numpy as np
import rasterio
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import ConfusionMatrixDisplay

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl



##########################################################
# INPUT/OUTPUT FUNCTIONS
##########################################################
def load_image(src_path):
    with rasterio.open(src_path) as src:
            image = src.read()
    # Map integers to floats in [0, 1]    
    max_val = np.iinfo(image.dtype).max
    image = image.astype(np.float32) / max_val
    return image


def load_label(src_path):
    with rasterio.open(src_path) as src:
        label = src.read(1)
    # 0 is the unknown class, so we remap it to -1
    # 0, 1, ... , k, --> -1, 0, ..., k-1
    label = label.astype(np.int64) - 1
    return label
    
    
def read_data(image_paths, label_paths):
    """Loads the images and labels in two separate lists."""
    images = []
    labels = []
    for x_path, y_path in zip(image_paths, label_paths):
        x_img = load_image(x_path)
        y_img = load_label(y_path)
        images.append(x_img)
        labels.append(y_img)
    return images, labels


def read_info(path) -> Dict[str, List]:
    """Read class colors and names"""
    with open(path) as f:
        reader = csv.DictReader(f)
        info = {field: [] for field in reader.fieldnames}
        for record in reader:
            for name in reader.fieldnames:
                info[name].append(record[name])
    return info


##########################################################
# PLOTTING FUNCTIONS
##########################################################
def display_image_and_label(
    image, label, cmap, norm, classnames, rgb,
    title=None, axs=None,
):
    if axs is None:
        fig, axs = plt.subplots(nrows=2, figsize=(16, 6))
    image = image[rgb, ...].transpose(1, 2, 0)
    image = image / image.max()
    axs[0].imshow(image)
    axs[1].imshow(label, cmap=cmap, norm=norm)
    axs[0].set_axis_off()
    axs[1].set_axis_off()
    
    cbar = plt.gcf().colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axs, shrink=0.9, location='right'
    )
    cbar.set_ticks(np.arange(cmap.N) - 0.5)
    cbar.set_ticklabels(classnames)
    plt.suptitle(title)
    
    return axs


def display_patch(dataset, idx, rgb, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    patch, label = dataset[idx]
    img = patch[rgb, ...]
    img = np.transpose(img, (1, 2, 0))
    ax.imshow(img)
    ax.set_axis_off()
    return ax


def display_segmentation(dataset, idx, cmap, norm, names, rgb, n_repeats=1, axs=None):
    if axs is None:
        fig, axgrid = plt.subplots(nrows=n_repeats, ncols=2, figsize= (4, n_repeats))
        
    img, label = dataset[idx]
    img = img[rgb, ...]
    img = img / img.max()
    img = np.transpose(img, (1, 2, 0))
    
    for ax1, ax2 in axgrid:
        ax1.imshow(img)
        ax2.imshow(label, cmap=cmap, norm=norm)
        ax1.set_axis_off()
        ax2.set_axis_off()
    
    cbar = plt.gcf().colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axgrid, shrink=0.9, location='right'
    )
    cbar.set_ticks(np.arange(cmap.N) - 0.5)
    cbar.set_ticklabels(names)
    
    return axgrid


##########################################################
# IMAGE PREDICTION FUNCTIONS
##########################################################

def evaluate_model(trainer, dataset_test, names, output_dir='output'):
    """Test a model and display the accuracy, the dice loss and the confusion matrix.
    This function assumes a very specific structure of the input.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    best_model = trainer.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    trainer.test(best_model, dataloaders=dataset_test)
    acc = best_model.test_accuracy.compute()
    confusion_matrix = best_model.test_confusion_matrix.compute().cpu().numpy()
    
    title = type(best_model).__name__
    fig, ax = plt.subplots(figsize=(9, 9))
    ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=names,
    ).plot(xticks_rotation=90, ax=ax, colorbar=False)
    ax.set_title(f'{title} (acc={acc : .2%})')
    plt.savefig(output_dir / f'{title}_confmat.png')
    return ax


def predict_image_cnn(
    src_path, dst_path, model, batch_size,patch_size=15, accelerator='cpu'
):
    # Load from disk
    image = load_image(src_path)
    dataset = PatchDatasetPostPad(image, None, patch_size=patch_size)
    loader = DataLoader(dataset, batch_size=batch_size)
    
    # Predict
    trainer = pl.Trainer(accelerator=accelerator)
    preds = trainer.predict(model, loader)
    preds = torch.cat(preds)
    preds = preds.reshape(dataset.image.shape[-2:])
    
    # Write to disk
    with rasterio.open(
        dst_path, 'w',
        height=dataset.image.shape[-2],
        width=dataset.image.shape[-1],
        count=1,
        dtype=rasterio.uint8,
        driver='Gtiff',
    ) as dst:
        dst.write(preds.numpy().astype(rasterio.uint8)[np.newaxis, ...])
        
        
def predict_all_images_cnn(
    paths, model, batch_size,
    cmap, norm, classnames, rgb, suffix='_cnn', accelerator='cpu',
    output_dir='output',
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for src_path in paths:
        src_path = Path(src_path)
        tiff_dst_path = output_dir / Path(src_path.stem + f'_pred{suffix}.tif')
        png_dst_path = output_dir / Path(src_path.stem + f'_pred{suffix}.png')
        
        predict_image_cnn(
            src_path, tiff_dst_path, model,
            batch_size, accelerator=accelerator
        )
        
        image = load_image(src_path)
        label = load_label(tiff_dst_path)
        display_image_and_label(
            image=image, label=label, cmap=cmap, norm=norm, classnames=classnames,
            rgb=rgb, title=src_path.stem + suffix
        )
        
        plt.savefig(png_dst_path)
        

def predict_image_unet(src_path, dst_path, size, model):
    # Load from disk
    image = load_image(src_path)
    dataset = CroppedDataset(image, None, size, size, use_padding=True)
    loader = DataLoader(dataset)
    height = dataset.image.shape[-2]
    width = dataset.image.shape[-1]
    pred_image = np.full((1, height, width), -1, dtype=np.int64)
    model.eval()
    
    for idx, x in enumerate(loader):
        logits = model(x)
        pred_patch = torch.argmax(logits, 1).numpy()
        min_i, min_j, max_i, max_j = dataset.get_bounds(idx)
        pred_image[0, min_i:max_i, min_j:max_j] = pred_patch
    
    p = dataset.padding
    s = (
        slice(p[0][0], -p[0][1] if p[0][1] > 0 else None),
        slice(p[1][0], -p[1][1] if p[1][1] > 0 else None)
    )
    pred_image = pred_image[:, s[0], s[1]]
    
    # Write to disk
    with rasterio.open(
        dst_path, 'w',
        height=dataset.image.shape[-2],
        width=dataset.image.shape[-1],
        count=1,
        dtype=rasterio.uint8,
        driver='Gtiff',
    ) as dst:
        dst.write((pred_image + 1).astype(rasterio.uint8))
            

def predict_all_images_unet(
    paths, size, model,
    cmap, norm, classnames, rgb, suffix='_unet',
    output_dir='output'
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for src_path in paths:
        src_path = Path(src_path)
        tiff_dst_path = output_dir / Path(src_path.stem + f'_pred{suffix}.tif')
        png_dst_path = output_dir / Path(src_path.stem + f'_pred{suffix}.png')
        
        predict_image_unet(src_path, tiff_dst_path, size, model)
        
        image = load_image(src_path)
        label = load_label(tiff_dst_path)
        display_image_and_label(
            image=image, label=label,
            cmap=cmap, norm=norm, classnames=classnames,
            rgb=rgb, title=src_path.stem + suffix
        )
        
        plt.savefig(png_dst_path)
