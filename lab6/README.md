# MMDetection and MMRotate setup and usage

## Installation

MMRotate is based on MMDetection, which in turn is based on MMCV. MMCV is the the base of all [OpenMMLab projects](https://github.com/open-mmlab/).

To install MMDetection or MMRotate, a careful selection of package versions is required. See
[MMRotate installation](https://mmrotate.readthedocs.io/en/latest/install.html#installation),
[MMRotate FAQ](https://mmrotate.readthedocs.io/en/latest/faq.html),
[MMDetection installation](https://mmdetection.readthedocs.io/en/latest/get_started.html#installation),
[MMDetection FAQ](https://mmdetection.readthedocs.io/en/latest/faq.html),
[MMCV installation](https://mmcv.readthedocs.io/en/latest/get_started/installation.html),
and [Pytorch Installation](https://pytorch.org/get-started/previous-versions/). An [MMRotate Dockerfile](./mmr/Dockerfile) and an [MMDetection Dockerfile](./mmd/Dockerfile) are included to setup working environments (the official dockerfiles don't work).

To install with pip, the following works

```shell
pip install -U torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
    --extra-index-url https://download.pytorch.org/whl/cu113
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install mmdet==2.25.1 mmengine==0.1.0 shapely==1.8.4 tensorboard==2.10.0 jupyter==1.0.0
git clone https://github.com/open-mmlab/mmrotate.git
cd mmrotate
git checkout tags/v0.3.2
pip install -v -e .

```


After installation there should be an `mmrotate` or an `mmdetection` directory (git repo clone) inside the working directory. The most common usage is to `cd` inside this directory and do all the work inside it.

## Documentation

To learn how to use [MMDetection](https://mmdetection.readthedocs.io/en/latest/) or [MMRotate](https://mmrotate.readthedocs.io/en/latest/), it is recommended to start with [MMCV](https://mmcv.readthedocs.io/en/latest/) first.

There is also good tutorials in the [MMdetection demo directory](https://github.com/open-mmlab/mmdetection/tree/master/demo) and in the [MMRotate demo directory](https://github.com/open-mmlab/mmrotate/tree/main/demo).

Sadly, the documentation is lacking and looking up for answers on the internet usually won't return any relevant results. For this reason, it is often the case that one needs to browse the source code to understand the project's usage, even for basic things.

## MMRotate Basic Usage

### Setting up a DOTA-formatted dataset

[Download DOTA v1.0](https://captain-whu.github.io/DOTA/) or transform the labels of a different dataset into the [DOTA format](https://mmrotate.readthedocs.io/en/latest/tutorials/customize_dataset.html).

Create a `data` directory inside `mmrotate` containing the following directory structure

```none
DOTA
├── test
│   └── images
├── train
│   ├── images
│   └── labelTxt
└── val
    ├── images
    └── labelTxt
```

Note the everything in this section works for any dataset with annotations in the DOTA format, as long as the appropriate `classes` attribute is set in the configuration file (see below). The "DOTA" name is just a convention.

Produce a new dataset by cropping the original images into `1024 x 1024` patches with an overlap of `200` by running

```shell
python tools/data/dota/split/img_split.py \
  --base-json tools/data/dota/split/split_configs/ss_train.json

python tools/data/dota/split/img_split.py \
  --base-json tools/data/dota/split/split_configs/ss_val.json

python tools/data/dota/split/img_split.py \
  --base-json tools/data/dota/split/split_configs/ss_test.json
```

The following structure will be created under `mmrotate/data`

```
split_ss_dota/
├── test
│   ├── {datetime}.log
│   ├── annfiles
│   └── images
├── train
│   ├── {datetime}.log
│   ├── annfiles
│   └── images
└── val
    ├── {datetime}.log
    ├── annfiles
    └── images
```

### Creating a Configuration

For a configuration to work with the dataset, one needs to inherit from the [configs/\_base\_/dotav1.py](https://github.com/open-mmlab/mmrotate/blob/main/configs/_base_/datasets/dotav1.py) configuration and then set the appropriate paths. A simple configuration file for training can look like this

```python
# simple_config.py

_base_ = './configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'

# classes = ['foo', 'bar']  # Use this for a different DOTA-formatted dataset.
data_root = './data/split_ss_dota/'
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        img_prefix='./data/split_ss_dota/train/images/',
        ann_file='./data/split_ss_dota/train/annfiles/'),
    val=dict(
        img_prefix='./data/split_ss_dota/val/images/',
        ann_file='./data/split_ss_dota/val/annfiles/'),
    test=dict(
        img_prefix='./data/split_ss_dota/test/images/',
        ann_file='./data/split_ss_dota/test/images/'))

log_config = dict(
    hooks=[dict(type='TextLoggerHook'), dict(type='TensorboardLoggerHook')])

runner = dict(
    max_epochs=13)
seed = 42
gpu_ids = range(1)
device = 'cuda'

evaluation = dict(
    save_best='mAP')
```

Notice that the annotations' directory for the test set is configured to be the images' directory. This is due to an "interesting" implementation in the source code which requires it this way and is nowhere documented... (see [DOTADataset](https://github.com/open-mmlab/mmrotate/blob/c62f148fcc2c8253218e67f2277cb8770bcd7df0/mmrotate/datasets/dota.py#L57))

### Training and Evaluating

To train with a config, e.g. `simple_config.py`, run the following command

```shell
python tools/train.py simple_config.py
```

This will create checkpoints and logs inside `workdirs/simple_config/`.

To test using the best checkpoint from training, one can run the following

```shell
python tools/test.py \
    simple_config.py \
    `ls work_dirs/simple_config/best_mAP_epoch_*.pth` \
    --format-only \
    --eval-options submission_dir=work_dirs/simple_config_preds/ nproc=1
```

which will create predictions for the original images (by assembling predictions on the image patches) inside `work_dirs/simple_config_preds/` in the format used for the [DOTA-v1.0 Evaluation Server](https://captain-whu.github.io/DOTA/evaluation.html).

You can also use the custom made `predictor.py` file to run the model on images

```shell
python predictor.py --config ${CONFIG} --checkpoint ${CHECKPOINT} --imagedir {IMAGEDIR}
```
