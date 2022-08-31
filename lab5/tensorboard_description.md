
# Faster R-CNN trained on [Object Detection: Batteries, Dice, and Toy Cars](https://www.kaggle.com/datasets/markcsizmadia/object-detection-batteries-dices-and-toy-cars)

Student ID: 03400149

## Trainable layers during Phase

1. Backbone + RPN
2. Backbone + ROI Heads
3. RPN
4. ROI Heads

## Further Details

* `BackboneWithFPN` from `Resnet50` with `trainable_layers=2`
* `batch_size=8` in all phases
* `EarlyStopping` during phase 2 and 4, tracking the mAP with `patience=7` and `min_delta=0`
* `max_epochs=15` for phase 1, 2 and 3, and `30` for phase 4
* `lr`: see hparams
