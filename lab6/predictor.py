from pathlib import Path
import argparse

from mmdet.apis.inference import show_result_pyplot
from mmcv.runner import load_checkpoint
from mmrotate.apis.inference import inference_detector_by_patches
from mmrotate.models import build_detector
from mmcv.utils.config import Config
from mmrotate.datasets.dota import DOTADataset


def visualize_predictions(config_path, checkpoint_path, images, out_dir=None):
    
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
    
    config = Config.fromfile(config_path)
    model = build_detector(config.model)
    load_checkpoint(model, checkpoint_path)
    model.cfg = config  # `inference_detector_by_patches` requires it for `model.cfg.data.test.pipeline`
    model.CLASSES = DOTADataset.CLASSES
    model.to(config.device)
    model.eval()

    for image_path in sorted(Path(images).iterdir()):
        dets = inference_detector_by_patches(
            model=model, img=image_path, sizes=[1024], steps=[824], ratios=[0.5, 1.0, 2.0], merge_iou_thr=0.1
        )
        out_file = out_dir / f'{image_path.stem}_pred.{image_path.suffix}' if out_dir is not None else None
        show_result_pyplot(model, image_path, dets, score_thr=0.4, palette=DOTADataset.PALETTE, out_file=out_file)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize predictions of model on the DOTA Dataset")
    parser.add_argument('--config', help="The configuration used for training the model")
    parser.add_argument('--checkpoint', help="The checkpoint to use for loading the model")
    parser.add_argument('--images', help="The directory containing the DOTA images")
    parser.add_argument('--output', help="The outputs directory. If not specified plt.show() will be used")
    args = parser.parse_args()
    return args 


def main():
    args = parse_args()
    visualize_predictions(args.config, args.checkpoint, args.images, args.output)
    
    
if __name__ == '__main__':
    main()
    