#
#  Copyright (c) 2025 Van Minh Nguyen
#  Licensed under the MIT license
#

# Try to detect small objects using another sclicing approach: SAHI

import os
import sys
sys.path.append(os.curdir)
import argparse
import logging

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from src.test_pretrained_model.config import SahiConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="",
        prog="Detect object with Sahi"
    )

    parser.add_argument(
        "image_paths",
        nargs="+",
        action="store",
        help="Paths of images"
    )
    parser.add_argument(
        "--image-dir",
        required=False,
        default=None,
        type=str,
        help="Directory containing images or videos"
    )
    parser.add_argument(
        "--model-type",
        required=False,
        default="ultralytics",
        help='Model type (e.g. ultralytics, huggingface, torchvision)'
    )
    parser.add_argument(
        "--model-name",
        required=False,
        default='yolo11n',
        help='Model name (e.g. yolo11n, yolo11s, yolo11m)'
    )
    parser.add_argument(
        "--model-path",
        required=False,
        default=None,
        help="Path to model",
    )
    parser.add_argument(
        "--project",
        required=False,
        default=None,
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--device",
        required=False,
        default='cpu',
    )
    parser.add_argument(
        "--save",
        required=False,
        default=False,
        action="store_true",
        help="Whether store output images/videos"
    )
    parser.add_argument(
        "--conf_threshold",
        required=False,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--slice_height",
        required=False,
        default=256,
        help="Height of slicer"
    )
    parser.add_argument(
        "--slice_width",
        required=False,
        default=256,
        help="Width of slicer"
    )
    parser.add_argument(
        "--h_overlap",
        required=False,
        default=0.2,
        help="Overlap height ratio"
    )
    parser.add_argument(
        "--w_overlap",
        required=False,
        default=0.2,
        help="Overlap width ratio"
    )

    return parser.parse_args()

def main(args: argparse.Namespace):
    logger.info("construct config")
    config = SahiConfig(
        model_type=args.model_type,
        model_path=args.model_path,
        model_name=args.model_name,
        project=args.project,
        device=args.device,
        save=args.save,
        hide_conf=True,
        conf_threshold=args.conf_threshold,
        slice_width=args.slice_width,
        slice_height=args.slice_height,
        overlap_width_ratio=args.w_overlap,
        overlap_height_ratio=args.h_overlap,
    )

    logger.info("construct model")
    detection_model = AutoDetectionModel.from_pretrained(
        model_path=config.model_path,
        model_type=config.model_type,
        confidence_threshold=config.conf_threshold,
        device=config.device,
    )
    logger.info("detect images")
    for image_path in args.image_paths:
        result = get_sliced_prediction(
            image=image_path,
            detection_model=detection_model,
            slice_height=config.slice_height,
            slice_width=config.slice_width,
            overlap_height_ratio=config.overlap_height_ratio,
            overlap_width_ratio=config.overlap_width_ratio,
        )
        if config.save:
            file_name = os.path.splitext(os.path.basename(image_path))[0]
            os.makedirs(config.project, exist_ok=True)
            result.export_visuals(
                export_dir=config.project,
                hide_conf=config.hide_conf,
                file_name=f"{file_name}_s",
                rect_th=config.rect_th,
                text_size=config.text_size,
                hide_labels=config.hide_labels,
            )


if __name__ == "__main__":
    args = get_args()
    main(args)

# Conclusion:
#   For the recommended image, the sliced model performs better than standard yolo.
#   However, for maybe strange images, the former model can detect some object but wrongly.
#   Although for small parking, the model can detect all, they are not wrong predictions.
#   -> Using slicing inference seems not a good choice.
