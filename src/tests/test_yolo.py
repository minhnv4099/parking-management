#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

# Test with various pretrained YOLO11 models when detecting objects

import sys
import os
sys.path.append(os.curdir)
import argparse
import logging

from ultralytics import YOLO
from src.configs.config import YoloConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=""
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
        "--model-name",
        required=False,
        default='yolo11n',
        help='Model type (e.g. yolo11n, yolo11s, yolo11m)'
    )
    parser.add_argument(
        "--model-path",
        required=False,
        default=None,
        help="Path to model",
    )
    parser.add_argument(
        "--task",
        required=False,
        default="detect",
        help="Task (e.g. detect, segment, ...)"
    )
    parser.add_argument(
        "--mode",
        required=False,
        default="predict",
        help="Mode (e.g. predict, train, val). This can only apply for 'predict'"
    )
    parser.add_argument(
        '--image-size',
        required=False,
        default=640,
        help="Image size"
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
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    logger.info("construct config")
    config = YoloConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        task=args.task,
        mode=args.mode,
        project=args.project,
        device=args.device,
        save=args.save,
        image_size=args.image_size,
    )

    logger.info("construct model")
    model = YOLO(
        model=config.model_path,
        task=config.task,
        verbose=True,
    )
    logger.info("Inference images")
    for image_path in args.image_paths:
        results = model.predict(
            source=image_path,
            stream=False,
            save=config.save_run,
            imgsz=config.image_size,
        )
        if config.save:
            f_name = os.path.splitext(os.path.basename(image_path))[0]
            os.makedirs(config.project, exist_ok=True)
            results[0].save(filename=os.path.join(config.project, f"{f_name}_{config.imgsz_predict}px.jpg"))
        else:
            results[0].show()


if __name__ == "__main__":
    args = get_args()
    main(args)

# Conclusion:
#   Regardless of using larger model or/and larger image size, the generalization error is too poor.
