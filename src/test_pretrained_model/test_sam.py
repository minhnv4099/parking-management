#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

# Try using Segment Anything to produce masks of objects

import os
import sys
sys.path.append(os.curdir)
import argparse
import logging

from src.test_pretrained_model.config import SamConfig
from src.test_pretrained_model.model import SamPredictor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="",
        prog="Segment object with SAM"
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
        default='sam2.1_b',
        help='Model name (e.g. sam2, sam2.1, sam2.1_l)'
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
        action="store_true",
        help="Whether store output images/videos"
    )
    parser.add_argument(
        "--prompt-type",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--prompt-dir",
        required=False,
        default=None,
        help="Directory containing prompt files"
    )
    parser.add_argument(
        "--export-box",
        required=False,
        action="store_true",
        help="Whether export inference bbox",
    )
    parser.add_argument(
        "--box-dir",
        required=False,
        default=None,
        help="Directory that save bounding boxes output"
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    logger.info("construct config")
    predict_config = SamConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        project=args.project,
        prompt_dir=args.prompt_dir,
        export_box=args.export_box,
        box_dir=args.box_dir,
        save=args.save,
        device=args.device,
        prompt_type=args.prompt_type
    )
    logger.info("construct model")
    sam_predictor = SamPredictor(config=predict_config)

    logger.info("segment images")
    for image_path in args.image_paths:
        sam_predictor.predict(source=image_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
