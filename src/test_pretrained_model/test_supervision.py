#
#  Copyright (c) 2025 Van Minh Nguyen
#  Licensed under the MIT license
#

# Try to detect small objects using slicer inference of supervision
import sys
import os
sys.path.append(os.curdir)
import argparse
import cv2
import numpy as np
import supervision as sv
import logging

from ultralytics import YOLO
from src.test_pretrained_model.config import SupervisionConfig

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="",
        prog='Detect object by supervision'
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
    parser.add_argument(
        "--slice-wh",
        required=False,
        default=(256, 256),
        type=tuple,
        help="Width and height of slicer"
    )
    parser.add_argument(
        "--overlap-wh",
        required=False,
        default=(0.2, 0.2),
        type=tuple,
        help="overlap ratio width and height",
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    logger.info("construct config")
    config = SupervisionConfig(
        model_name=args.model_name,
        model_path=args.model_path,
        project=args.project,
        save=args.save,
        device=args.device,
        slice_wh=args.slice_wh,
        overlap_ratio_wh=args.overlap_wh,
        task=args.task,
    )

    logger.info("construct YOLO model")
    model = YOLO(
        model=config.model_path,
        task=config.task,
        verbose=True,
    )

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)

    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(256, 256),
        overlap_ratio_wh=(0.2, 0.2),
    )
    logger.info("detects images")
    for image_path in args.image_paths:
        image = cv2.imread(image_path)

        detections = slicer(image)
        box_annotator = sv.BoxAnnotator(
            thickness=config.thickness,
        )
        label_annotator = sv.LabelAnnotator(
            text_scale=config.text_scale,
            text_padding=config.text_padding,
            text_color=config.text_color,
            color=config.color,
        )

        annotated_image = box_annotator.annotate(
            scene=image, detections=detections)
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections)

        if config.save:
            f_name = os.path.splitext(os.path.basename(image_path))[0]
            os.makedirs(config.project, exist_ok=True)
            cv2.imwrite(
                os.path.join(config.project, f"{f_name}.jpg"),
                annotated_image,
            )

if __name__ == "__main__":
    args = get_args()
    main(args)

# Conclusion:
#   The base YOLO11n cannot also do anything well (no detection).
#   In contrast, although supervision could detect more objects in image, almost all of them is not correct.
#   -> The idea using a slicing window is also not good.
