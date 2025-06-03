#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

# Try using Segment Anything to produce masks of objects

import os
import sys
sys.path.append(os.curdir)

import argparse
import pickle

from ultralytics import SAM
from src.utils.data import make_points, write_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "image_paths",
        nargs="+",
    )
    parser.add_argument(
        "--point_path",
        required=False,
        default=True,
    )
    parser.add_argument(
        "--out_dir",
        required=False,
        default="prior_test/outputs_sam/",
    )

    return parser.parse_args()


def main(args: argparse.Namespace):
    predictor = SAM(model="models/pretrained/sam2.1_b.pt", )

    for image_path in args.image_paths:
        assert os.path.isfile(image_path)

        f_name = os.path.basename(image_path).split(".")[0]
        os.makedirs(args.out_dir, exist_ok=True)

        point_path = args.point_path

        if not point_path or not os.path.isfile(point_path):
            point_path = f"data/external/points/jsons/{f_name}.json"
            if not os.path.isfile(point_path):
                point_path = point_path.replace('json', 'txt')

        points = None
        if os.path.isfile(point_path):
            points = make_points(point_path)
            f_name = f"{f_name}_box"

        results = predictor.predict(
            source=image_path,
            save=False,
            points=points,
        )

        write_boxes(
            boxes=results[0].boxes.xywhn,
            text_path=f"data/iterm/boxes/{f_name}.txt",
        )

        results[0].plot(
            filename=os.path.join(args.out_dir, f"{f_name}_{len(os.listdir(args.out_dir))}.jpg"),
            save=True,
            line_width=1,
            labels=False,
            boxes=True,
            masks=True,
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
