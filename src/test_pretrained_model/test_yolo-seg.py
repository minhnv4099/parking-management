#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

# Try using YOLO segmentation model to produce masks of objects

import os
import sys

if __name__ == "__main__":
    img_path = sys.argv[1]
    out_dir = sys.argv[2]
    assert os.path.isfile(img_path), FileNotFoundError(img_path)
    os.makedirs(out_dir, exist_ok=True)

    from ultralytics import YOLO
    import cv2

    img = cv2.imread(img_path)

    # Try different pretrained models: smaller to larger
    for model_name in ("yolo11n-seg", ):
        model = YOLO(model="models/pretrained/{}.pt".format(model_name))
        img_size = max(img.shape)
        # With different images size: default size, original size, double size
        for imgsz, f_name in zip((640, img_size, 2 * img_size), ("default", "origin", "double")):
            name = f"{f_name}"
            model.predict(
                source=img_path,
                stream=False,
                save=True,
                project=out_dir,
                name=name,
                imgsz=imgsz,
            )
