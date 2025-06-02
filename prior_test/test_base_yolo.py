#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

# Test with various pretrained YOLO11 model when detecting objects with different size of images.
# Particularly default size: 640 and original size of images.

import sys
import os.path
import cv2


if __name__ == "__main__":
    f_path = sys.argv[1]
    assert os.path.isfile(f_path) or os.path.isdir(f_path), FileNotFoundError(f_path)

    img = cv2.imread(f_path)

    from ultralytics import YOLO

    # Try different pretrained models: smaller to larger
    for model_name in ("yolo11n", "yolo11m", "yolo11l"):
        model = YOLO(model="models/pretrained/{}.pt".format(model_name))
        img_size = max(img.shape)
        # With different images size: default size, original size, double size
        for imgsz, f_name in zip((640, img_size, 2*img_size), ("default", "origin", "double")):
            name = f"{f_name}"
            model.predict(
                source=f_path,
                stream=False,
                save=True,
                project=f"prior_test/outputs_1/{model_name}/",
                name=name,
                imgsz=imgsz,
            )

# Conclusion:
#   Regardless of using larger model or/and larger image size, the generalization error is too poor.
