#
#  Copyright (c) 2025 Van Minh Nguyen
#  Licensed under the MIT license
#

# Try to detect small objects using slicer inference of supervision

import sys
import os

if __name__ == "__main__":
    f_path = sys.argv[1]
    f_name = f_path.split("/")[-1].split(".")[0]
    assert os.path.isfile(f_path) or os.path.isdir(f_path), FileNotFoundError(f_path)

    import cv2
    import numpy as np
    import supervision as sv
    from ultralytics import YOLO

    # Try base detector (YOLO11n)
    model_name = "yolo11n"

    model = YOLO(f"models/pretrained/{model_name}.pt")
    image = cv2.imread(f_path)

    results = model(image)[0]
    detections = sv.Detections.from_ultralytics(results)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    cv2.imwrite(f"prior_test/outputs_3/{f_name}_{model_name}.png", annotated_image)

    # Try Inference slicer from supervision
    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice)[0]
        return sv.Detections.from_ultralytics(result)


    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(256, 256),
        overlap_ratio_wh=(0.2, 0.2),
    )

    detections = slicer(image)

    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    annotated_image = box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    cv2.imwrite(f"prior_test/outputs_3/{f_name}_supervision.png", annotated_image)

# Conclusion:
#   The base YOLO11n cannot also do anything well (no detection).
#   In contrast, although supervision could detect more objects in image, almost all of them is not correct.
#   -> The idea using a slicing window is also not good.
