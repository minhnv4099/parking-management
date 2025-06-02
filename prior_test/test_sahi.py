#
#  Copyright (c) 2025 Van Minh Nguyen
#  Licensed under the MIT license
#

# Try to detect small objects using another sclicing approach: SAHI

import sys
import os

if __name__ == "__main__":
    f_path = sys.argv[1]
    f_name = f_path.split("/")[-1].split(".")[0]
    assert os.path.isfile(f_path) or os.path.isdir(f_path), FileNotFoundError(f_path)

    from sahi import AutoDetectionModel
    from sahi.predict import get_prediction, get_sliced_prediction

    model_type = "yolo11n"

    detection_model = AutoDetectionModel.from_pretrained(
        model_path=f"models/pretrained/{model_type}.pt",
        model_type="ultralytics",
        confidence_threshold=0.35,
        device="mps",
    )

    # Detection using standard YOLO11n
    result = get_prediction(
        image=f_path,
        detection_model=detection_model,
    )

    result.export_visuals(
        export_dir="prior_test/outputs_2/",
        hide_conf=True,
        file_name=f"{f_name}_b_{model_type}"
    )

    # Detection using sliced prediction
    result = get_sliced_prediction(
        image=f_path,
        detection_model=detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    result.export_visuals(
        export_dir="prior_test/outputs_2/",
        hide_conf=True,
        file_name=f"{f_name}_s_{model_type}"
    )


# Conclusion:
#   For the recommended image, the sliced model performs better than standard yolo.
#   However, for maybe strange images, the former model can detect some object but wrongly.
#   Although for small parking, the model can detect all, they are not wrong predictions.
#   -> Using slicing inference seems not a good choice.
