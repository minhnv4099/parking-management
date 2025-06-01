# Try to detect small objects using another sclicing approach: SAHI

from sahi import AutoDetectionModel
from sahi.predict import get_prediction, get_sliced_prediction

# Use standard YOLO11n
detection_model = AutoDetectionModel.from_pretrained(
    model_path="assets/models/yolo11n.pt",
    model_type="ultralytics",
    confidence_threshold=0.35,
    device="mps",
)

result = get_prediction("assets/data/small_parking.png", detection_model)

result.export_visuals(
    export_dir="prior_test/outputs_3/",
    hide_conf=True,
    file_name="standard_yolo11_3"
)

# Use sliced prediction

result = get_sliced_prediction(
    "assets/data/small_parking.png",
    detection_model,
    slice_height=256,
    slice_width=256,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2
)

result.export_visuals(
    export_dir="prior_test/outputs_3/",
    hide_conf=True,
    file_name="sliced_yolo11_3"
)

# Conclusion:
#   For the recommended image, the sliced model performs better than standard yolo.
#   However, for maybe strange images, the former model can only detect some object but incorrecly.
#   Although for small parking, the model can detect all, they are not wrong predictions.
#   -> Using slicing inference seems not a good choice.
