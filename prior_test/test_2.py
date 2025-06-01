# Try to detect small objects using slicer inference of supervision

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


# Try baseline detector (YOLO11n)

model = YOLO("assets/models/yolo11n.pt")
image = cv2.imread("assets/data/big_parking.png")

results = model(image)[0]
detections = sv.Detections.from_ultralytics(results)

box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated_image = box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

cv2.imwrite("prior_test/outputs_2/yolo11n_output.png", annotated_image)

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

cv2.imwrite("prior_test/outputs_2/supervision_output.png", annotated_image)

# Conclusion:
#   The base YOLO11n cannot do anything well (no detection).
#   In contrast, although supervision could detect more objects in image, almost all of them is not correct!
#   -> The idea using a slicing window is also not good.
