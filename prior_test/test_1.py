# Test with various pretrained YOLO11 model when detecting objects with different size of images.
# Particularly default size: 640 and original size of images.

from ultralytics import YOLO

for model_name in ("yolo11n", "yolo11m", "yolo11l"):
    model = YOLO(model="assets/models/{}.pt".format(model_name))

    # predict with default size of model
    model.predict(
        source="assets/data/parking_lot.mp4",
        stream=False,
        save=True,
        project="prior_test/outputs/",
        name=f"{model_name}_default_size",
    )

    # predict with resizing (original size)
    model.predict(
        source="assets/data/parking_lot.mp4",
        stream=False,
        save=True,
        project="prior_test/outputs/",
        name=f"{model_name}_ori_size",
        imgsz=768,
    )

# Conclusion:
#   No matter using larger model or/and larger image size, the generalization error is too poor.
