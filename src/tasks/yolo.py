#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#
import os
from dataclasses import dataclass, field
from typing import Optional
from .task import TaskConfig


@dataclass(init=True)
class YoloPredictionConfig(TaskConfig):
    task: Optional[str] = field(
        default=None,
        metadata={"help": "e.g detect/segment/track/..."}
    )
    image_size: Optional[int] = field(
        default=640,
        metadata={"help": "size of image"}
    )
    save_run: Optional[bool] = field(
        default=False,
        metadata={"help": "if set, allow runtime save output by default"}
    )
    model_task: Optional[str] = field(
        default='yolo',
        metadata={"help": "useful for naming"}
    )

    TASK_SUFFIX: dict = field(default_factory=lambda: {
        "detect": "",
        "seg": "-seg",
    })

    def __post_init__(self):
        if self.task is None:
            self.task = "detect"
        if self.mode is None:
            self.mode = "predict"
        if self.model_name is None:
            self.model_name = "yolo11n"
        self.model_task = 'yolo' + self.TASK_SUFFIX[self.task]
        if self.model is None:
            self.model = f"models/pretrained/{self.model_name + self.TASK_SUFFIX[self.task]}.pt"
            assert os.path.isfile(self.model), f"This checkpoint is not on machine '{self.model}"
        if self.project is None:
            self.project = f"assets/test_pretrained_results/{self.model_task}/{self.model_name}"


@dataclass
class YoloTrainingConfig(TaskConfig):
    model: Optional[str] = field(
        default=None,
        metadata={"help": "for defining the model structure or initializing weights."},
    )
    data: Optional[str] = field(
        default=None,
        metadata={"help": "path to the dataset configuration file (e.g., coco8.yaml)"}
    )
    epoches: Optional[int] = field(
        default=100,
        metadata={"help": "Total number of training epochs"}
    )
    time: Optional[float] = field(
        default=None,
        metadata={"help": "Maximum training time in hours. \
        If set, this overrides the epochs argument, allowing \
        training to automatically stop after the specified duration. \
        Useful for time-constrained training scenarios."}
    )
    patience: Optional[int] = field(
        default=100,
        metadata={"help": "Number of epochs to wait without improvement \
        in validation metrics before early stopping the training. \
        Helps prevent overfitting by stopping training when performance plateaus."}
    )
    batch: Optional[int] = field(
        default=16,
        metadata={"help": "bacth size"}
    )
    imgsz: Optional[int | list[int]] = field(
        default=640,
        metadata={"help": "Target image size for training"}
    )
    save: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables saving of training checkpoints and final model weights. \
        Useful for resuming training or model deployment"}
    )
    save_period: Optional[int] = field(
        default=-1,
        metadata={"help": "Frequency of saving model checkpoints, specified in epochs. \
        A value of -1 disables this feature. \
        Useful for saving interim models during long training sessions."}
    )
    cache: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables caching of dataset images in memory (True/ram), \
        on disk (disk), or disables it (False). Improves training speed by \
        reducing disk I/O at the cost of increased memory usage."}
    )
    device: Optional[int | str | list] = field(
        default=None,
        metadata={"help": "Specifies the computational device(s) for training: \
        a single GPU (device=0), multiple GPUs (device=[0,1]), CPU (device=cpu), \
        MPS for Apple silicon (device=mps), or auto-selection of most idle GPU (device=-1)\
         or multiple idle GPUs (device=[-1,-1])"}
    )
    project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the project directory where training outputs \
        are saved. Allows for organized storage of different experiments."}
    )
    name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the training run. Used for creating a subdirectory \
        within the project folder, where training logs and outputs are stored."}
    )
    exist_ok: Optional[bool] = field(
        default=False,
        metadata={"help": "If True, allows overwriting of an existing project/name\
        directory. Useful for iterative experimentation without needing \
        to manually clear previous outputs."}
    )
    pretrained: Optional[bool|str] = field(
        default=True,
        metadata={"help": "Determines whether to start training from a pretrained model.\
        Can be a boolean value or a string path to a specific model from which \
        to load weights. Enhances training efficiency and model performance."}
    )
    optimizer: Optional[str] = field(
        default=None,
        metadata={"help": "Choice of optimizer for training. Options include \
        SGD, Adam, AdamW, NAdam, RAdam, RMSProp etc., or auto for \
        automatic selection based on model configuration.\
        Affects convergence speed and stability."}
    )
    seed: Optional[int] = field(
        default=0,
        metadata={"help": "Sets the random seed for training, ensuring reproducibility \
        of results across runs with the same configurations."}
    )
    single_cls: Optional[bool] = field(
        default=False,
        metadata={"help": ""}
    )
    classes: Optional[list[int]] = field(
        default=None,
        metadata={"help": "Specifies a list of class IDs to train on. Useful for filtering "
                          "out and focusing only on certain classes during training."}
    )
    multi_scale: Optional[bool] = field(
        default=False,
        metadata={"help": ""},
    )
    resume: Optional[bool] = field(
        default=False,
        metadata={"help": "Resumes training from the last saved checkpoint. "
                          "Automatically loads model weights, optimizer state, "
                          "and epoch count, continuing training seamlessly."}
    )
    amp: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables Automatic Mixed Precision (AMP) training,"
                          " reducing memory usage and possibly speeding up "
                          "training with minimal impact on accuracy."}
    )
    fraction: Optional[float] = field(
        default=1.0,
        metadata={"help": "Specifies the fraction of the dataset to use for training."
                          " Allows for training on a subset of the full dataset, "
                          "useful for experiments or when resources are limited."}
    )
    profile: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables profiling of ONNX and TensorRT speeds during"
                          " training, useful for optimizing model deployment."}
    )
    freeze: Optional[int | list] = field(
        default=None,
        metadata={"help": "Freezes the first N layers of the model or specified layers "
                          "by index, reducing the number of trainable parameters. "
                          "Useful for fine-tuning or transfer learning."}
    )
    lr0: Optional[float] = field(
        default=0.1,
        metadata={"help": "Initial learning rate (i.e. SGD=1E-2, Adam=1E-3). "
                          "Adjusting this value is crucial for the optimization process, "
                          "influencing how rapidly model weights are updated."}
    )
    lr1: Optional[float] = field(
        default=0.01,
        metadata={"help": "Final learning rate as a fraction of the initial rate = (lr0 * lrf), "
                          "used in conjunction with schedulers to adjust the learning rate over time."}
    )
    momentum: Optional[float] = field(
        default=0.937,
        metadata={"help": "Momentum factor for SGD or beta1 for Adam optimizers, influencing "
                          "the incorporation of past gradients in the current update."}
    )
    weight_decay: Optional[float] = field(
        default=0.0005,
        metadata={"help": "L2 regularization term, penalizing large weights to prevent overfitting."}
    )
    warmup_epochs: Optional[float] = field(
        default=3.0,
        metadata={"help": "Number of epochs for learning rate warmup, gradually "
                          "increasing the learning rate from a low value to the"
                          " initial learning rate to stabilize training early on."}
    )
    warmup_momentum: Optional[float] = field(
        default=0.8,
        metadata={"helo": "Initial momentum for warmup phase, gradually adjusting to "
                          "the set momentum over the warmup period."}
    )
    warmup_bias_lr: Optional[float] = field(
        default=0.1,
        metadata={"help": "Learning rate for bias parameters during the warmup phase, "
                          "helping stabilize model training in the initial epochs."}
    )
    box: Optional[float] = field(
        default=7.5,
        metadata={"help": "Weight of the box loss component in the loss function, "
                          "influencing how much emphasis is placed on accurately "
                          "predicting bounding box coordinates."}
    )
    cls: Optional[float] = field(
        default=0.5,
        metadata={"help": "Weight of the classification loss in the total loss function, "
                          "affecting the importance of correct class prediction relative to other components."}
    )
    pose: Optional[float] = field(
        default=12.0,
        metadata={"help": "Weight of the pose loss in models trained for pose estimation, "
                          "influencing the emphasis on accurately predicting pose keypoints."}
    )
    kobj: Optional[float] = field(
        default=2.0,
        metadata={"help": "Weight of the keypoint objectness loss in pose estimation models, "
                          "balancing detection confidence with pose accuracy."}
    )
    nbs: Optional[int] = field(
        default=64,
        metadata={"help": "Nominal batch size for normalization of loss."}
    )
    overlap_mask: Optional[bool] = field(
        default=True,
        metadata={"help": ""}
    )
    mask_ratio: Optional[int] = field(
        default=4,
        metadata={"help": ""}
    )
    dropout: Optional[float] = field(
        default=0.0,
        metadata={"help": "Dropout rate for regularization in classification tasks, "
                          "preventing overfitting by randomly omitting units during training."}
    )
    val: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables validation during training, allowing for periodic "
                          "evaluation of model performance on a separate dataset."}
    )
    plots: Optional[bool] = field(
        default=False,
        metadata={"help": "Generates and saves plots of training and validation metrics, "
                          "as well as prediction examples, providing visual insights into "
                          "model performance and learning progression."}
    )

    def __post_init__(self):
        if self.task is None:
            self.task = "detect"
        if self.mode is None:
            self.mode = "train"
        if self.model_name is None:
            self.model_name = "yolo11n"
        if self.model is None:
            self.model = f"models/pretrained/{self.model_name}.pt"
            assert os.path.isfile(self.model), f"This checkpoint is not on machine '{self.model}"
        if self.project is None:
            self.project = os.getcwd()


#tensorboard --logdir ultralytics/runs # replace with 'runs' directory