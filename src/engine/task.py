#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskConfig:
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "name of model (e.g. yolo11n, yolo11l)"},
    )
    model: Optional[str] = field(
        default=None,
        metadata={"help": "name of model with extension (.pt)"}
    )
    ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "project-root-relative path to checkpoint"},
    )
    abs_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "absolute path to checkpoint"},
    )
    project: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the project directory where training outputs "
                          "are saved. Allows for organized storage of different experiments."}
    )
    save: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables saving of training checkpoints and final model weights. "
                          "Useful for resuming training or model deployment"}
    )
    save_run: Optional[bool] = field(
        default=False,
        metadata={"help": "if set, allow runtime save output by default"}
    )
    device: Optional[int | str | list] = field(
        default=None,
        metadata={"help": "Specifies the computational device(s) for training: "
                          "a single GPU (device=0), multiple GPUs (device=[0,1]), CPU (device=cpu), "
                          "MPS for Apple silicon (device=mps), or auto-selection of most idle GPU (device=-1)"
                          "or multiple idle GPUs (device=[-1,-1])"}
    )
    mode: Optional[str] = field(
        default=None,
        metadata={"help": "e.g. train/predict"}
    )
    task: Optional[str] = field(
        default=None,
        metadata={"help": "e.g detect/segment/track/..."}
    )
    EXCLUSIVE_KEYS: Optional[list] = field(
        default_factory=lambda: [],
        metadata={"help": ""},
    )
