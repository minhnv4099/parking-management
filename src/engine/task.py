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
        metadata={"help": "location to store outputs"}
    )
    save: Optional[bool] = field(
        default=False,
        metadata={"help": "if set, save outputs"}
    )
    device: Optional[str] = field(
        default="cpu",
        metadata={"help": "device to run model"}
    )
    mode: Optional[str] = field(
        default=None,
        metadata={"help": "e.g. train/predict"}
    )
    task: Optional[str] = field(
        default="detect",
        metadata={"help": ""}
    )