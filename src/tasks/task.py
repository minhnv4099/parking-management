#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TaskConfig:
    model_name: Optional[str] = field(
        default=None,
        metadata={"help": "name of model"},
    )
    model: Optional[str] = field(
        default=None,
        metadata={"help": "path of model"}
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
        metadata={"help": "e.g. train/predict/val"}
    )
    task: Optional[str] = field(
        default="detect",
        metadata={"help": ""}
    )
