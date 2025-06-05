#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#
from dataclasses import dataclass, field
from .config import ModelConfig


@dataclass(init=True)
class YoloConfig(ModelConfig):
    ...
