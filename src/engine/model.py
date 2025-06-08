#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

from torch import nn
from dataclasses import dataclass


@dataclass
class ModelConfig:
    ...


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_name = None  # model name
        self.model = None  # model object

        self.trainer = None  # trainer object
        self.predictor = None  # reuse predictor

        self.ckpt_path = None
        self.ckpt = {}  # if loaded from *.pt
        self.cfg = None  # if loaded from *.yaml

    def __call__(
        self,
        source: str,
        *args, **kwargs
    ):
        return self.predict(source, *args, **kwargs)

    def preprocess(self, *args, **kwargs):
        ...

    def predict(self, source, *args, **kwargs):
        ...

    def postprocess(self, *args, **kwargs):
        ...

    def train(self, *args, **kwargs):
        ...
