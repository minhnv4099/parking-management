#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

from src.engine.task import TaskConfig


class Trainer:
    def __init__(self, model):
        self.model = model

    def train(self, train_cfg: TaskConfig):
        self.model.train(**train_cfg.__dict__)