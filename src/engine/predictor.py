#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

from src.engine.task import TaskConfig


class Predictor:
    def __init__(self, model):
        self.model = model

    def predict(self, source: str, predict_cfg: TaskConfig):
        return self.model.predict(
            source,
            **predict_cfg.__dict__
        )