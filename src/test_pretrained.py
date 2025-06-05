#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#
import sys
import os
sys.path.append(os.getcwd())
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs/", config_name="config", version_base="1.1")
def main(cfg: DictConfig):
    ...


if __name__ == '__main__':
    main()
