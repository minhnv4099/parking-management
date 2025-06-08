#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT License.

import sys
import os
sys.path.append(os.getcwd())
import logging
import warnings
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(os.path.basename(__file__))


@hydra.main(config_path="../configs/", config_name="train", version_base="1.1")
def main(cfg: DictConfig):
    logger.info(f"override config: {cfg}")
    # logger.info("check paths:")
    # for k in cfg.paths:
    #     logger.info(f"{' ' * 6}{k}{' ' * (15 - len(k))}{cfg.paths[k]}")

    logger.info(f"data: {cfg.data}")

    logger.info("instantiate model")
    model_cls = instantiate(cfg.model)
    model = model_cls()

    logger.info("start training")
    model.train()
    logger.info("finish training")


if __name__ == '__main__':
    main()