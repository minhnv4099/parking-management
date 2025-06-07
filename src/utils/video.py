#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

import os
import cv2
from typing import Union, List, Iterable
import numpy as np


def extract_frames(
        video_path: str,
        out_dir: str = None,
        frac: Union[float, List[float]] = 0.0,
) -> tuple:
    """
    :param video_path: Path to video needed to extract frames
    :param out_dir:
    :param num_frame: Number of frames
    :param index: Index of frame
    :return: Path or list of paths to all individual extracted frames.
    :rtype: None
    """
    assert os.path.isfile(video_path), FileNotFoundError(video_path)
    video_name = video_path.split("/")[-1].split(".")[0]
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = count_frame(cap)

    frac_array = np.array(
        frac if isinstance(frac, Iterable) else [frac],
        dtype=np.float16
    )
    index_of_frame = (frac_array * frame_count).astype(np.int16)
    index_of_frame = np.clip(
        index_of_frame,
        a_min=0, a_max=frame_count-1,
    )
    if frac == -1: index_of_frame = list(range(frame_count))

    f_paths = list()

    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            if i in index_of_frame:
                frame_path = os.path.join(out_dir, f"{video_name}_frame_{i}th.jpg")
                cv2.imwrite(frame_path, frame)
                f_paths.append(frame_path)
        else:
            break

    cap.release()

    return out_dir, f_paths


def count_frame(video: Union[str|cv2.VideoCapture]) -> int:
    if not isinstance(video, cv2.VideoCapture):
        video = cv2.VideoCapture(video)
    assert video.isOpened()
    return int(video.get(cv2.CAP_PROP_FRAME_COUNT))


def get_fps(video: Union[str|cv2.VideoCapture]) -> int:
    if not isinstance(video, cv2.VideoCapture):
        video = cv2.VideoCapture(video)
    assert video.isOpened()
    return int(video.get(cv2.CAP_PROP_FPS))
