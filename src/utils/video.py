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
) -> None:
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

    video = cv2.VideoCapture(video_path)
    assert video.isOpened()

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
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

    for i in range(frame_count):
        ret, frame = video.read()
        if ret:
            if i in index_of_frame:
                frame_path = os.path.join(out_dir, f"{video_name}_frame_{i}th.png")
                cv2.imwrite(frame_path, frame)
        else:
            break

    video.release()

    return out_dir
