#
#  Copyright (c) 2025  Van Minh Nguyen
#  Licensed under the MIT license
#

import os
import cv2
from typing import Union, List


def extract_frames(
        video_path: str,
        out_dir: str = None,
        num_frame: int = 1,
        index: Union[int, List[int]] = 0,
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
    i = 0
    while True:
        ret, frame = video.read()

        if ret:
            frame_path = os.path.join(out_dir, f"{video_name}_frame_{i}th.png")
            print(frame_path)
            cv2.imwrite(frame_path, frame)
            i += 1
            if i == num_frame and num_frame > 0:
                break
    video.release()

    return out_dir
