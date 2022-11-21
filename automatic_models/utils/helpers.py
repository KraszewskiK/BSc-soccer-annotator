"""General helpers for video processing"""
import math
import cv2
from typing import Optional


def divide_video_into_frames(video_url: str,
                             output_folder: str,
                             desired_frequency: Optional[int] = None) -> None:
    """
    Divide video file into images with desired frequency rate
    :param video_url: path to video
    :param desired_frequency: number of frames per second generated by splitter, if N
    :param output_folder: location to save new images
    :return: None
    """
    try:
        capture = cv2.VideoCapture(video_url)
    except:
        print('Given folder has no video')
        return None

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not desired_frequency:
        desired_frequency = fps # if no desired frequency, we perform fps of initial video
    if (fps / desired_frequency) < 1:
        raise Exception("""Desired fps is bigger than initial fps. Please change desired_frequency parameter to smaller
                        value""")

    frame_number = 0
    iterator = 0

    while True:
        success, frame = capture.read()
        if success:
            if iterator == math.floor(fps / desired_frequency):
                cv2.imwrite(f'{output_folder}/frame_{frame_number}.jpg', frame)
                frame_number += 1
                iterator = 0
                print(frame_number)
            else:
                iterator += 1
        else:
            return None



if __name__ == '__main__':
    divide_video_into_frames(video_url='./../data/barcelona_valencia.mp4',
                             desired_frequency=2,
                             output_folder='./../data/barcelona_valencia_frames')
