"""
Local video file ingestion — useful for testing and offline analysis
of recorded drone footage.
"""
from __future__ import annotations
import time
from typing import Generator

import cv2

from .base import VideoSource, Frame


class FileSource(VideoSource):
    """
    Reads a local video file (mp4, avi, mkv, etc.).

    Args:
        uri:        Absolute or relative path to the video file
        frame_skip: Process every Nth frame (default 5)
    """

    def __init__(self, uri: str, frame_skip: int = 5):
        super().__init__(uri, frame_skip)
        self._cap = None
        self._fps: float = 30.0

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.uri)
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video file: {self.uri}")
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._open = True

    def frames(self) -> Generator[Frame, None, None]:
        idx = 0
        start_time = time.time()
        while True:
            ret, img = self._cap.read()
            if not ret:
                break

            idx += 1
            if idx % self.frame_skip != 0:
                continue

            # Reconstruct a plausible timestamp from frame position
            ts = start_time + (idx / self._fps)

            yield Frame(
                image=img,
                timestamp=ts,
                source_uri=self.uri,
                frame_index=idx,
            )

    def close(self) -> None:
        if self._cap:
            self._cap.release()
        self._open = False
