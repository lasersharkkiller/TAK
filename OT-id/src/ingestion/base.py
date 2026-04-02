"""
Abstract base for all video ingestion sources.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generator, Optional
import time


@dataclass
class Frame:
    """A single decoded video frame plus optional geolocation metadata."""
    image: object          # numpy ndarray (BGR, OpenCV)
    timestamp: float       # Unix epoch seconds
    source_uri: str
    lat: Optional[float] = None   # from CoT/drone telemetry if available
    lon: Optional[float] = None
    alt_m: Optional[float] = None
    frame_index: int = 0


class VideoSource(ABC):
    """
    Abstract video source.  Subclass and implement open(), frames(), close().
    Use as a context manager:

        with RTSPSource("rtsp://...") as src:
            for frame in src.frames():
                process(frame)
    """

    def __init__(self, uri: str, frame_skip: int = 5):
        self.uri = uri
        self.frame_skip = frame_skip   # process every Nth frame
        self._open = False

    @abstractmethod
    def open(self) -> None:
        """Open the underlying stream / file."""

    @abstractmethod
    def frames(self) -> Generator[Frame, None, None]:
        """Yield Frame objects indefinitely (or until stream ends)."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()
