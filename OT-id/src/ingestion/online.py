"""
Online video stream ingestion via yt-dlp.

Supports any URL that yt-dlp handles: YouTube, Twitch live, direct HLS/DASH,
public RTMP streams, NASA TV, etc.

Usage:
    with OnlineSource("https://www.youtube.com/watch?v=LIVE_ID") as src:
        for frame in src.frames():
            ...
"""
from __future__ import annotations
import time
from typing import Generator

import cv2

from .base import VideoSource, Frame


def _resolve_stream_url(url: str) -> str:
    """Use yt-dlp to extract the direct streamable URL."""
    try:
        import yt_dlp
    except ImportError:
        raise ImportError("yt-dlp is required for online sources: pip install yt-dlp")

    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        # Live streams have a manifest_url or direct url
        if "url" in info:
            return info["url"]
        # Pick highest quality format with a direct URL
        for fmt in reversed(info.get("formats", [])):
            if fmt.get("url") and fmt.get("vcodec") != "none":
                return fmt["url"]
    raise ValueError(f"Could not extract stream URL from: {url}")


class OnlineSource(VideoSource):
    """
    Video source for any URL supported by yt-dlp.

    Args:
        uri:        Public video/stream URL
        frame_skip: Process every Nth frame (default 10 — online streams
                    often run at 30-60fps; no need to analyse every frame)
    """

    def __init__(self, uri: str, frame_skip: int = 10):
        super().__init__(uri, frame_skip)
        self._stream_url: str = ""
        self._cap = None

    def open(self) -> None:
        print(f"[Online] Resolving stream URL for: {self.uri}")
        self._stream_url = _resolve_stream_url(self.uri)
        print(f"[Online] Got stream URL, opening with OpenCV…")
        self._cap = cv2.VideoCapture(self._stream_url)
        if not self._cap.isOpened():
            raise ConnectionError(f"OpenCV could not open resolved stream: {self._stream_url[:80]}…")
        self._open = True

    def frames(self) -> Generator[Frame, None, None]:
        idx = 0
        while True:
            ret, img = self._cap.read()
            if not ret:
                break   # stream ended or stalled

            idx += 1
            if idx % self.frame_skip != 0:
                continue

            yield Frame(
                image=img,
                timestamp=time.time(),
                source_uri=self.uri,
                frame_index=idx,
                # No GPS for online sources
            )

    def close(self) -> None:
        if self._cap:
            self._cap.release()
        self._open = False
