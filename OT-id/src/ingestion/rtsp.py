"""
RTSP / TAK drone video feed ingestion.

TAK notes:
  - Drone video comes in as an RTSP stream (typically announced via CoT <video> detail).
  - GPS telemetry arrives on a separate CoT UDP stream; bind them by timestamp.
  - Pass tak_cot_host/tak_cot_port to receive live CoT position updates.
"""
from __future__ import annotations
import time
import threading
import socket
import xml.etree.ElementTree as ET
from typing import Generator, Optional

import cv2

from .base import VideoSource, Frame


class CoTPositionListener(threading.Thread):
    """
    Listens on a UDP port for CoT XML events from a TAK server or drone
    and keeps the most recent lat/lon/alt updated.
    """
    def __init__(self, host: str, port: int):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.lat: Optional[float] = None
        self.lon: Optional[float] = None
        self.alt_m: Optional[float] = None
        self._stop_event = threading.Event()

    def run(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            sock.bind((self.host, self.port))
            sock.settimeout(1.0)
            while not self._stop_event.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                    self._parse_cot(data.decode("utf-8", errors="ignore"))
                except socket.timeout:
                    pass
        finally:
            sock.close()

    def _parse_cot(self, xml_str: str):
        try:
            root = ET.fromstring(xml_str)
            point = root.find("point")
            if point is not None:
                self.lat = float(point.get("lat", 0))
                self.lon = float(point.get("lon", 0))
                self.alt_m = float(point.get("hae", 0))
        except ET.ParseError:
            pass

    def stop(self):
        self._stop_event.set()


class RTSPSource(VideoSource):
    """
    Consumes an RTSP stream (TAK drone feed or network camera).

    Args:
        uri:            rtsp://user:pass@host:port/path
        frame_skip:     process every Nth frame (default 5 ≈ 6 fps on 30fps feed)
        tak_cot_host:   bind address for CoT UDP listener (None = no CoT)
        tak_cot_port:   UDP port for CoT telemetry (default 4242)
        reconnect_s:    seconds to wait before reconnect on drop (default 5)
    """

    def __init__(
        self,
        uri: str,
        frame_skip: int = 5,
        tak_cot_host: Optional[str] = None,
        tak_cot_port: int = 4242,
        reconnect_s: int = 5,
    ):
        super().__init__(uri, frame_skip)
        self._cap: Optional[cv2.VideoCapture] = None
        self._cot_listener: Optional[CoTPositionListener] = None
        self._reconnect_s = reconnect_s
        self._tak_cot_host = tak_cot_host
        self._tak_cot_port = tak_cot_port

    def open(self) -> None:
        self._cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
        if not self._cap.isOpened():
            raise ConnectionError(f"Cannot open RTSP stream: {self.uri}")
        self._open = True

        if self._tak_cot_host:
            self._cot_listener = CoTPositionListener(
                self._tak_cot_host, self._tak_cot_port
            )
            self._cot_listener.start()

    def frames(self) -> Generator[Frame, None, None]:
        idx = 0
        while True:
            if self._cap is None or not self._cap.isOpened():
                print(f"[RTSP] Stream lost, reconnecting in {self._reconnect_s}s…")
                time.sleep(self._reconnect_s)
                self.open()

            ret, img = self._cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            idx += 1
            if idx % self.frame_skip != 0:
                continue

            lat = self._cot_listener.lat if self._cot_listener else None
            lon = self._cot_listener.lon if self._cot_listener else None
            alt = self._cot_listener.alt_m if self._cot_listener else None

            yield Frame(
                image=img,
                timestamp=time.time(),
                source_uri=self.uri,
                lat=lat,
                lon=lon,
                alt_m=alt,
                frame_index=idx,
            )

    def close(self) -> None:
        if self._cap:
            self._cap.release()
        if self._cot_listener:
            self._cot_listener.stop()
        self._open = False
