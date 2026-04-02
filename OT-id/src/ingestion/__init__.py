from .base import VideoSource, Frame
from .rtsp import RTSPSource
from .online import OnlineSource
from .file import FileSource

def get_source(source_type: str, uri: str, **kwargs) -> VideoSource:
    """Factory: returns the right VideoSource for the given type."""
    sources = {
        "rtsp": RTSPSource,
        "tak": RTSPSource,   # TAK drone feeds are RTSP under the hood
        "online": OnlineSource,
        "file": FileSource,
    }
    cls = sources.get(source_type.lower())
    if cls is None:
        raise ValueError(f"Unknown source type '{source_type}'. Choose from: {list(sources)}")
    return cls(uri, **kwargs)
