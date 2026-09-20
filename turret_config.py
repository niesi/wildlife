"""
Central configuration of the turret app.

Data only: no logic, no camera, no hardware. The app reads a single
``config.json``; it never writes it (edit by hand). Unknown keys are
rejected so typos in the JSON do not fail silently.
"""

from dataclasses import dataclass, field, fields
import json
from pathlib import Path


def reject_unknown_keys(section: str, data: dict, dataclass_type) -> None:
    """Raise ValueError when ``data`` contains keys the dataclass does not know."""
    if not isinstance(data, dict):
        raise ValueError(f"config section {section!r} must be an object, got {type(data).__name__}")
    known = {f.name for f in fields(dataclass_type)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown keys in section {section!r}: {', '.join(unknown)}")


@dataclass
class VideoConfig:
    """Which frames to process and where they come from."""

    source: str = "synthetic"          # video_file | webcam | synthetic
    path: str = None                   # video file path (source="video_file")
    webcam_index: int = 0              # camera index (source="webcam")
    sprite: str = None                 # transparent PNG/JPG (source="synthetic")
    loop: bool = True                  # restart video_file at the end
    show_motion: bool = True           # tint detected motion pixels red

    def __post_init__(self):
        if self.source not in ("video_file", "webcam", "synthetic"):
            raise ValueError(f"video.source must be video_file, webcam or synthetic, got {self.source!r}")
        if self.source == "video_file" and not self.path:
            raise ValueError("video.path is required when video.source is 'video_file'")

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "path": self.path,
            "webcam_index": self.webcam_index,
            "sprite": self.sprite,
            "loop": self.loop,
            "show_motion": self.show_motion,
        }


@dataclass
class PerceptionConfig:
    """Motion detection, classification and tracking thresholds."""

    min_area: int = 1200               # smallest accepted motion region (px)
    max_area: int = 50000              # largest accepted motion region (px)
    min_animal_area: int = 1500        # crop area the mock classifier accepts
    reclassify_every: int = 60         # tracker re-classification interval (frames)
    downscale_width: int = 320         # motion is detected at this width (0 = full resolution)

    def __post_init__(self):
        for name in ("min_area", "max_area", "min_animal_area"):
            if getattr(self, name) <= 0:
                raise ValueError(f"perception.{name} must be positive, got {getattr(self, name)!r}")
        if self.min_area > self.max_area:
            raise ValueError(f"perception.min_area must not exceed max_area ({self.min_area!r} > {self.max_area!r})")
        if self.reclassify_every < 1:
            raise ValueError(f"perception.reclassify_every must be at least 1, got {self.reclassify_every!r}")
        if self.downscale_width < 0:
            raise ValueError(f"perception.downscale_width must not be negative, got {self.downscale_width!r}")

    def to_dict(self) -> dict:
        return {
            "min_area": self.min_area,
            "max_area": self.max_area,
            "min_animal_area": self.min_animal_area,
            "reclassify_every": self.reclassify_every,
            "downscale_width": self.downscale_width,
        }


@dataclass
class AppConfig:
    """Everything the turret app needs: video input and perception."""

    video: VideoConfig = field(default_factory=VideoConfig)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)

    def to_dict(self) -> dict:
        return {"video": self.video.to_dict(), "perception": self.perception.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        """Build a config from a dict; unknown keys raise ValueError."""
        if not isinstance(data, dict):
            raise ValueError(f"config must be an object, got {type(data).__name__}")
        unknown = sorted(set(data) - {"video", "perception"})
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        video_data = data.get("video", {})
        perception_data = data.get("perception", {})
        reject_unknown_keys("video", video_data, VideoConfig)
        reject_unknown_keys("perception", perception_data, PerceptionConfig)
        return cls(video=VideoConfig(**video_data), perception=PerceptionConfig(**perception_data))

    @classmethod
    def load(cls, path) -> "AppConfig":
        """Read a config from a UTF-8 JSON file (json.JSONDecodeError on bad input)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
