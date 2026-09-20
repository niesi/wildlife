"""
Servo limits, camera geometry and persistence for the turret.

``Calibration`` holds the data the aiming layer and the safety checks need:
soft limits, centre position, movement rate, field of view and the aim
tolerance. It is stored next to the config as JSON, so a calibrated turret does
not have to repeat the procedure after a restart.

``Calibrator`` performs the hardware part (background warm-up and a limit
sweep). It only talks to injected objects (frame source, motion detector,
actuator), so the procedure itself is testable without a camera.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time

from turret_config import reject_unknown_keys


DEFAULT_CALIBRATION_FILE = "calibration.json"


def _check_limits(name, limits):
    try:
        low, high = limits
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair (low, high), got {limits!r}") from None
    if low >= high:
        raise ValueError(f"{name} must be ordered, got {limits!r}")
    return (float(low), float(high))


@dataclass
class Calibration:
    """Everything the aiming layer needs to know about the hardware."""

    pan_center: float = 0.0
    tilt_center: float = 0.0
    pan_limits: tuple = (-90.0, 90.0)
    tilt_limits: tuple = (-45.0, 45.0)
    deg_per_s: float = 60.0
    hfov_deg: float = 60.0
    vfov_deg: float = 36.0
    aim_tolerance_px: float = 24.0
    settle_frames: int = 3
    warmup_frames: int = 30
    created: str = ""
    notes: str = ""

    def __post_init__(self):
        self.pan_center = float(self.pan_center)
        self.tilt_center = float(self.tilt_center)
        self.pan_limits = _check_limits("pan_limits", self.pan_limits)
        self.tilt_limits = _check_limits("tilt_limits", self.tilt_limits)

        if not self.pan_limits[0] <= self.pan_center <= self.pan_limits[1]:
            raise ValueError(f"pan_center {self.pan_center!r} outside pan_limits {self.pan_limits!r}")
        if not self.tilt_limits[0] <= self.tilt_center <= self.tilt_limits[1]:
            raise ValueError(f"tilt_center {self.tilt_center!r} outside tilt_limits {self.tilt_limits!r}")
        if self.deg_per_s <= 0:
            raise ValueError(f"deg_per_s must be positive, got {self.deg_per_s!r}")
        for name in ("hfov_deg", "vfov_deg"):
            value = getattr(self, name)
            if not 0.0 < value < 180.0:
                raise ValueError(f"{name} must be between 0 and 180, got {value!r}")
        if self.aim_tolerance_px < 0:
            raise ValueError(f"aim_tolerance_px must not be negative, got {self.aim_tolerance_px!r}")
        if self.settle_frames < 1:
            raise ValueError(f"settle_frames must be at least 1, got {self.settle_frames!r}")
        if self.warmup_frames < 0:
            raise ValueError(f"warmup_frames must not be negative, got {self.warmup_frames!r}")

    def clamp(self, pan_deg, tilt_deg) -> tuple:
        """Clamp a position to the soft limits."""
        low_pan, high_pan = self.pan_limits
        low_tilt, high_tilt = self.tilt_limits
        return (max(low_pan, min(high_pan, float(pan_deg))),
                max(low_tilt, min(high_tilt, float(tilt_deg))))

    def to_dict(self) -> dict:
        data = asdict(self)
        data["pan_limits"] = list(self.pan_limits)
        data["tilt_limits"] = list(self.tilt_limits)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Calibration":
        """Build a calibration from a dict; unknown keys raise ValueError."""
        reject_unknown_keys("calibration", data, Calibration)
        return cls(**data)

    def save(self, path) -> Path:
        """Write the calibration as UTF-8 JSON and return the path."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "Calibration":
        """Read a calibration from a UTF-8 JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def default_calibration(wall_clock=datetime.now) -> Calibration:
    """Factory defaults, stamped with the time they were created."""
    return Calibration(created=wall_clock().isoformat(timespec="seconds"))


def field_of_view_deg(angle_deg, pixel_offset, frame_size) -> float:
    """
    Field of view from one measurement of the calibration sweep.

    ``angle_deg`` is the servo angle at which a marker sits ``pixel_offset`` px
    away from the optical axis in a frame of ``frame_size`` px (inverse of
    ``aim_controller.pixel_to_angle``).
    """
    if pixel_offset == 0:
        raise ValueError("pixel_offset must not be zero")
    if frame_size <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size!r}")
    angle = abs(math.radians(float(angle_deg)))
    if not 0.0 < angle < math.pi / 2.0:
        raise ValueError(f"angle_deg must be within -90 and 90 (excluding both), got {angle_deg!r}")
    return math.degrees(2.0 * math.atan(math.tan(angle) * frame_size / (2.0 * abs(pixel_offset))))


class Calibrator:
    """
    Runs the calibration procedure on injected objects.

    ``warm_up`` feeds frames into the motion detector so its background model
    has seen the empty garden before the turret starts searching. ``sweep``
    drives the head to both limits and back to the centre, which proves the soft
    limits are reachable without hitting a mechanical stop; a head that does not
    get there in time fails the calibration instead of silently measuring
    nonsense.
    """

    def __init__(self, actuator, calibration=None, motion=None,
                 clock=time.monotonic, wall_clock=datetime.now):
        self.actuator = actuator
        self.calibration = calibration if calibration is not None else Calibration()
        self.motion = motion
        self._clock = clock
        self._wall_clock = wall_clock
        self.visited = []

    def warm_up(self, source, frames=None) -> int:
        """Pull frames (default ``warmup_frames``) into the motion model."""
        wanted = self.calibration.warmup_frames if frames is None else int(frames)
        pulled = 0
        for _ in range(max(0, wanted)):
            ok, frame = source.get_frame()
            if not ok or frame is None:
                break
            if self.motion is not None:
                self.motion.detect(frame)
            pulled += 1
        return pulled

    def sweep(self):
        """Visit centre and both limits per axis; returns the reached positions."""
        calibration = self.calibration
        low_pan, high_pan = calibration.pan_limits
        low_tilt, high_tilt = calibration.tilt_limits
        waypoints = [
            (calibration.pan_center, calibration.tilt_center),
            (low_pan, calibration.tilt_center),
            (high_pan, calibration.tilt_center),
            (calibration.pan_center, low_tilt),
            (calibration.pan_center, high_tilt),
            (calibration.pan_center, calibration.tilt_center),
        ]

        self.visited = []
        for pan_deg, tilt_deg in waypoints:
            self.actuator.point_at(pan_deg, tilt_deg)
            if not self.actuator.wait_until_settled():
                raise RuntimeError(f"limit sweep did not reach pan={pan_deg}, tilt={tilt_deg}")
            self.visited.append(self.actuator.position())
        return list(self.visited)

    def run(self, source=None, warmup_frames=None, **overrides) -> Calibration:
        """
        Warm up (when a source is given), sweep and return the calibration.

        ``overrides`` replace single values of the stored calibration, e.g.
        ``hfov_deg`` measured with :func:`field_of_view_deg`.
        """
        if source is not None:
            self.warm_up(source, warmup_frames)
        self.sweep()

        values = asdict(self.calibration)
        values.update(overrides)
        values["created"] = self._wall_clock().isoformat(timespec="seconds")
        return Calibration(**values)