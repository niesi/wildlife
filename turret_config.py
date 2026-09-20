"""
Central parameters of the water turret.

Data only: no logic, no camera, no hardware. Later phases read the same
objects, so thresholds can be tuned - and loaded from JSON - without
touching code.
"""

from dataclasses import dataclass, field, fields
import json
from pathlib import Path


@dataclass
class StateTimeouts:
    """Nominal durations of the timed states (seconds)."""

    aim_s: float = 2.0             # AIMING -> AIM_TIMEOUT
    spray_s: float = 0.3           # WATERING -> SPRAY_FINISHED
    verify_s: float = 5.0          # VERIFYING -> VERIFY_TIMEOUT
    target_confirm_s: float = 1.0  # TARGET_FOUND -> TARGET_LOST

    def __post_init__(self):
        for name in ("aim_s", "spray_s", "verify_s", "target_confirm_s"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")


@dataclass
class SafetyLimits:
    """Hard limits enforced by safety.SafetyGuard."""

    armed: bool = False                                     # never armed by default
    target_labels: tuple = ("cat",)                          # only this class may be sprayed
    min_confidence: float = 0.6                              # classifier confidence floor
    confirm_frames: int = 3                                  # frames needed for confirmation
    spray_zone: tuple = (0.05, 0.05, 0.95, 0.95)             # x0, y0, x1, y1 relative to the frame
    min_target_height: float = 0.02                           # closer/bigger = unsafe
    max_target_height: float = 0.9                            # farther/smaller = useless
    cooldown_s: float = 3.0                                   # pause between two sprays
    max_spray_s: float = 0.4                                  # hard cap per spray burst
    max_sprays_per_target: int = 2                            # budget for one cat
    max_sprays_per_hour: int = 12                             # rolling one hour window
    allowed_hours: tuple | None = (6, 22)                     # local time, end exclusive
    frame_timeout_s: float = 1.0                              # older frame = no spraying

    def __post_init__(self):
        self.target_labels = tuple(self.target_labels)
        self.spray_zone = tuple(float(v) for v in self.spray_zone)
        if self.allowed_hours is not None:
            self.allowed_hours = tuple(int(v) for v in self.allowed_hours)

        if not self.target_labels:
            raise ValueError("target_labels must not be empty")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within 0..1, got {self.min_confidence!r}")
        if self.confirm_frames < 1:
            raise ValueError(f"confirm_frames must be at least 1, got {self.confirm_frames!r}")
        if len(self.spray_zone) != 4:
            raise ValueError("spray_zone must contain x0, y0, x1, y1")
        x0, y0, x1, y1 = self.spray_zone
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(f"spray_zone must be ordered within 0..1, got {self.spray_zone!r}")
        if not 0.0 <= self.min_target_height < self.max_target_height <= 1.0:
            raise ValueError("min_target_height must be below max_target_height within 0..1")
        if self.cooldown_s < 0:
            raise ValueError(f"cooldown_s must not be negative, got {self.cooldown_s!r}")
        if self.max_spray_s <= 0:
            raise ValueError(f"max_spray_s must be positive, got {self.max_spray_s!r}")
        if self.max_sprays_per_target < 1:
            raise ValueError(f"max_sprays_per_target must be at least 1, got {self.max_sprays_per_target!r}")
        if self.max_sprays_per_hour < 1:
            raise ValueError(f"max_sprays_per_hour must be at least 1, got {self.max_sprays_per_hour!r}")
        if self.allowed_hours is not None:
            start, end = self.allowed_hours
            if not (0 <= start < end <= 24):
                raise ValueError(f"allowed_hours must be (start, end) with 0 <= start < end <= 24, got {self.allowed_hours!r}")
        if self.frame_timeout_s <= 0:
            raise ValueError(f"frame_timeout_s must be positive, got {self.frame_timeout_s!r}")


def reject_unknown_keys(section: str, data: dict, dataclass_type) -> None:
    """Raise ValueError when ``data`` contains keys the dataclass does not know."""
    if not isinstance(data, dict):
        raise ValueError(f"config section {section!r} must be an object, got {type(data).__name__}")
    known = {f.name for f in fields(dataclass_type)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown keys in section {section!r}: {', '.join(unknown)}")


@dataclass
class TurretConfig:
    """Everything the controller needs to run the turret."""

    state: StateTimeouts = field(default_factory=StateTimeouts)
    safety: SafetyLimits = field(default_factory=SafetyLimits)

    def __post_init__(self):
        if self.state.spray_s > self.safety.max_spray_s:
            raise ValueError(
                "state.spray_s must not exceed safety.max_spray_s "
                f"({self.state.spray_s!r} > {self.safety.max_spray_s!r})"
            )

    def to_dict(self) -> dict:
        return {
            "state": {
                "aim_s": self.state.aim_s,
                "spray_s": self.state.spray_s,
                "verify_s": self.state.verify_s,
                "target_confirm_s": self.state.target_confirm_s,
            },
            "safety": {
                "armed": self.safety.armed,
                "target_labels": list(self.safety.target_labels),
                "min_confidence": self.safety.min_confidence,
                "confirm_frames": self.safety.confirm_frames,
                "spray_zone": list(self.safety.spray_zone),
                "min_target_height": self.safety.min_target_height,
                "max_target_height": self.safety.max_target_height,
                "cooldown_s": self.safety.cooldown_s,
                "max_spray_s": self.safety.max_spray_s,
                "max_sprays_per_target": self.safety.max_sprays_per_target,
                "max_sprays_per_hour": self.safety.max_sprays_per_hour,
                "allowed_hours": None if self.safety.allowed_hours is None else list(self.safety.allowed_hours),
                "frame_timeout_s": self.safety.frame_timeout_s,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TurretConfig":
        """Build a config from a dict; unknown keys raise ValueError."""
        if not isinstance(data, dict):
            raise ValueError(f"config must be an object, got {type(data).__name__}")
        unknown = sorted(set(data) - {"state", "safety"})
        if unknown:
            raise ValueError(f"unknown config keys: {', '.join(unknown)}")
        state_data = data.get("state", {})
        safety_data = data.get("safety", {})
        reject_unknown_keys("state", state_data, StateTimeouts)
        reject_unknown_keys("safety", safety_data, SafetyLimits)
        return cls(state=StateTimeouts(**state_data), safety=SafetyLimits(**safety_data))

    def save(self, path) -> Path:
        """Write the config as UTF-8 JSON and return the path."""
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path) -> "TurretConfig":
        """Read a config from a UTF-8 JSON file (json.JSONDecodeError on bad input)."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))