"""
Aim control: turn a target position in the frame into pan/tilt angles.

Pure geometry and control logic - no camera and no servo imports. The
controller is proportional, rate limited (protects the servo gearbox), clamped
to the calibration limits and it counts how many consecutive frames the aim
error stayed inside the tolerance, which is what the state machine needs for
``AIM_SETTLED``.

Sign conventions: ``pan_deg`` grows to the right, ``tilt_deg`` grows upwards,
image ``y`` grows downwards. Flip ``invert_pan`` / ``invert_tilt`` when a servo
is mounted the other way round.
"""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PanTilt:
    """Commanded or measured servo position in degrees."""

    pan_deg: float
    tilt_deg: float


@dataclass(frozen=True)
class AimError:
    """How far the target is away from the optical axis."""

    dx_px: float
    dy_px: float
    pan_deg: float
    tilt_deg: float
    distance_px: float
    aimed: bool


@dataclass(frozen=True)
class AimResult:
    """Aim error, resulting command and settle progress."""

    error: AimError
    command: PanTilt
    aimed_frames: int
    settled: bool


def pixel_to_angle(dx_px, frame_size, fov_deg):
    """
    Angle of a pixel offset from the optical axis (pinhole model).

    ``frame_size`` is the width for horizontal offsets and the height for
    vertical ones, so one helper serves both axes.
    """
    if frame_size <= 0:
        raise ValueError(f"frame_size must be positive, got {frame_size!r}")
    if not 0.0 < fov_deg < 180.0:
        raise ValueError(f"fov_deg must be between 0 and 180, got {fov_deg!r}")
    half_fov = math.radians(fov_deg) / 2.0
    return math.degrees(math.atan(2.0 * dx_px / frame_size * math.tan(half_fov)))


class AimController:
    """
    Proportional pan/tilt controller for one target at a time.

    ``pan_limits`` / ``tilt_limits`` come from the calibration, ``gain`` scales
    the angle error into a servo step, ``max_rate_deg_s`` caps how fast the
    turret may move and ``settle_frames`` is the number of consecutive frames
    inside ``tolerance_px`` before the aim counts as settled.
    """

    def __init__(self, pan_limits=(-90.0, 90.0), tilt_limits=(-45.0, 45.0),
                 hfov_deg=60.0, vfov_deg=36.0, gain=0.6, max_rate_deg_s=60.0,
                 tolerance_px=24.0, settle_frames=3, invert_pan=False, invert_tilt=False):
        self.pan_limits = self._check_limits("pan_limits", pan_limits)
        self.tilt_limits = self._check_limits("tilt_limits", tilt_limits)
        if not 0.0 < hfov_deg < 180.0:
            raise ValueError(f"hfov_deg must be between 0 and 180, got {hfov_deg!r}")
        if not 0.0 < vfov_deg < 180.0:
            raise ValueError(f"vfov_deg must be between 0 and 180, got {vfov_deg!r}")
        if gain <= 0:
            raise ValueError(f"gain must be positive, got {gain!r}")
        if max_rate_deg_s <= 0:
            raise ValueError(f"max_rate_deg_s must be positive, got {max_rate_deg_s!r}")
        if tolerance_px < 0:
            raise ValueError(f"tolerance_px must not be negative, got {tolerance_px!r}")
        if settle_frames < 1:
            raise ValueError(f"settle_frames must be at least 1, got {settle_frames!r}")

        self.hfov_deg = float(hfov_deg)
        self.vfov_deg = float(vfov_deg)
        self.gain = float(gain)
        self.max_rate_deg_s = float(max_rate_deg_s)
        self.tolerance_px = float(tolerance_px)
        self.settle_frames = int(settle_frames)
        self.invert_pan = bool(invert_pan)
        self.invert_tilt = bool(invert_tilt)
        self._aimed_frames = 0

    @classmethod
    def from_calibration(cls, calibration, **overrides):
        """Build a controller from a calibration object (duck-typed)."""
        settings = {
            "pan_limits": calibration.pan_limits,
            "tilt_limits": calibration.tilt_limits,
            "hfov_deg": calibration.hfov_deg,
            "vfov_deg": calibration.vfov_deg,
            "tolerance_px": calibration.aim_tolerance_px,
            "settle_frames": calibration.settle_frames,
            "max_rate_deg_s": calibration.deg_per_s,
        }
        settings.update(overrides)
        return cls(**settings)

    @property
    def aimed_frames(self) -> int:
        """Consecutive frames the aim error stayed inside the tolerance."""
        return self._aimed_frames

    def error(self, target_center, frame_shape) -> AimError:
        """Pixel and angle offset of the target relative to the optical axis."""
        frame_h, frame_w = self._frame_size(frame_shape)
        cx, cy = target_center
        dx_px = float(cx) - frame_w / 2.0
        dy_px = float(cy) - frame_h / 2.0

        pan_deg = pixel_to_angle(dx_px, frame_w, self.hfov_deg)
        tilt_deg = -pixel_to_angle(dy_px, frame_h, self.vfov_deg)
        if self.invert_pan:
            pan_deg = -pan_deg
        if self.invert_tilt:
            tilt_deg = -tilt_deg

        distance_px = math.hypot(dx_px, dy_px)
        return AimError(dx_px=dx_px, dy_px=dy_px, pan_deg=pan_deg, tilt_deg=tilt_deg,
                        distance_px=distance_px, aimed=distance_px <= self.tolerance_px)

    def command(self, error, current, dt_s) -> PanTilt:
        """
        Next position: proportional step from ``current``, limited and clamped.

        The step is computed from this frame's error, so with a working camera
        the next frame reports a smaller error once the turret has moved. The
        rate limit is what keeps a single frame from swinging the servos.
        """
        if dt_s <= 0:
            return self._clamp(current)
        max_step = self.max_rate_deg_s * dt_s
        pan = current.pan_deg + self._limit(error.pan_deg * self.gain, max_step)
        tilt = current.tilt_deg + self._limit(error.tilt_deg * self.gain, max_step)
        return self._clamp(PanTilt(pan_deg=pan, tilt_deg=tilt))

    def update(self, target_center, frame_shape, current, dt_s) -> AimResult:
        """Error, command and settle state for one frame."""
        error = self.error(target_center, frame_shape)
        command = self.command(error, current, dt_s)
        self._aimed_frames = self._aimed_frames + 1 if error.aimed else 0
        return AimResult(
            error=error,
            command=command,
            aimed_frames=self._aimed_frames,
            settled=self._aimed_frames >= self.settle_frames,
        )

    def reset(self) -> None:
        """Forget the settle progress (when leaving AIMING or for a new target)."""
        self._aimed_frames = 0

    # --- internals ---------------------------------------------------------
    @staticmethod
    def _check_limits(name, limits):
        try:
            low, high = limits
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a pair (low, high), got {limits!r}") from None
        if low >= high:
            raise ValueError(f"{name} must be ordered, got {limits!r}")
        return (float(low), float(high))

    @staticmethod
    def _frame_size(frame_shape):
        if not frame_shape or len(frame_shape) < 2:
            raise ValueError(f"frame_shape must contain height and width, got {frame_shape!r}")
        frame_h, frame_w = float(frame_shape[0]), float(frame_shape[1])
        if frame_h <= 0 or frame_w <= 0:
            raise ValueError(f"frame_shape must be positive, got {frame_shape!r}")
        return frame_h, frame_w

    @staticmethod
    def _limit(delta, max_step):
        return max(-max_step, min(max_step, delta))

    def _clamp(self, position) -> PanTilt:
        return PanTilt(
            pan_deg=self._clamp_axis(position.pan_deg, self.pan_limits),
            tilt_deg=self._clamp_axis(position.tilt_deg, self.tilt_limits),
        )

    @staticmethod
    def _clamp_axis(value, limits):
        low, high = limits
        return max(low, min(high, value))
