"""
Safety gate for aiming and spraying.

Pure logic without camera, tracker or hardware dependencies: the guard
receives a generic target object (``label``, ``center``, ``box``,
``confidence``, ``frames_confirmed``) plus the frame shape and answers with a
:class:`SafetyDecision`. Switching the pump off stays the job of the
controller that owns the actuator - this module only allows or vetoes.

Every veto carries a stable code, so the controller can map it onto an event
(``SAFETY_BLOCK`` / ``SPRAY_BUDGET_EXHAUSTED``) and the display can show a
reason without parsing prose.
"""

from dataclasses import dataclass, field
from datetime import datetime
import time

from turret_config import SafetyLimits


OK = "ok"
EMERGENCY_STOP = "emergency_stop"
NOT_ARMED = "not_armed"
NO_FRAME = "no_frame"
NO_TARGET = "no_target"
LABEL_NOT_ALLOWED = "label_not_allowed"
TARGET_NOT_CONFIRMED = "target_not_confirmed"
CONFIDENCE_TOO_LOW = "confidence_too_low"
OUTSIDE_SPRAY_ZONE = "outside_spray_zone"
TARGET_TOO_SMALL = "target_too_small"
TARGET_TOO_LARGE = "target_too_large"
SPRAY_IN_PROGRESS = "spray_in_progress"
BUDGET_TARGET_EXHAUSTED = "budget_target_exhausted"
BUDGET_HOURLY_EXHAUSTED = "budget_hourly_exhausted"
COOLDOWN_ACTIVE = "cooldown_active"
OUTSIDE_TIME_WINDOW = "outside_time_window"

# Evaluation order of the checks; the first matching veto wins. Kept as data
# so the documented order and the implementation cannot drift apart unnoticed.
VETO_PRIORITY = (
    EMERGENCY_STOP,
    NOT_ARMED,
    NO_FRAME,
    NO_TARGET,
    LABEL_NOT_ALLOWED,
    TARGET_NOT_CONFIRMED,
    CONFIDENCE_TOO_LOW,
    OUTSIDE_SPRAY_ZONE,
    TARGET_TOO_SMALL,
    TARGET_TOO_LARGE,
    SPRAY_IN_PROGRESS,
    BUDGET_TARGET_EXHAUSTED,
    BUDGET_HOURLY_EXHAUSTED,
    COOLDOWN_ACTIVE,
    OUTSIDE_TIME_WINDOW,
)

# Codes that mean "the spray budget is used up" (controller -> SPRAY_BUDGET_EXHAUSTED).
BUDGET_CODES = frozenset({BUDGET_TARGET_EXHAUSTED, BUDGET_HOURLY_EXHAUSTED})


@dataclass(frozen=True)
class SafetyDecision:
    """Result of a safety check: released or vetoed with a code."""

    allowed: bool
    code: str
    message: str

    @classmethod
    def allow(cls, message="released") -> "SafetyDecision":
        return cls(True, OK, message)

    @classmethod
    def veto(cls, code: str, message: str) -> "SafetyDecision":
        return cls(False, code, message)


@dataclass
class SafetyStats:
    """Counters for the display and for the event log."""

    sprays_total: int = 0
    sprays_last_hour: int = 0
    sprays_for_target: int = 0
    last_spray_end: float | None = None
    veto_counts: dict = field(default_factory=dict)


class SafetyGuard:
    """
    Decides whether the turret may aim at or spray a target.

    ``clock`` is a monotonic seconds source (cooldown, spray duration, frame
    staleness); ``wall_clock`` returns a datetime and is only used for the
    allowed time window. Both are injectable so tests stay deterministic.

    Disarmed means "neither aiming nor spraying": the state machine keeps the
    turret in IDLE anyway, but the guard repeats the check so a single wrong
    event can never make the turret move.
    """

    HOUR_S = 3600.0

    def __init__(self, limits=None, clock=time.monotonic, wall_clock=datetime.now):
        self.limits = limits if limits is not None else SafetyLimits()
        self._clock = clock
        self._wall_clock = wall_clock
        self.armed = self.limits.armed
        self.stopped = False
        self.stop_reason = ""
        self._sprays: list[float] = []
        self._sprays_total = 0
        self._sprays_for_target = 0
        self._last_spray_end = None
        self._spray_started_at = None
        self._last_frame_at = None
        self._veto_counts: dict = {}

    def _resolve(self, now):
        return self._clock() if now is None else now

    # --- operator controls -------------------------------------------------
    def arm(self) -> None:
        """Release the turret; only an armed guard can aim or spray."""
        self.armed = True

    def disarm(self) -> None:
        self.armed = False

    def emergency_stop(self, reason="emergency stop") -> None:
        """Latch a stop: vetoes everything and disarms until it is cleared."""
        self.stopped = True
        self.stop_reason = reason
        self.disarm()

    def clear_emergency_stop(self) -> None:
        """Release the stop latch; the turret stays disarmed and must be armed again."""
        self.stopped = False
        self.stop_reason = ""

    # --- inputs -----------------------------------------------------------
    def register_frame(self, now=None) -> None:
        """Call once per processed frame; stale frames veto everything."""
        self._last_frame_at = self._resolve(now)

    def target_left(self) -> None:
        """Reset the per-target budget (call as soon as the cat is gone)."""
        self._sprays_for_target = 0

    # --- spray bookkeeping -------------------------------------------------
    def register_spray_start(self, now=None) -> None:
        """Count a spray that is starting; the pump must be switched on afterwards."""
        now = self._resolve(now)
        self._spray_started_at = now
        self._sprays_total += 1
        self._sprays_for_target += 1
        self._sprays = self._recent_sprays(now) + [now]

    def register_spray_end(self, now=None) -> None:
        """Note the end of a spray; starts the cooldown."""
        now = self._resolve(now)
        self._last_spray_end = now
        self._spray_started_at = None

    def spray_overrun(self, now=None) -> bool:
        """True if the running spray exceeded ``max_spray_s`` (force the pump off)."""
        if self._spray_started_at is None:
            return False
        return self._resolve(now) - self._spray_started_at > self.limits.max_spray_s

    @property
    def spraying(self) -> bool:
        """True while a spray is registered as running."""
        return self._spray_started_at is not None

    @property
    def sprays_total(self) -> int:
        """Number of sprays started since start-up."""
        return self._sprays_total

    @property
    def sprays_for_target(self) -> int:
        """Sprays counted for the current target (reset by ``target_left``)."""
        return self._sprays_for_target

    def _recent_sprays(self, now) -> list[float]:
        return [started for started in self._sprays if now - started < self.HOUR_S]

    def stats(self) -> SafetyStats:
        """Snapshot of the counters (does not change any state)."""
        return SafetyStats(
            sprays_total=self._sprays_total,
            sprays_last_hour=len(self._recent_sprays(self._clock())),
            sprays_for_target=self._sprays_for_target,
            last_spray_end=self._last_spray_end,
            veto_counts=dict(self._veto_counts),
        )

    # --- checks ------------------------------------------------------------
    def check_aim(self, target, frame_shape, now=None) -> SafetyDecision:
        """
        May the turret point at ``target``?

        Cooldown, budgets and the time window are ignored here: they gate the
        water, not the servos.
        """
        return self._evaluate(target, frame_shape, now, spraying=False)

    def check_spray(self, target, frame_shape, now=None) -> SafetyDecision:
        """May the turret spray ``target``? Full check including all budgets."""
        return self._evaluate(target, frame_shape, now, spraying=True)

    def _evaluate(self, target, frame_shape, now, spraying) -> SafetyDecision:
        """Run the checks in VETO_PRIORITY order; the first veto wins."""
        now = self._resolve(now)
        checks = [
            lambda: self._check_operator(),
            lambda: self._check_frame(now),
            lambda: self._check_present(target),
            lambda: self._check_label(target),
            lambda: self._check_confirmation(target),
            lambda: self._check_confidence(target),
            lambda: self._check_position(target, frame_shape),
        ]
        if spraying:
            checks += [
                lambda: self._check_spray_state(),
                lambda: self._check_budgets(now),
                lambda: self._check_cooldown(now),
                lambda: self._check_time_window(),
            ]

        for check in checks:
            decision = check()
            if decision is not None:
                self._veto_counts[decision.code] = self._veto_counts.get(decision.code, 0) + 1
                return decision
        return SafetyDecision.allow("spray released" if spraying else "aim released")

    def _check_operator(self):
        if self.stopped:
            return SafetyDecision.veto(EMERGENCY_STOP, f"emergency stop latched: {self.stop_reason}")
        if not self.armed:
            return SafetyDecision.veto(NOT_ARMED, "turret is disarmed")
        return None

    def _check_frame(self, now):
        if self._last_frame_at is None:
            return SafetyDecision.veto(NO_FRAME, "no frame processed yet")
        age = now - self._last_frame_at
        if age > self.limits.frame_timeout_s:
            return SafetyDecision.veto(NO_FRAME, f"frame is {age:.2f}s old")
        return None

    def _check_present(self, target):
        if target is None:
            return SafetyDecision.veto(NO_TARGET, "no target")
        return None

    def _check_label(self, target):
        label = getattr(target, "label", None)
        if label not in self.limits.target_labels:
            allowed = ", ".join(str(value) for value in self.limits.target_labels)
            return SafetyDecision.veto(LABEL_NOT_ALLOWED, f"label {label!r} is not one of: {allowed}")
        return None

    def _check_confirmation(self, target):
        seen = int(getattr(target, "frames_confirmed", 0) or 0)
        if seen < self.limits.confirm_frames:
            return SafetyDecision.veto(
                TARGET_NOT_CONFIRMED,
                f"{seen}/{self.limits.confirm_frames} confirmation frames",
            )
        return None

    def _check_confidence(self, target):
        confidence = float(getattr(target, "confidence", 0.0) or 0.0)
        if confidence < self.limits.min_confidence:
            return SafetyDecision.veto(
                CONFIDENCE_TOO_LOW,
                f"confidence {confidence:.2f} below {self.limits.min_confidence:.2f}",
            )
        return None

    def _check_position(self, target, frame_shape):
        """Spray zone and target size, both relative to the frame."""
        if not frame_shape or len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            return SafetyDecision.veto(NO_FRAME, "frame shape is unknown")
        frame_h, frame_w = float(frame_shape[0]), float(frame_shape[1])

        box = getattr(target, "box", None)
        center = getattr(target, "center", None)
        if center is None and box is not None and len(box) == 4:
            center = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        if center is None or box is None or len(box) != 4:
            return SafetyDecision.veto(NO_TARGET, "target has no position")

        x0, y0, x1, y1 = self.limits.spray_zone
        cx, cy = center[0] / frame_w, center[1] / frame_h
        if not (x0 <= cx <= x1 and y0 <= cy <= y1):
            return SafetyDecision.veto(
                OUTSIDE_SPRAY_ZONE,
                f"target at ({cx:.2f}, {cy:.2f}) outside zone {self.limits.spray_zone}",
            )

        height = box[3] / frame_h
        if height < self.limits.min_target_height:
            return SafetyDecision.veto(
                TARGET_TOO_SMALL,
                f"target height {height:.3f} below {self.limits.min_target_height}",
            )
        if height > self.limits.max_target_height:
            return SafetyDecision.veto(
                TARGET_TOO_LARGE,
                f"target height {height:.3f} above {self.limits.max_target_height}",
            )
        return None

    def _check_spray_state(self):
        if self.spraying:
            return SafetyDecision.veto(SPRAY_IN_PROGRESS, "a spray is already running")
        return None

    def _check_budgets(self, now):
        if self._sprays_for_target >= self.limits.max_sprays_per_target:
            return SafetyDecision.veto(
                BUDGET_TARGET_EXHAUSTED,
                f"{self._sprays_for_target}/{self.limits.max_sprays_per_target} sprays for this target",
            )
        recent = len(self._recent_sprays(now))
        if recent >= self.limits.max_sprays_per_hour:
            return SafetyDecision.veto(
                BUDGET_HOURLY_EXHAUSTED,
                f"{recent}/{self.limits.max_sprays_per_hour} sprays in the last hour",
            )
        return None

    def _check_cooldown(self, now):
        if self._last_spray_end is None:
            return None
        elapsed = now - self._last_spray_end
        if elapsed < self.limits.cooldown_s:
            return SafetyDecision.veto(
                COOLDOWN_ACTIVE,
                f"{elapsed:.2f}s of {self.limits.cooldown_s:.2f}s cooldown elapsed",
            )
        return None

    def _check_time_window(self):
        if self.limits.allowed_hours is None:
            return None
        start, end = self.limits.allowed_hours
        hour = self._wall_clock().hour
        if not start <= hour < end:
            return SafetyDecision.veto(
                OUTSIDE_TIME_WINDOW,
                f"local hour {hour} outside {start}:00-{end}:00",
            )
        return None