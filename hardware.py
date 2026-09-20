"""
Actuator interfaces plus the simulated implementations used on the PC.

The pan/tilt head and the water pump sit behind small ABCs, so the controller
never talks to a serial port or a GPIO line directly. Implemented are the
simulated devices (``SimulatedPanTilt``, ``SimulatedWaterPump``) plus
``DryRunWaterPump`` for training runs. The real drivers are deliberately still
skeletons: each raises ``NotImplementedError`` naming the protocol and the
wiring that has to be decided before real servos are allowed to move.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import time


@dataclass(frozen=True)
class Position:
    """Pan/tilt angle in degrees (pan grows right, tilt grows up)."""

    pan_deg: float
    tilt_deg: float


def _check_limits(name, limits):
    try:
        low, high = limits
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a pair (low, high), got {limits!r}") from None
    if low >= high:
        raise ValueError(f"{name} must be ordered, got {limits!r}")
    return (float(low), float(high))


def _clamp(value, limits):
    low, high = limits
    return max(low, min(high, float(value)))


class PanTiltActuator(ABC):
    """Two-axis servo head."""

    @abstractmethod
    def point_at(self, pan_deg, tilt_deg) -> None:
        """Command an absolute position; out-of-range targets are clamped."""

    @abstractmethod
    def position(self) -> Position:
        """Current position (measured, or the last commanded one)."""

    def wait_until_settled(self, timeout_s=2.0) -> bool:
        """
        Block until the head has reached its target; False if it did not.

        Drivers that acknowledge the move inside ``point_at`` need not override
        this; the simulated head emulates the travel time.
        """
        return True

    def center(self) -> Position:
        """Drive to the neutral position (0, 0) and report it."""
        self.point_at(0.0, 0.0)
        return self.position()

    def close(self) -> None:
        """Release the hardware; must be safe to call more than once."""


class WaterPump(ABC):
    """Water valve or pump."""

    @abstractmethod
    def on(self) -> None:
        """Start the jet."""

    @abstractmethod
    def off(self) -> None:
        """Stop the jet; must be idempotent."""

    @property
    @abstractmethod
    def is_on(self) -> bool:
        """True while the jet is running."""

    def close(self) -> None:
        """Release the hardware (always switches the pump off)."""
        self.off()


class SimulatedPanTilt(PanTiltActuator):
    """
    Servo head stand-in that moves towards its target at ``deg_per_s``.

    The movement is integrated from the clock, so the distance a real head
    could cover between two frames is realistic and ``AIMING`` settles the same
    way it would with real servos. ``snap()`` jumps straight to the target and
    is meant for tests.
    """

    def __init__(self, pan_limits=(-90.0, 90.0), tilt_limits=(-45.0, 45.0),
                 deg_per_s=120.0, clock=time.monotonic, start=(0.0, 0.0)):
        self.pan_limits = _check_limits("pan_limits", pan_limits)
        self.tilt_limits = _check_limits("tilt_limits", tilt_limits)
        if deg_per_s <= 0:
            raise ValueError(f"deg_per_s must be positive, got {deg_per_s!r}")
        self.deg_per_s = float(deg_per_s)
        self._clock = clock
        self._position = Position(pan_deg=_clamp(start[0], self.pan_limits),
                                  tilt_deg=_clamp(start[1], self.tilt_limits))
        self._target = self._position
        self._updated_at = self._clock()
        self.closed = False

    def point_at(self, pan_deg, tilt_deg) -> None:
        self._advance()
        self._target = Position(pan_deg=_clamp(pan_deg, self.pan_limits),
                                tilt_deg=_clamp(tilt_deg, self.tilt_limits))

    def position(self) -> Position:
        self._advance()
        return self._position

    @property
    def target(self) -> Position:
        """Position the head is heading for."""
        self._advance()
        return self._target

    @property
    def settled(self) -> bool:
        """True when the head has reached its target."""
        self._advance()
        return self._position == self._target

    def snap(self) -> Position:
        """Jump to the target without waiting (tests only)."""
        self._advance()
        self._position = self._target
        return self._position

    def wait_until_settled(self, timeout_s=2.0, poll_s=0.005) -> bool:
        """Sleep until the head has reached its target or the timeout expired."""
        deadline = self._clock() + max(0.0, timeout_s)
        while not self.settled:
            remaining = deadline - self._clock()
            if remaining <= 0:
                break
            before = self._clock()
            time.sleep(min(poll_s, remaining))
            if self._clock() <= before:
                break       # frozen clock (tests): do not spin forever
        return self.settled

    def close(self) -> None:
        self.closed = True

    # --- internals ---------------------------------------------------------
    def _advance(self) -> None:
        now = self._clock()
        elapsed = now - self._updated_at
        if elapsed <= 0:
            return
        self._updated_at = now
        max_step = self.deg_per_s * elapsed
        self._position = Position(
            pan_deg=self._step(self._position.pan_deg, self._target.pan_deg, max_step),
            tilt_deg=self._step(self._position.tilt_deg, self._target.tilt_deg, max_step),
        )

    @staticmethod
    def _step(value, target, max_step):
        if abs(target - value) <= max_step:
            return target
        return value + math.copysign(max_step, target - value)


class SimulatedWaterPump(WaterPump):
    """Pump stand-in that records every switch for tests, logs and stats."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self._is_on = False
        self.history = []       # list of (timestamp, "on" | "off")
        self.closed = False

    def on(self) -> None:
        if not self._is_on:
            self._is_on = True
            self.history.append((self._clock(), "on"))

    def off(self) -> None:
        if self._is_on:
            self._is_on = False
            self.history.append((self._clock(), "off"))

    @property
    def is_on(self) -> bool:
        return self._is_on

    @property
    def switches(self) -> int:
        """Number of recorded on/off actions."""
        return len(self.history)

    def on_seconds(self, now=None) -> float:
        """Total time the pump has been running (including a running jet)."""
        total = 0.0
        started = None
        for at, action in self.history:
            if action == "on":
                started = at
            elif started is not None:
                total += at - started
                started = None
        if started is not None:
            total += (self._clock() if now is None else now) - started
        return total

    def close(self) -> None:
        self.off()
        self.closed = True


class DryRunWaterPump(WaterPump):
    """
    Pump wrapper for dry runs: the inner pump is never switched on.

    ``on()`` only counts the request, so a training run shows what would have
    happened; ``off()`` always reaches the inner pump so a dry run can release
    a latched hardware state as well.
    """

    def __init__(self, inner):
        self.inner = inner
        self.requests = 0
        self.closed = False

    def on(self) -> None:
        self.requests += 1

    def off(self) -> None:
        self.inner.off()

    @property
    def is_on(self) -> bool:
        """Always False: a dry run never waters."""
        return False

    @property
    def inner_is_on(self) -> bool:
        """Inner state, for tests checking that the wrapper really blocks."""
        return bool(getattr(self.inner, "is_on", False))

    def close(self) -> None:
        self.closed = True
        self.inner.close()


class SerialPanTilt(PanTiltActuator):
    """
    Real pan/tilt head (skeleton).

    Planned: newline terminated text protocol over USB serial to a small MCU
    that drives the servos, answers with an acknowledgement and enforces the
    soft limits itself. ``pyserial`` is imported lazily, so neither the PC
    prototype nor the tests need that dependency.
    """

    def __init__(self, port, baudrate=115200, pan_limits=(-90.0, 90.0), tilt_limits=(-45.0, 45.0)):
        raise NotImplementedError(
            "SerialPanTilt needs the servo controller firmware and its serial "
            "protocol; see the hardware notes in TURRET.md."
        )

    def point_at(self, pan_deg, tilt_deg) -> None:
        raise NotImplementedError

    def position(self) -> Position:
        raise NotImplementedError


class GpioWaterPump(WaterPump):
    """
    Real pump on the target board (skeleton).

    Planned: switch a MOSFET/relay through libgpiod on the RV1106 and repeat the
    ``max_spray_s`` limit in firmware, so a crashed host process cannot water
    forever. ``libgpiod`` is imported lazily for the same reason as above.
    """

    def __init__(self, pin, max_spray_s=0.4, active_high=True):
        raise NotImplementedError(
            "GpioWaterPump needs the board wiring (pin, MOSFET/relay, flyback "
            "diode); see the hardware notes in TURRET.md."
        )

    def on(self) -> None:
        raise NotImplementedError

    def off(self) -> None:
        raise NotImplementedError

    @property
    def is_on(self) -> bool:
        raise NotImplementedError


def create_hardware(kind: str, **kwargs):
    """
    Build an ``(actuator, pump)`` pair for a link type.

    ``simulated`` (development default) integrates the movement in software and
    records the pump switches. ``device`` returns the serial/GPIO drivers, which
    are still skeletons and therefore raise. ``dry_run=True`` wraps the pump so
    the turret can be trained without water.
    """
    if kind == "simulated":
        allowed = {"pan_limits", "tilt_limits", "deg_per_s", "clock"}
        actuator = SimulatedPanTilt(**{key: value for key, value in kwargs.items() if key in allowed})
        pump = SimulatedWaterPump()
        if kwargs.get("dry_run"):
            pump = DryRunWaterPump(pump)
        return actuator, pump

    if kind == "device":
        actuator = SerialPanTilt(port=kwargs.get("port"),
                                 baudrate=kwargs.get("baudrate", 115200),
                                 pan_limits=kwargs.get("pan_limits", (-90.0, 90.0)),
                                 tilt_limits=kwargs.get("tilt_limits", (-45.0, 45.0)))
        pump = GpioWaterPump(pin=kwargs.get("pump_pin"),
                             max_spray_s=kwargs.get("max_spray_s", 0.4))
        return actuator, pump

    raise ValueError(f"unknown hardware kind: {kind!r}")
