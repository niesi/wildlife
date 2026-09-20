import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from datetime import datetime

import numpy as np

from aim_controller import AimController
from calibration import Calibration
from hardware import DryRunWaterPump, Position, SimulatedPanTilt, SimulatedWaterPump
from safety import CONFIDENCE_TOO_LOW, OK, OUTSIDE_SPRAY_ZONE, SafetyGuard
from state_machine import Event, State, StateMachine, default_timeouts
from target_detector import Target
from turret_config import SafetyLimits, StateTimeouts, TurretConfig
from turret_controller import TurretController


FRAME = np.full((240, 320), 80, dtype=np.uint8)
BOX = (140, 100, 40, 40)
CENTER = (160.0, 120.0)
DT = 0.033
_DEFAULT = object()


class FakeClock:
    """Deterministic stand-in for time.monotonic."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


class FakeDetector:
    """Target detector stand-in with a scripted or a fixed answer."""

    def __init__(self, targets=None, default=None):
        self.targets = list(targets or [])
        self.default = default
        self.calls = 0
        self.resets = 0

    def update(self, frame, now):
        self.calls += 1
        if self.targets:
            return self.targets.pop(0)
        return self.default

    def reset(self):
        self.resets += 1


class BrokenHead(SimulatedPanTilt):
    """Head that fails as soon as it is asked to leave the centre."""

    def point_at(self, pan_deg, tilt_deg):
        if abs(pan_deg) > 1e-9 or abs(tilt_deg) > 1e-9:
            raise OSError("servo bus down")
        super().point_at(pan_deg, tilt_deg)


class FakeCalibrator:
    """Calibrator stand-in returning a fixed result or raising."""

    def __init__(self, calibration=None, error=None):
        self.calibration = calibration
        self.error = error
        self.calls = 0

    def run(self, source=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.calibration


def make_target(frames_confirmed=3, label="cat", confidence=0.9, center=CENTER, box=BOX):
    return Target(box=box, center=center, label=label, confidence=confidence,
                  frames_confirmed=frames_confirmed, first_seen_at=0.0, last_seen_at=0.0)


def build(armed=True, default_target=None, targets=None, limits=None, state=None,
          calibration=_DEFAULT, head=None, pump=None, calibrator=None, machine=None,
          clock=None, log=None, source=None):
    """Controller with simulated hardware, a fake detector and a fixed clock."""
    clock = clock if clock is not None else FakeClock()
    limits = limits if limits is not None else SafetyLimits(
        armed=armed, cooldown_s=0.0, allowed_hours=None,
        max_sprays_per_target=2, max_sprays_per_hour=10,
    )
    detector = FakeDetector(targets, default=default_target)
    guard = SafetyGuard(limits, clock=clock, wall_clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    head = head if head is not None else SimulatedPanTilt(clock=clock)
    pump = pump if pump is not None else SimulatedWaterPump(clock=clock)
    aimer = AimController(gain=1.0, max_rate_deg_s=1000.0, settle_frames=2)
    config = TurretConfig(state=state if state is not None else StateTimeouts(), safety=limits)
    controller = TurretController(
        detector=detector, aimer=aimer, guard=guard, actuator=head, pump=pump, config=config,
        machine=machine, calibrator=calibrator, source=source, clock=clock, log=log,
        calibration=Calibration() if calibration is _DEFAULT else calibration,
    )
    return controller, detector, head, pump, guard, clock


def step(controller, clock, frame=FRAME, dt=DT):
    """Advance the clock by one frame and run one control cycle."""
    clock.advance(dt)
    return controller.step(frame, clock.now)


def step_until(controller, clock, state, limit=40, frame=FRAME):
    """Run frames until the controller reports ``state``."""
    for _ in range(limit):
        status = step(controller, clock, frame)
        if status.state is state:
            return status
    raise AssertionError(f"state {state.name} was not reached")


class LifecycleTest(unittest.TestCase):
    def test_first_step_initialises_and_parks_the_head(self):
        controller, _, head, pump, _, clock = build()

        status = controller.step(FRAME, clock.now)

        self.assertIs(status.state, State.IDLE)
        self.assertIn("initialised", status.message)
        self.assertFalse(pump.is_on)
        self.assertEqual(status.safety_code, OK)
        self.assertTrue(status.armed)
        self.assertEqual(head.target, Position(0.0, 0.0))

    def test_missing_calibration_fails_into_error(self):
        controller, _, _, _, _, clock = build(calibration=None)

        status = step(controller, clock)

        self.assertIs(status.state, State.ERROR)
        self.assertIn("no calibration", status.message)

    def test_error_is_latched_until_the_operator_resets(self):
        controller, _, _, _, _, clock = build(calibration=None)
        step(controller, clock)

        for _ in range(3):
            self.assertIs(step(controller, clock).state, State.ERROR)

        controller.calibration = Calibration()
        controller.reset()

        self.assertIs(step(controller, clock).state, State.IDLE)
        self.assertEqual([transition.target for transition in controller.machine.history],
                         [State.ERROR, State.INIT, State.IDLE])

    def test_disarmed_turret_stays_idle(self):
        controller, _, _, pump, _, clock = build(armed=False)

        for _ in range(6):
            status = step(controller, clock)
            self.assertIs(status.state, State.IDLE)

        self.assertIn("disarmed", status.message)
        self.assertFalse(pump.is_on)

        controller.arm()

        self.assertIs(step(controller, clock).state, State.SEARCHING)

    def test_disarm_stops_searching(self):
        controller, _, _, _, _, clock = build()
        step_until(controller, clock, State.SEARCHING)

        controller.disarm()
        status = step(controller, clock)

        self.assertIs(status.state, State.IDLE)
        self.assertFalse(status.armed)
        self.assertEqual(controller.last_event, Event.DISARM)


class HappyPathTest(unittest.TestCase):
    def test_full_cycle_reaches_watering_and_keeps_the_pump_invariant(self):
        controller, _, _, pump, guard, clock = build(default_target=make_target())

        statuses = [step(controller, clock) for _ in range(12)]

        for expected in (State.IDLE, State.SEARCHING, State.TARGET_FOUND, State.AIMING, State.WATERING):
            self.assertIn(expected, [status.state for status in statuses])
        for status in statuses:
            self.assertEqual(status.pump_on, status.state is State.WATERING,
                             msg=f"{status.state.name} reported pump_on={status.pump_on}")
        self.assertEqual([transition.event for transition in controller.machine.history],
                         [Event.INIT_DONE, Event.ARM, Event.TARGET_SPOTTED,
                          Event.TARGET_CONFIRMED, Event.AIM_SETTLED])
        self.assertEqual(pump.switches, 1)
        self.assertEqual(guard.sprays_total, 1)

    def test_the_burst_ends_after_the_nominal_spray_time(self):
        controller, _, _, pump, _, clock = build(default_target=make_target())
        step_until(controller, clock, State.WATERING)
        self.assertTrue(pump.is_on)

        status = step_until(controller, clock, State.VERIFYING)

        self.assertFalse(status.pump_on)
        self.assertGreaterEqual(pump.on_seconds(), 0.3)
        self.assertLess(pump.on_seconds(), 0.45)

    def test_configured_spray_time_is_used(self):
        controller, _, _, pump, _, clock = build(
            default_target=make_target(),
            state=StateTimeouts(aim_s=2.0, spray_s=0.1, verify_s=5.0, target_confirm_s=1.0),
        )

        step_until(controller, clock, State.WATERING)
        step_until(controller, clock, State.VERIFYING)

        self.assertLess(pump.on_seconds(), 0.2)

    def test_stubborn_cat_gets_a_second_burst(self):
        controller, _, _, pump, guard, clock = build(default_target=make_target())

        step_until(controller, clock, State.WATERING)
        step_until(controller, clock, State.VERIFYING)
        step_until(controller, clock, State.WATERING)

        self.assertEqual(guard.sprays_total, 2)
        self.assertEqual(pump.switches, 3)
        self.assertTrue(pump.is_on)
