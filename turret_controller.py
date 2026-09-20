"""
Per-frame orchestration: perception -> safety -> state machine -> hardware.

This is the only module that knows all the parts. Each frame it

1. turns what the target detector sees into exactly one event for the state
   machine,
2. switches servos and pump through the state change hook,
3. makes sure the pump is off whenever the machine is not in ``WATERING``.

Invariants enforced here (and covered by tests):

* the pump is only on in ``WATERING``; leaving that state always stops the water,
* any exception from the detector, the tracker, the servos or the pump becomes
  ``HARDWARE_FAULT`` instead of killing the loop,
* a safety veto while aiming stops the movement, a veto while spraying stops
  the water.
"""

from dataclasses import dataclass
import time

from aim_controller import AimError, PanTilt
from hardware import Position
from safety import BUDGET_CODES, OK
from state_machine import Event, State, StateMachine, default_timeouts
from turret_config import TurretConfig


@dataclass(frozen=True)
class TurretStatus:
    """Everything the display, the log and the tests need for one frame."""

    state: State
    event: Event | None
    message: str
    target: object | None
    aim: AimError | None
    settled_frames: int
    sprays_for_target: int
    sprays_total: int
    safety_code: str
    armed: bool
    pump_on: bool
    position: Position | None


class TurretController:
    """
    Wires detector, aim controller, safety gate, state machine and hardware.

    All collaborators are injected, so the whole command chain can be tested
    with stand-ins; ``config`` only provides the state timeouts.
    """

    def __init__(self, detector, aimer, guard, actuator, pump, config=None,
                 machine=None, calibration=None, source=None, calibrator=None,
                 clock=time.monotonic, log=None):
        self.detector = detector
        self.aimer = aimer
        self.guard = guard
        self.actuator = actuator
        self.pump = pump
        self.config = config if config is not None else TurretConfig()
        self.calibration = calibration
        self.source = source
        self.calibrator = calibrator
        self._clock = clock
        self._log = log
        if machine is not None:
            self.machine = machine
        else:
            self.machine = StateMachine(clock=clock, timeouts=default_timeouts(
                aim_s=self.config.state.aim_s,
                spray_s=self.config.state.spray_s,
                verify_s=self.config.state.verify_s,
                target_confirm_s=self.config.state.target_confirm_s,
            ))
        self.machine.on_change(self._on_state_change)

        self.target = None
        self.aim = None
        self.message = "created"
        self.last_event = None
        self.last_safety_code = OK
        self._position = None
        self._last_step_at = self._clock()

    def _resolve(self, now):
        return self._clock() if now is None else now

    def _log_line(self, text):
        if self._log is not None:
            self._log(text)

    # --- operator controls -------------------------------------------------
    def arm(self):
        """Release the turret (the safety gate does the actual gating)."""
        self.guard.arm()
        self._log_line("operator: armed")

    def disarm(self):
        self.guard.disarm()
        self._log_line("operator: disarmed")

    def emergency_stop(self, reason="operator emergency stop", now=None):
        """Latch the stop, cut the water immediately and report ERROR."""
        now = self._resolve(now)
        self.guard.emergency_stop(reason)
        self.pump.off()
        self.message = f"emergency stop: {reason}"
        self._log_line(f"operator: emergency stop ({reason})")
        self.machine.handle(Event.HARDWARE_FAULT, now)

    def reset(self, now=None):
        """Operator reset: leave ERROR and run INIT again."""
        now = self._resolve(now)
        self.message = "operator reset"
        self._log_line("operator: reset")
        self.machine.handle(Event.RESET, now)
        self.guard.clear_emergency_stop()

    def request_calibration(self, now=None):
        """Ask for a calibration run (from any state that allows it)."""
        self.machine.handle(Event.CALIBRATION_REQUESTED, self._resolve(now))

    # --- state change hook -------------------------------------------------
    def _on_state_change(self, transition):
        """Switch pump and housekeeping whenever the machine changes state."""
        self.last_event = transition.event
        self._log_line(f"{transition.source.name} -> {transition.target.name} ({transition.event.name})")

        if transition.target is State.WATERING:
            self.pump.on()
            self.guard.register_spray_start(transition.at)
        elif transition.source is State.WATERING:
            # leaving WATERING always stops the water, whatever comes next
            self.pump.off()
            self.guard.register_spray_end(transition.at)

        if transition.source is State.AIMING:
            # every aiming episode needs its own settle confirmation
            self.aimer.reset()

        if transition.source in (State.TARGET_FOUND, State.AIMING, State.VERIFYING) and \
                transition.target in (State.IDLE, State.SEARCHING):
            # the attempt is over: forget the candidate and its aiming state
            self.detector.reset()
            self.aimer.reset()
            self.aim = None
            if self.target is None:
                # nothing visible any more: the visit is over, so does its budget
                self.guard.target_left()

    # --- per frame ---------------------------------------------------------
    def step(self, frame=None, now=None) -> TurretStatus:
        """
        One control cycle.

        ``frame`` may be ``None`` (frame source failed or shut down): the state
        machine still runs its timeouts, the pump invariant is enforced and the
        safety gate reports ``no_frame``.
        """
        now = self._resolve(now)
        self._frame_interval = max(0.0, now - self._last_step_at)
        self._last_step_at = now
        self.message = ""
        self.last_event = None

        try:
            self._process(frame, now)
        except Exception as error:                      # any hardware error is a fault
            self._hardware_fault(error, now)
        self._enforce_pump_safety(now)
        return self.status()

    def status(self) -> TurretStatus:
        """Snapshot for display, log and tests."""
        return TurretStatus(
            state=self.machine.state,
            event=self.last_event,
            message=self.message or self._default_message(),
            target=self.target,
            aim=self.aim,
            settled_frames=self.aimer.aimed_frames if self.aimer is not None else 0,
            sprays_for_target=self.guard.sprays_for_target,
            sprays_total=self.guard.sprays_total,
            safety_code=self.last_safety_code,
            armed=self.guard.armed,
            pump_on=bool(self.pump.is_on),
            position=self._position,
        )

    def close(self):
        """Stop the water, park the head and release both devices."""
        try:
            self.pump.off()
        finally:
            self.actuator.close()
            self.pump.close()
        self._log_line("closed")

    def _process(self, frame, now):
        state = self.machine.state
        if state is State.INIT:
            self._initialise(now)
            return
        if state is State.CALIBRATION:
            self._calibrate(now)
            return
        if state is State.ERROR:
            return

        if frame is not None:
            self.guard.register_frame(now)
        self.target = self.detector.update(frame, now) if frame is not None else None
        event = self._event_for_state(frame, now)
        if event is not None:
            self.machine.handle(event, now)
        self.machine.tick(now)

    def _initialise(self, now):
        """INIT: the turret needs a calibration before it may aim."""
        if self.calibration is None:
            self.message = "no calibration available"
            self._log_line("init failed: no calibration")
            self.machine.handle(Event.INIT_FAILED, now)
            return
        self._apply_calibration()
        stamp = self.calibration.created or "unknown"
        self.message = f"initialised (calibration {stamp})"
        self.machine.handle(Event.INIT_DONE, now)

    def _apply_calibration(self):
        """Adopt the calibration: limits, centre position, aim settle state."""
        calibration = self.calibration
        if self.aimer is not None:
            self.aimer.reset()
        if self.actuator is not None:
            pan_deg, tilt_deg = calibration.clamp(calibration.pan_center, calibration.tilt_center)
            self._position = Position(pan_deg=pan_deg, tilt_deg=tilt_deg)
            self.actuator.point_at(pan_deg, tilt_deg)
        self._log_line(f"calibration loaded: pan_limits={calibration.pan_limits}, "
                       f"tilt_limits={calibration.tilt_limits}, hfov_deg={calibration.hfov_deg}")

    def _calibrate(self, now):
        """CALIBRATION: run the injected calibrator, then store the result."""
        if self.calibrator is None:
            self.message = "no calibrator configured"
            self._log_line("calibration failed: no calibrator")
            self.machine.handle(Event.CALIBRATION_FAILED, now)
            return
        try:
            calibration = self.calibrator.run(source=self.source)
        except Exception as error:
            self.message = f"calibration failed: {error}"
            self._log_line(f"calibration failed: {error!r}")
            self.machine.handle(Event.CALIBRATION_FAILED, now)
            return
        self.calibration = calibration
        self._apply_calibration()
        self.message = "calibrated"
        self.machine.handle(Event.CALIBRATION_DONE, now)

    def _event_for_state(self, frame, now):
        """Decide which event the current frame produces (at most one)."""
        frame_shape = getattr(frame, "shape", None)
        state = self.machine.state
        if state is State.IDLE:
            return self._event_in_idle()
        if state is State.SEARCHING:
            return self._event_in_searching()
        if state is State.TARGET_FOUND:
            return self._event_in_target_found()
        if state is State.AIMING:
            return self._event_in_aiming(frame_shape, now)
        if state is State.WATERING:
            return self._event_in_watering(now)
        if state is State.VERIFYING:
            return self._event_in_verifying(frame_shape, now)
        return None

    # --- per state decisions ----------------------------------------------
    def _event_in_idle(self):
        """IDLE: start searching as soon as the operator armed the turret."""
        if not self.guard.armed:
            return None
        if self.target is not None and self._is_confirmed(self.target):
            return Event.TARGET_SPOTTED
        return Event.ARM

    def _event_in_searching(self):
        """SEARCHING: report a candidate, or stand down when disarmed."""
        if not self.guard.armed:
            return Event.DISARM
        if self.target is not None:
            return Event.TARGET_SPOTTED
        return None

    def _event_in_target_found(self):
        """TARGET_FOUND: confirm the candidate, give up when it disappears."""
        if self.target is None:
            return Event.TARGET_LOST
        if self._is_confirmed(self.target):
            self.aimer.reset()
            return Event.TARGET_CONFIRMED
        return None

    def _event_in_aiming(self, frame_shape, now):
        """AIMING: move towards the target, then ask for permission to spray."""
        if self.target is None:
            return Event.TARGET_LOST

        aim_decision = self.guard.check_aim(self.target, frame_shape, now)
        self.last_safety_code = aim_decision.code
        if not aim_decision.allowed:
            self.message = f"aim blocked: {aim_decision.message}"
            return Event.SAFETY_BLOCK

        result = self.aimer.update(self.target.center, frame_shape, self._current_position(), self._frame_interval)
        self.aim = result.error
        self._command(result.command)
        if not result.settled:
            return None

        spray_decision = self.guard.check_spray(self.target, frame_shape, now)
        self.last_safety_code = spray_decision.code
        if spray_decision.allowed:
            return Event.AIM_SETTLED
        self.message = f"spray blocked: {spray_decision.message}"
        return Event.SAFETY_BLOCK

    def _event_in_watering(self, now):
        """WATERING: only safety can end a burst early, plus the hard time cap."""
        if self.guard.stopped or not self.guard.armed:
            self.message = "spray aborted: turret no longer armed"
            return Event.SAFETY_BLOCK
        if self.guard.spray_overrun(now):
            self.message = "spray cut at the hard time limit"
            return Event.SPRAY_FINISHED
        return None

    def _event_in_verifying(self, frame_shape, now):
        """VERIFYING: did the cat leave, may we repeat, or are we done for now?"""
        if self.target is None:
            return Event.TARGET_REPELLED

        decision = self.guard.check_spray(self.target, frame_shape, now)
        self.last_safety_code = decision.code
        if decision.allowed:
            return Event.TARGET_STILL_PRESENT
        if decision.code in BUDGET_CODES:
            self.message = f"spray budget used up: {decision.message}"
            return Event.SPRAY_BUDGET_EXHAUSTED
        self.message = f"waiting: {decision.message}"
        return None

    # --- helpers -----------------------------------------------------------
    def _is_confirmed(self, target) -> bool:
        return target.frames_confirmed >= self.guard.limits.confirm_frames

    def _current_position(self) -> PanTilt:
        if self._position is None:
            return PanTilt(pan_deg=0.0, tilt_deg=0.0)
        return PanTilt(pan_deg=self._position.pan_deg, tilt_deg=self._position.tilt_deg)

    def _command(self, position: PanTilt) -> None:
        self.actuator.point_at(position.pan_deg, position.tilt_deg)
        self._position = Position(pan_deg=position.pan_deg, tilt_deg=position.tilt_deg)

    def _enforce_pump_safety(self, now) -> bool:
        """Watchdog: the pump must never run outside WATERING."""
        if self.machine.state is State.WATERING or not self.pump.is_on:
            return False
        self.pump.off()
        self.guard.register_spray_end(now)
        self.message = "pump watchdog tripped"
        self._log_line("watchdog: pump forced off outside WATERING")
        self.machine.handle(Event.HARDWARE_FAULT, now)
        return True

    def _hardware_fault(self, error, now):
        """Turn any exception from the hardware chain into a latched ERROR."""
        self.message = f"hardware fault: {error}"
        self._log_line(f"hardware fault: {error!r}")
        try:
            self.pump.off()
        except Exception:                                # the pump is the priority
            self._log_line("hardware fault: pump could not be switched off")
        self.machine.handle(Event.HARDWARE_FAULT, now)

    def _default_message(self) -> str:
        state = self.machine.state
        if state is State.IDLE:
            return "idle (armed)" if self.guard.armed else "idle (disarmed)"
        return {
            State.INIT: "initialising",
            State.SEARCHING: "searching for a cat",
            State.TARGET_FOUND: "confirming the candidate",
            State.AIMING: "aiming at the target",
            State.WATERING: "spraying",
            State.VERIFYING: "verifying the effect",
            State.CALIBRATION: "calibrating",
            State.ERROR: "error - press r to reset",
        }.get(state, str(state))