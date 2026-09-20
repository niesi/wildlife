"""
State machine of the automatic water turret (cat deterrent).

Pure logic only: this module imports no camera, no tracker and no hardware.
The controller added in a later phase feeds events into
``StateMachine.handle`` / ``StateMachine.tick`` and reacts to state changes
through the ``on_change`` callbacks - that is where the pump is switched off
whenever the machine leaves WATERING and where a spray is started when
WATERING is entered.

Printing this file dumps the transition table and the timeouts as Markdown,
which is the source for the tables in TURRET.md::

    python state_machine.py
"""

from dataclasses import dataclass
from enum import Enum
import time


class State(Enum):
    """Operating states of the turret."""

    INIT = 0          # start-up: prepare sources, actuators and calibration
    IDLE = 1          # powered but not scanning (disarmed or nothing to do)
    SEARCHING = 2     # scanning the garden for a cat
    TARGET_FOUND = 3  # candidate detected, waiting for confirmation
    AIMING = 4        # servos are pointing at the target
    WATERING = 5      # pump is on
    VERIFYING = 6     # watching whether the cat left
    CALIBRATION = 7   # measuring servo limits / field of view
    ERROR = 8         # fault; only RESET or CALIBRATION_REQUESTED leaves it


class Event(Enum):
    """Inputs that can drive the state machine."""

    INIT_DONE = 0                  # sources and actuators ready
    INIT_FAILED = 1                # something could not be opened
    ARM = 2                        # operator arms the turret
    DISARM = 3                     # operator disarms the turret
    TARGET_SPOTTED = 4             # motion region worth investigating
    TARGET_CONFIRMED = 5           # classifier confirmed the target class
    TARGET_LOST = 6                # target disappeared while acquiring/aiming
    AIM_SETTLED = 7                # aim error inside tolerance for N frames
    AIM_TIMEOUT = 8                # could not settle in time
    SPRAY_FINISHED = 9             # nominal spray duration elapsed
    TARGET_REPELLED = 10           # cat moved away after the spray
    TARGET_STILL_PRESENT = 11      # cat did not react, budget may allow a repeat
    VERIFY_TIMEOUT = 12            # verification window elapsed without verdict
    SPRAY_BUDGET_EXHAUSTED = 13    # per-target or hourly spray budget used up
    SAFETY_BLOCK = 14              # safety guard vetoed aim or spray
    SAFETY_CLEARED = 15            # veto reason gone (controller may search again)
    CALIBRATION_REQUESTED = 16     # operator or start-up asked for calibration
    CALIBRATION_DONE = 17          # calibration stored successfully
    CALIBRATION_FAILED = 18        # calibration could not be completed
    HARDWARE_FAULT = 19            # actuator, camera or communication failure
    RESET = 20                     # operator acknowledged an error


# Allowed transitions. An event that is not listed for the current state is
# ignored ("event not allowed here"), so a bad sensor reading can never push
# the machine into an arbitrary state.
TRANSITIONS: dict[State, dict[Event, State]] = {
    State.INIT: {
        Event.INIT_DONE: State.IDLE,
        Event.INIT_FAILED: State.ERROR,
        Event.CALIBRATION_REQUESTED: State.CALIBRATION,
    },
    State.IDLE: {
        Event.ARM: State.SEARCHING,
        Event.TARGET_SPOTTED: State.SEARCHING,
        Event.CALIBRATION_REQUESTED: State.CALIBRATION,
        Event.HARDWARE_FAULT: State.ERROR,
    },
    State.SEARCHING: {
        Event.TARGET_SPOTTED: State.TARGET_FOUND,
        Event.DISARM: State.IDLE,
        Event.SAFETY_BLOCK: State.IDLE,
        Event.CALIBRATION_REQUESTED: State.CALIBRATION,
        Event.HARDWARE_FAULT: State.ERROR,
    },
    State.TARGET_FOUND: {
        Event.TARGET_CONFIRMED: State.AIMING,
        Event.TARGET_LOST: State.SEARCHING,
        Event.SAFETY_BLOCK: State.IDLE,
        Event.HARDWARE_FAULT: State.ERROR,
    },
    State.AIMING: {
        Event.AIM_SETTLED: State.WATERING,
        Event.AIM_TIMEOUT: State.SEARCHING,
        Event.TARGET_LOST: State.SEARCHING,
        Event.SAFETY_BLOCK: State.IDLE,
        Event.HARDWARE_FAULT: State.ERROR,
    },
    # WATERING can only be left through a safety event, never by "just
    # continuing": the pump must stop before anything else happens, and a
    # safety veto while spraying is a fault that needs an operator reset.
    State.WATERING: {
        Event.SPRAY_FINISHED: State.VERIFYING,
        Event.SAFETY_BLOCK: State.ERROR,
        Event.HARDWARE_FAULT: State.ERROR,
    },
    State.VERIFYING: {
        Event.TARGET_REPELLED: State.IDLE,
        Event.TARGET_STILL_PRESENT: State.AIMING,
        Event.SPRAY_BUDGET_EXHAUSTED: State.IDLE,
        Event.TARGET_LOST: State.SEARCHING,
        Event.VERIFY_TIMEOUT: State.SEARCHING,
        Event.HARDWARE_FAULT: State.ERROR,
    },
    State.CALIBRATION: {
        Event.CALIBRATION_DONE: State.IDLE,
        Event.CALIBRATION_FAILED: State.ERROR,
        Event.HARDWARE_FAULT: State.ERROR,
    },
    State.ERROR: {
        Event.RESET: State.INIT,
        Event.CALIBRATION_REQUESTED: State.CALIBRATION,
    },
}


@dataclass(frozen=True)
class Timeout:
    """Automatic event after ``seconds`` in a state."""

    seconds: float
    event: Event


@dataclass(frozen=True)
class Transition:
    """One recorded state change."""

    at: float
    source: State
    event: Event
    target: State


def default_timeouts(aim_s=2.0, spray_s=0.3, verify_s=5.0, target_confirm_s=1.0) -> dict[State, Timeout]:
    """Per-state timeouts; the controller builds them from StateTimeouts."""
    return {
        State.AIMING: Timeout(aim_s, Event.AIM_TIMEOUT),
        State.WATERING: Timeout(spray_s, Event.SPRAY_FINISHED),
        State.VERIFYING: Timeout(verify_s, Event.VERIFY_TIMEOUT),
        State.TARGET_FOUND: Timeout(target_confirm_s, Event.TARGET_LOST),
    }


TIMEOUTS: dict[State, Timeout] = default_timeouts()


def resolve_transition(state: State, event: Event) -> State:
    """Next state for ``event`` in ``state``; unknown events keep the state."""
    return TRANSITIONS.get(state, {}).get(event, state)


def validate_transitions() -> None:
    """Raise ValueError if TRANSITIONS or TIMEOUTS are inconsistent."""
    problems = []
    for state in State:
        events = TRANSITIONS.get(state)
        if not events:
            problems.append(f"{state.name} has no transitions")
            continue
        for event, target in events.items():
            if not isinstance(event, Event):
                problems.append(f"{state.name}: {event!r} is not an Event")
            if not isinstance(target, State):
                problems.append(f"{state.name}: {event!r} points to {target!r}")

    for state, timeout in TIMEOUTS.items():
        if state not in TRANSITIONS:
            problems.append(f"timeout configured for unknown state {state!r}")
            continue
        if timeout.seconds <= 0:
            problems.append(f"{state.name}: timeout must be positive")
        if timeout.event not in TRANSITIONS[state]:
            problems.append(f"{state.name}: timeout event {timeout.event.name} is not allowed here")

    if problems:
        raise ValueError("invalid state machine table: " + "; ".join(problems))


class StateMachine:
    """
    Event driven state machine with per-state timeouts and a change log.

    ``clock`` is injectable so tests can drive time deterministically; all
    ``now`` arguments use the same clock domain (monotonic seconds).
    """

    def __init__(self, state=State.INIT, timeouts=None, clock=time.monotonic, history_limit=64):
        if history_limit < 1:
            raise ValueError("history_limit must be at least 1")
        self._clock = clock
        self._timeouts = dict(TIMEOUTS if timeouts is None else timeouts)
        self._history_limit = history_limit
        self._history: list[Transition] = []
        self._callbacks = []
        self.state = state
        self.state_since = self._clock()
        self.ignored = 0

    def _resolve(self, now):
        return self._clock() if now is None else now

    @property
    def history(self) -> tuple[Transition, ...]:
        """Recorded transitions, oldest first."""
        return tuple(self._history)

    def elapsed(self, now=None) -> float:
        """Seconds spent in the current state."""
        return self._resolve(now) - self.state_since

    def timeout(self) -> Timeout | None:
        """Timeout configured for the current state, if any."""
        return self._timeouts.get(self.state)

    def remaining(self, now=None) -> float | None:
        """Seconds until the state timeout fires, or None without timeout."""
        timeout = self.timeout()
        if timeout is None:
            return None
        return max(0.0, timeout.seconds - self.elapsed(now))

    def can_handle(self, event: Event) -> bool:
        """True if ``event`` is declared for the current state."""
        return event in TRANSITIONS.get(self.state, {})

    def handle(self, event: Event, now=None) -> State:
        """
        Apply ``event`` and return the new state.

        Events that are not allowed in the current state are counted in
        ``ignored`` and leave state, timer and history untouched.
        """
        now = self._resolve(now)
        if not self.can_handle(event):
            self.ignored += 1
            return self.state

        transition = Transition(at=now, source=self.state, event=event,
                                target=TRANSITIONS[self.state][event])
        self.state = transition.target
        self.state_since = now
        self._history.append(transition)
        if len(self._history) > self._history_limit:
            del self._history[0:len(self._history) - self._history_limit]
        for callback in list(self._callbacks):
            callback(transition)
        return self.state

    def tick(self, now=None) -> State:
        """
        Fire the state timeout if it is due.

        Returns the (possibly unchanged) state; a second call right after a
        fired timeout does nothing because the machine moved on to a state
        with its own timer.
        """
        now = self._resolve(now)
        timeout = self.timeout()
        if timeout is None or self.elapsed(now) < timeout.seconds:
            return self.state
        return self.handle(timeout.event, now)

    def reset(self, state=State.INIT, now=None) -> State:
        """
        Hard reset without history entry and without callbacks.

        Used for explicit operator resets; regular changes should go through
        ``handle`` so the pump hooks stay effective.
        """
        self.state = state
        self.state_since = self._resolve(now)
        return self.state

    def on_change(self, callback) -> None:
        """Register ``callback(Transition)``; fired after state and timer changed."""
        self._callbacks.append(callback)


def transitions_markdown() -> str:
    """TRANSITIONS as a Markdown table (used for documentation)."""
    lines = ["| State | Event | Next state |", "| --- | --- | --- |"]
    for state in State:
        for event, target in TRANSITIONS.get(state, {}).items():
            lines.append(f"| {state.name} | {event.name} | {target.name} |")
    return "\n".join(lines)


def timeouts_markdown() -> str:
    """TIMEOUTS as a Markdown table (used for documentation)."""
    lines = ["| State | Timeout (s) | Emitted event | Next state |", "| --- | --- | --- | --- |"]
    for state in State:
        timeout = TIMEOUTS.get(state)
        if timeout is None:
            lines.append(f"| {state.name} | - | - | - |")
        else:
            target = TRANSITIONS[state][timeout.event]
            lines.append(f"| {state.name} | {timeout.seconds:g} | {timeout.event.name} | {target.name} |")
    return "\n".join(lines)


if __name__ == "__main__":
    validate_transitions()
    print("## Transition table\n")
    print(transitions_markdown())
    print("\n## State timeouts\n")
    print(timeouts_markdown())
