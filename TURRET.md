# Water turret (cat deterrent) - software structure

An automatic water turret that aims at cats in the garden and gives them a
short, harmless jet of water. This document describes the software structure,
the state machine and the safety rules.

The camera, tracking and hardware wiring are **not** part of this phase: the
state machine, the safety gate and the configuration are pure Python and fully
unit tested, so the behaviour can be reviewed and tuned before any servo or
pump is connected.

## Scope

| Layer | Task | Module | Status |
| --- | --- | --- | --- |
| Perception | frames from file/webcam/synthetic | `frame_source.py` | existing (reused later) |
| Perception | motion regions (MOG2) | `motion_detector.py` | existing (reused later) |
| Perception | animal/cat classification | `classifier.py` | existing, cat labels later |
| Perception | tracking a confirmed target | `tracker.py` | existing (reused later) |
| Target selection | motion -> confirmed cat target | `target_detector.py` | planned |
| Decision | state machine, timeouts, history | `state_machine.py` | **implemented** |
| Decision | safety gate (arming, zone, budgets) | `safety.py` | **implemented** |
| Decision | parameters (timeouts, limits) | `turret_config.py` | **implemented** |
| Control | pixels -> pan/tilt angles, deadband | `aim_controller.py` | planned |
| Control | servo limits, calibration data | `calibration.py` | planned |
| Actuators | pan/tilt servo + water pump | `hardware.py` | planned |
| Orchestration | wiring everything per frame | `turret_controller.py` | planned |
| UI | CLI, status rows, aim overlay, keys | `turret.py` | planned |

The three implemented modules import nothing but the standard library, and the
tests use plain stand-in objects instead of camera frames - a state machine bug
or a missing safety check shows up in `python -m unittest`, not in the garden.

## Run and verify

```bash
python state_machine.py        # prints the tables below as Markdown
python -m unittest discover -v # 214 tests (20 pipeline + 194 turret)
```

The turret modules are expected to hold two rules:

* `state_machine.py` and `safety.py` contain no camera, tracking or hardware
  imports - everything is injected or duck-typed.
* Every veto is data (`SafetyDecision`), never a print or an exception, so the
  controller can log it, display it and map it onto an event.

## States

| State | Meaning |
| --- | --- |
| `INIT` | start-up: open sources and actuators, load calibration, warm up |
| `IDLE` | powered, not scanning (disarmed, or a verification just finished) |
| `SEARCHING` | scanning the garden for a cat |
| `TARGET_FOUND` | a candidate was spotted, waiting for the confirmation window |
| `AIMING` | servos are pointing at the target |
| `WATERING` | the pump is on |
| `VERIFYING` | watching whether the cat moved away |
| `CALIBRATION` | measuring servo limits and field of view |
| `ERROR` | fault; only an operator `RESET` or a calibration leaves it |

## Events

| Event | Emitted by |
| --- | --- |
| `INIT_DONE`, `INIT_FAILED` | start-up sequence |
| `ARM`, `DISARM`, `RESET` | operator (CLI key / switch) |
| `TARGET_SPOTTED` | motion detector plus classifier |
| `TARGET_CONFIRMED` | confirmation counter reached `confirm_frames` |
| `TARGET_LOST` | tracker lost the target or the acquisition window expired |
| `AIM_SETTLED`, `AIM_TIMEOUT` | aim controller |
| `SPRAY_FINISHED` | state timeout of `WATERING` (nominal spray time) |
| `TARGET_REPELLED`, `TARGET_STILL_PRESENT` | verification step |
| `SPRAY_BUDGET_EXHAUSTED` | safety stats (per-target or hourly budget) |
| `SAFETY_BLOCK`, `SAFETY_CLEARED` | safety gate veto |
| `CALIBRATION_REQUESTED`, `CALIBRATION_DONE`, `CALIBRATION_FAILED` | calibration |
| `HARDWARE_FAULT` | any actuator or camera error (latched) |

## State diagram

```
INIT --INIT_DONE--> IDLE --ARM/TARGET_SPOTTED--> SEARCHING
                      ^                              |
                      |                        TARGET_SPOTTED
             DISARM/SAFETY_BLOCK                  v
                      |                        TARGET_FOUND
                      |                              |
                      |                       TARGET_CONFIRMED
                      |                              v
                      |                           AIMING --AIM_SETTLED--> WATERING
                      |                              ^      |                |
                      |                     TARGET_STILL_PRESENT             | 0.3 s
                      |                              |      v                v
                      +--TARGET_REPELLED/BUDGET-- VERIFYING <----- SPRAY_FINISHED
                                      |
                                      +--TARGET_LOST/VERIFY_TIMEOUT--> SEARCHING

CALIBRATION <--> IDLE                 HARDWARE_FAULT (from any active state) --> ERROR
                                      ERROR --RESET--> INIT
```

## Transition table

Generated with `python state_machine.py`:

| State | Event | Next state |
| --- | --- | --- |
| INIT | INIT_DONE | IDLE |
| INIT | INIT_FAILED | ERROR |
| INIT | CALIBRATION_REQUESTED | CALIBRATION |
| IDLE | ARM | SEARCHING |
| IDLE | TARGET_SPOTTED | SEARCHING |
| IDLE | CALIBRATION_REQUESTED | CALIBRATION |
| IDLE | HARDWARE_FAULT | ERROR |
| SEARCHING | TARGET_SPOTTED | TARGET_FOUND |
| SEARCHING | DISARM | IDLE |
| SEARCHING | SAFETY_BLOCK | IDLE |
| SEARCHING | CALIBRATION_REQUESTED | CALIBRATION |
| SEARCHING | HARDWARE_FAULT | ERROR |
| TARGET_FOUND | TARGET_CONFIRMED | AIMING |
| TARGET_FOUND | TARGET_LOST | SEARCHING |
| TARGET_FOUND | SAFETY_BLOCK | IDLE |
| TARGET_FOUND | HARDWARE_FAULT | ERROR |
| AIMING | AIM_SETTLED | WATERING |
| AIMING | AIM_TIMEOUT | SEARCHING |
| AIMING | TARGET_LOST | SEARCHING |
| AIMING | SAFETY_BLOCK | IDLE |
| AIMING | HARDWARE_FAULT | ERROR |
| WATERING | SPRAY_FINISHED | VERIFYING |
| WATERING | SAFETY_BLOCK | ERROR |
| WATERING | HARDWARE_FAULT | ERROR |
| VERIFYING | TARGET_REPELLED | IDLE |
| VERIFYING | TARGET_STILL_PRESENT | AIMING |
| VERIFYING | SPRAY_BUDGET_EXHAUSTED | IDLE |
| VERIFYING | TARGET_LOST | SEARCHING |
| VERIFYING | VERIFY_TIMEOUT | SEARCHING |
| VERIFYING | HARDWARE_FAULT | ERROR |
| CALIBRATION | CALIBRATION_DONE | IDLE |
| CALIBRATION | CALIBRATION_FAILED | ERROR |
| CALIBRATION | HARDWARE_FAULT | ERROR |
| ERROR | RESET | INIT |
| ERROR | CALIBRATION_REQUESTED | CALIBRATION |

Rules enforced by tests:

* An event that is not listed for the current state is ignored and counted in
  `StateMachine.ignored` - a wrong sensor reading can never move the machine.
* `WATERING` can only be left through `SPRAY_FINISHED`, `SAFETY_BLOCK` or
  `HARDWARE_FAULT`. A safety veto while the pump runs is treated as a fault
  (`ERROR`) because something had to cut the water.
* `ERROR` is latched: only `RESET` (operator) or `CALIBRATION_REQUESTED` leaves it.
* `HARDWARE_FAULT` leads to `ERROR` from every active state.

## State timeouts

Automatically emitted events after a nominal duration:

| State | Timeout (s) | Emitted event | Next state |
| --- | --- | --- | --- |
| INIT | - | - | - |
| IDLE | - | - | - |
| SEARCHING | - | - | - |
| TARGET_FOUND | 1 | TARGET_LOST | SEARCHING |
| AIMING | 2 | AIM_TIMEOUT | SEARCHING |
| WATERING | 0.3 | SPRAY_FINISHED | VERIFYING |
| VERIFYING | 5 | VERIFY_TIMEOUT | SEARCHING |
| CALIBRATION | - | - | - |
| ERROR | - | - | - |

`StateMachine.tick(now)` fires the timeout of the current state if it is due.
After a fired timeout the machine is in a state with its own timer, so a second
call does nothing. Values come from `TurretConfig.state` via
`state_machine.default_timeouts(...)`, so no timing constant is hard coded twice.

## Safety gate

`safety.SafetyGuard` answers two questions per frame:

* `check_aim(target, frame_shape, now)` - may the servos move?
* `check_spray(target, frame_shape, now)` - may the pump run?

Both return a `SafetyDecision(allowed, code, message)`. The checks run in a
fixed order (`VETO_PRIORITY`, enforced by a test) and the first hit wins:

| # | Code | Trigger |
| --- | --- | --- |
| 1 | `emergency_stop` | operator stop is latched (also disarms) |
| 2 | `not_armed` | turret is disarmed (default!) |
| 3 | `no_frame` | no frame yet, or older than `frame_timeout_s` |
| 4 | `no_target` | `target is None` |
| 5 | `label_not_allowed` | label is not in `target_labels` (only `cat`) |
| 6 | `target_not_confirmed` | fewer than `confirm_frames` confirmations |
| 7 | `confidence_too_low` | confidence below `min_confidence` |
| 8 | `outside_spray_zone` | target centre outside the relative spray zone |
| 9 | `target_too_small` | target smaller than `min_target_height` |
| 10 | `target_too_large` | target bigger than `max_target_height` (too close) |
| 11 | `spray_in_progress` | a spray is already registered as running |
| 12 | `budget_target_exhausted` | `max_sprays_per_target` used up for this cat |
| 13 | `budget_hourly_exhausted` | `max_sprays_per_hour` used up (rolling hour) |
| 14 | `cooldown_active` | less than `cooldown_s` since the last spray |
| 15 | `outside_time_window` | local hour outside `allowed_hours` |

Codes 11-15 only apply to spraying. Aiming is intentionally more permissive:
cooldown, budgets and the time window gate the water, not the servos.

Fixed entry point 1-2 means a **disarmed turret neither aims nor sprays**, even
if the state machine sent a wrong event. `emergency_stop()` latches the veto;
`clear_emergency_stop()` only clears the latch and keeps the turret disarmed, so
the operator has to arm it again on purpose.

Budget codes are collected in `BUDGET_CODES`; the controller maps them onto
`SPRAY_BUDGET_EXHAUSTED`, every other veto onto `SAFETY_BLOCK`.

`SafetyGuard` never touches hardware: it only allows or vetoes. Bookkeeping that
must be exact lives next to it:

* `register_frame(now)` - once per processed frame (staleness check)
* `register_spray_start(now)` / `register_spray_end(now)` - count and cooldown
* `spray_overrun(now)` - `True` when a running spray exceeds `max_spray_s`
  (the controller must switch the pump off immediately)
* `target_left()` - resets the per-target budget once the cat is gone
* `stats()` - snapshot of the counters (`sprays_total`, `sprays_last_hour`,
  `sprays_for_target`, `last_spray_end`, `veto_counts`)

## Configuration

`turret_config.TurretConfig` holds everything in one object
(`state` = `StateTimeouts`, `safety` = `SafetyLimits`) and can be written and
read as UTF-8 JSON (`save`/`load`). Defaults are deliberately timid:

| Parameter | Default | Purpose |
| --- | --- | --- |
| `armed` | `False` | turret must be armed on purpose |
| `target_labels` | `("cat",)` | only cats are sprayed |
| `min_confidence` | `0.6` | classifier confidence floor |
| `confirm_frames` | `3` | frames needed before a target counts |
| `spray_zone` | `(0.05, 0.05, 0.95, 0.95)` | relative x0, y0, x1, y1 of the frame |
| `min_target_height` / `max_target_height` | `0.02` / `0.9` | useful vs. too close |
| `cooldown_s` | `3.0` | pause between two sprays |
| `max_spray_s` | `0.4` | hard cap per burst |
| `max_sprays_per_target` | `2` | budget for one cat in one visit |
| `max_sprays_per_hour` | `12` | rolling one hour window |
| `allowed_hours` | `(6, 22)` | local time, end exclusive, `None` = always |
| `frame_timeout_s` | `1.0` | stale frame = no spraying |
| `aim_s` / `spray_s` / `verify_s` / `target_confirm_s` | `2.0` / `0.3` / `5.0` / `1.0` | state timeouts |

Validation is strict on purpose: unknown JSON keys, reversed zones, non-positive
durations, `max_sprays_per_target = 0` and `state.spray_s > safety.max_spray_s`
all raise `ValueError` at load time instead of misbehaving in the garden.

## Planned interfaces (next phases)

The implemented modules are already written against these shapes, so the
following phases only add files:

```python
# target_detector.py
@dataclass
class Target:
    box: tuple                 # x, y, w, h in the frame
    center: tuple              # cx, cy
    label: str                 # must be in target_labels ("cat")
    confidence: float
    frames_confirmed: int
    first_seen_at: float
    last_seen_at: float

class CatTargetDetector:       # motion -> classify -> confirm -> track
    def __init__(self, detector, classifier, tracker, config): ...
    def update(self, frame, now) -> Target | None: ...
    def reset(self) -> None: ...

# aim_controller.py
def pixel_to_angle(dx_px, frame_width, hfov_deg) -> float: ...
@dataclass(frozen=True)
class AimError:
    dx_px: float; dy_px: float
    pan_deg: float; tilt_deg: float
    distance_px: float; aimed: bool

class AimController:           # proportional, rate limited, deadband
    def error(self, target_center, frame_shape) -> AimError: ...
    def step(self, target_center, frame_shape, dt_s) -> tuple: ...
    def is_aimed(self, error) -> bool: ...

# hardware.py
class PanTiltActuator(ABC):    # point_at, position, center, close
class WaterPump(ABC):          # on, off, is_on, close
class SimulatedPanTilt / SimulatedWaterPump   # full implementations
class SerialPanTilt / GpioWaterPump           # skeletons with lazy imports
def create_hardware(kind, **kwargs): ...

# turret_controller.py
class TurretController:
    def __init__(self, detector, aimer, guard, machine, actuator, pump, config): ...
    def step(self, frame, now) -> TurretStatus: ...
    def close(self) -> None: ...
```

Invariants the controller must enforce (and that the tests of phase B check):

1. The pump is **only** on in `WATERING`. Leaving `WATERING` for any reason -
   transition, `spray_overrun`, exception, `close()` - switches it off.
2. `HARDWARE_FAULT` is emitted whenever an actuator call raises or the frame
   source stops delivering frames.
3. A veto while `AIMING` means: stop moving, `pump.off()`, emit `SAFETY_BLOCK`.
4. `SafetyGuard.register_frame()` is called once per processed frame; without it
   the turret vetoes everything.

## Operation modes

| Mode | Behaviour |
| --- | --- |
| disarmed (default) | no movement, no water; the state machine stays in `IDLE` |
| armed (`--arm`) | full cycle, water allowed subject to all vouchers above |
| dry run (`--dry-run`) | full cycle and full logging, pump commands are simulated/logged only |
| calibration (`--calibrate` or key) | `CALIBRATION` state: measure limits and field of view, store JSON |

## Calibration (planned)

1. Warm up the background model with `warmup_frames` empty frames.
2. Home the servos towards their soft limits, then to the centre position.
3. Measure the field of view: a marker at a known pixel position is moved until
   it reaches the frame edge; the servo angle difference gives degrees per pixel.
4. Store `Calibration` (centre, limits, degrees per second, field of view,
   tolerance, `created`) next to the config; `INIT` fails into `ERROR` when the
   file is missing and calibration is not requested.

## Hardware and safety notes

* Servos and pump need their own supply; a shared rail causes brown-outs and
  random reboots of the camera/SBC.
* Switch the pump with a MOSFET or relay plus a flyback diode, never directly
  from a GPIO.
* Enforce `max_spray_s` a second time in the firmware of the servo/pump board -
  a crashed Python process must not be able to water forever.
* Fit soft limits and, if possible, end switches; the aim controller clamps to
  `pan_limits` / `tilt_limits` from the calibration file.
* Provide a physical emergency stop that cuts the pump power, plus the software
  `emergency_stop()` path.
* Only spray when a cat is confirmed; never aim at people, other animals or the
  neighbour's window. Prefer short, low-pressure bursts, no additives, and do
  not run the turret below freezing.
* Log every spray (time, target label, confidence, aim error) so a false
  positive can be reconstructed afterwards.

## What is deliberately missing

* Real cat detection: `classifier.MockClassifier` is rule based and would spray
  at anything large enough. Replace it (later `RknnClassifier`) before arming
  for real - everything else stays unchanged.
* Camera/tracking wiring, servo and pump drivers, CLI and display: phase B.

