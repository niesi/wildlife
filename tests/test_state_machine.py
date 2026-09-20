import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from unittest.mock import patch

from state_machine import (
    TIMEOUTS,
    TRANSITIONS,
    Event,
    State,
    StateMachine,
    Timeout,
    default_timeouts,
    resolve_transition,
    timeouts_markdown,
    transitions_markdown,
    validate_transitions,
)


class FakeClock:
    """Deterministic stand-in for time.monotonic."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


def machine_at(*events, clock=None, **kwargs):
    """Build a machine that already walked through the given events."""
    if clock is not None:
        kwargs["clock"] = clock
    machine = StateMachine(**kwargs)
    for event in events:
        machine.handle(event)
    return machine


class TransitionTableTest(unittest.TestCase):
    def test_table_is_consistent(self):
        validate_transitions()

    def test_every_state_has_transitions(self):
        for state in State:
            with self.subTest(state=state):
                self.assertTrue(TRANSITIONS.get(state))

    def test_transitions_only_use_events_and_states(self):
        for state, events in TRANSITIONS.items():
            for event, target in events.items():
                with self.subTest(state=state, event=event):
                    self.assertIsInstance(event, Event)
                    self.assertIsInstance(target, State)

    def test_resolve_transition_keeps_state_for_unknown_event(self):
        self.assertIs(resolve_transition(State.IDLE, Event.SPRAY_FINISHED), State.IDLE)

    def test_resolve_transition_keeps_unknown_state(self):
        sentinel = object()
        self.assertIs(resolve_transition(sentinel, Event.RESET), sentinel)

    def test_validate_rejects_state_without_transitions(self):
        with patch.dict(TRANSITIONS, {State.IDLE: {}}):
            with self.assertRaises(ValueError) as context:
                validate_transitions()
        self.assertIn("IDLE has no transitions", str(context.exception))

    def test_validate_rejects_timeout_event_not_allowed_in_state(self):
        with patch.dict(TIMEOUTS, {State.IDLE: Timeout(1.0, Event.SPRAY_FINISHED)}):
            with self.assertRaises(ValueError) as context:
                validate_transitions()
        self.assertIn("SPRAY_FINISHED", str(context.exception))

    def test_validate_rejects_non_positive_timeout(self):
        with patch.dict(TIMEOUTS, {State.AIMING: Timeout(0.0, Event.AIM_TIMEOUT)}):
            with self.assertRaises(ValueError) as context:
                validate_transitions()
        self.assertIn("positive", str(context.exception))


class SafetyRelevantRulesTest(unittest.TestCase):
    def test_watering_only_leaves_through_safety_events(self):
        self.assertEqual(
            set(TRANSITIONS[State.WATERING]),
            {Event.SPRAY_FINISHED, Event.SAFETY_BLOCK, Event.HARDWARE_FAULT},
        )
        self.assertIs(TRANSITIONS[State.WATERING][Event.SAFETY_BLOCK], State.ERROR)

    def test_error_only_leaves_via_reset_or_calibration(self):
        self.assertEqual(set(TRANSITIONS[State.ERROR]), {Event.RESET, Event.CALIBRATION_REQUESTED})

    def test_hardware_fault_leads_to_error_from_every_active_state(self):
        active = (State.IDLE, State.SEARCHING, State.TARGET_FOUND, State.AIMING,
                  State.WATERING, State.VERIFYING, State.CALIBRATION)
        for state in active:
            with self.subTest(state=state):
                self.assertIs(TRANSITIONS[state][Event.HARDWARE_FAULT], State.ERROR)

    def test_nominal_cycle_walks_through_all_working_states(self):
        machine = StateMachine()
        steps = (
            (Event.INIT_DONE, State.IDLE),
            (Event.ARM, State.SEARCHING),
            (Event.TARGET_SPOTTED, State.TARGET_FOUND),
            (Event.TARGET_CONFIRMED, State.AIMING),
            (Event.AIM_SETTLED, State.WATERING),
            (Event.SPRAY_FINISHED, State.VERIFYING),
            (Event.TARGET_REPELLED, State.IDLE),
        )
        for event, expected in steps:
            with self.subTest(event=event):
                self.assertIs(machine.handle(event), expected)

    def test_stubborn_target_gets_a_second_spray_before_budget_ends(self):
        machine = machine_at(Event.INIT_DONE, Event.ARM, Event.TARGET_SPOTTED,
                             Event.TARGET_CONFIRMED, Event.AIM_SETTLED, Event.SPRAY_FINISHED)
        self.assertIs(machine.state, State.VERIFYING)
        self.assertIs(machine.handle(Event.TARGET_STILL_PRESENT), State.AIMING)
        self.assertIs(machine.handle(Event.AIM_SETTLED), State.WATERING)
        self.assertIs(machine.handle(Event.SPRAY_FINISHED), State.VERIFYING)
        self.assertIs(machine.handle(Event.SPRAY_BUDGET_EXHAUSTED), State.IDLE)

    def test_disarm_and_safety_block_return_to_idle(self):
        for event in (Event.DISARM, Event.SAFETY_BLOCK):
            with self.subTest(event=event):
                machine = machine_at(Event.INIT_DONE, Event.ARM)
                self.assertIs(machine.handle(event), State.IDLE)


class StateMachineBehaviourTest(unittest.TestCase):
    def test_ignored_event_keeps_state_and_is_counted(self):
        machine = StateMachine()
        self.assertFalse(machine.can_handle(Event.SPRAY_FINISHED))

        self.assertIs(machine.handle(Event.SPRAY_FINISHED), State.INIT)

        self.assertEqual(machine.ignored, 1)
        self.assertEqual(machine.history, ())

    def test_on_change_receives_transition_with_timestamp(self):
        clock = FakeClock(100.0)
        machine = StateMachine(clock=clock)
        seen = []
        machine.on_change(seen.append)

        clock.advance(2.5)
        machine.handle(Event.INIT_DONE)

        self.assertEqual(len(seen), 1)
        transition = seen[0]
        self.assertEqual(transition.at, 102.5)
        self.assertIs(transition.source, State.INIT)
        self.assertIs(transition.event, Event.INIT_DONE)
        self.assertIs(transition.target, State.IDLE)
        self.assertIs(machine.state, State.IDLE)

    def test_every_callback_is_called(self):
        machine = StateMachine()
        first, second = [], []
        machine.on_change(first.append)
        machine.on_change(second.append)

        machine.handle(Event.INIT_DONE)

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)

    def test_history_is_capped_and_keeps_the_newest(self):
        machine = StateMachine(history_limit=2)

        machine.handle(Event.INIT_DONE)
        machine.handle(Event.ARM)
        machine.handle(Event.TARGET_SPOTTED)

        self.assertEqual([t.event for t in machine.history], [Event.ARM, Event.TARGET_SPOTTED])

    def test_history_limit_must_be_positive(self):
        with self.assertRaises(ValueError):
            StateMachine(history_limit=0)

    def test_reset_switches_state_without_history_or_callback(self):
        clock = FakeClock()
        machine = machine_at(Event.INIT_DONE, Event.ARM, clock=clock)
        seen = []
        machine.on_change(seen.append)
        clock.advance(3.0)

        self.assertIs(machine.reset(State.INIT), State.INIT)

        self.assertEqual(machine.elapsed(), 0.0)
        self.assertEqual(seen, [])
        self.assertEqual(len(machine.history), 2)

    def test_custom_timeouts_are_copied(self):
        clock = FakeClock()
        machine = StateMachine(state=State.IDLE,
                               timeouts={State.IDLE: Timeout(1.0, Event.ARM)},
                               clock=clock)

        clock.advance(1.0)

        self.assertIs(machine.tick(), State.SEARCHING)
        self.assertNotIn(State.IDLE, TIMEOUTS)


class StateMachineTimingTest(unittest.TestCase):
    def test_aiming_timeout_fires_once(self):
        clock = FakeClock()
        machine = machine_at(Event.INIT_DONE, Event.ARM, Event.TARGET_SPOTTED,
                             Event.TARGET_CONFIRMED, clock=clock)
        self.assertIs(machine.state, State.AIMING)
        self.assertIs(machine.timeout().event, Event.AIM_TIMEOUT)
        self.assertAlmostEqual(machine.remaining(), 2.0)

        clock.advance(1.9)
        self.assertIs(machine.tick(), State.AIMING)

        clock.advance(0.1)
        self.assertIs(machine.tick(), State.SEARCHING)

        self.assertIs(machine.tick(), State.SEARCHING)
        self.assertEqual(machine.ignored, 0)

    def test_spray_and_verification_timeouts(self):
        clock = FakeClock()
        machine = machine_at(Event.INIT_DONE, Event.ARM, Event.TARGET_SPOTTED,
                             Event.TARGET_CONFIRMED, Event.AIM_SETTLED, clock=clock)
        self.assertIs(machine.state, State.WATERING)

        clock.advance(0.3)
        self.assertIs(machine.tick(), State.VERIFYING)

        clock.advance(4.0)
        self.assertIs(machine.tick(), State.VERIFYING)

        clock.advance(1.0)
        self.assertIs(machine.tick(), State.SEARCHING)

    def test_candidate_timeout_loses_the_target(self):
        clock = FakeClock()
        machine = machine_at(Event.INIT_DONE, Event.ARM, Event.TARGET_SPOTTED, clock=clock)

        clock.advance(0.9)
        self.assertIs(machine.tick(), State.TARGET_FOUND)

        clock.advance(0.1)
        self.assertIs(machine.tick(), State.SEARCHING)

    def test_states_without_timeout_never_tick(self):
        for events in ((Event.INIT_DONE,), (Event.INIT_DONE, Event.ARM)):
            with self.subTest(events=events):
                clock = FakeClock()
                machine = machine_at(*events, clock=clock)
                self.assertIsNone(machine.timeout())
                self.assertIsNone(machine.remaining())
                clock.advance(600.0)
                self.assertIs(machine.tick(), machine.state)

    def test_elapsed_tracks_the_state_entry(self):
        clock = FakeClock(50.0)
        machine = StateMachine(clock=clock)

        clock.advance(4.0)
        self.assertEqual(machine.elapsed(), 4.0)
        self.assertIsNone(machine.remaining())

        machine.handle(Event.INIT_DONE)
        self.assertEqual(machine.elapsed(), 0.0)

    def test_explicit_now_overrides_the_clock(self):
        machine = machine_at(Event.INIT_DONE, Event.ARM, Event.TARGET_SPOTTED, Event.TARGET_CONFIRMED)

        self.assertIs(machine.tick(now=machine.state_since + 2.0), State.SEARCHING)


class TimeoutConfigurationTest(unittest.TestCase):
    def test_default_timeouts_match_the_module_constant(self):
        self.assertEqual(default_timeouts(), TIMEOUTS)

    def test_default_timeouts_map_states_to_events(self):
        timeouts = default_timeouts(aim_s=1.0, spray_s=2.0, verify_s=3.0, target_confirm_s=4.0)

        self.assertEqual(timeouts, {
            State.AIMING: Timeout(1.0, Event.AIM_TIMEOUT),
            State.WATERING: Timeout(2.0, Event.SPRAY_FINISHED),
            State.VERIFYING: Timeout(3.0, Event.VERIFY_TIMEOUT),
            State.TARGET_FOUND: Timeout(4.0, Event.TARGET_LOST),
        })

    def test_states_that_must_not_time_out(self):
        for state in (State.INIT, State.IDLE, State.SEARCHING, State.CALIBRATION, State.ERROR):
            with self.subTest(state=state):
                self.assertNotIn(state, TIMEOUTS)


class MarkdownDocumentationTest(unittest.TestCase):
    def test_transition_table_lists_every_transition(self):
        lines = transitions_markdown().splitlines()

        self.assertEqual(lines[0], "| State | Event | Next state |")
        self.assertEqual(len(lines) - 2, sum(len(events) for events in TRANSITIONS.values()))
        self.assertIn("| WATERING | SPRAY_FINISHED | VERIFYING |", lines)

    def test_timeout_table_marks_states_without_timeout(self):
        lines = timeouts_markdown().splitlines()

        self.assertEqual(lines[0], "| State | Timeout (s) | Emitted event | Next state |")
        self.assertIn("| IDLE | - | - | - |", lines)
        self.assertIn("| AIMING | 2 | AIM_TIMEOUT | SEARCHING |", lines)
        self.assertIn("| WATERING | 0.3 | SPRAY_FINISHED | VERIFYING |", lines)


if __name__ == "__main__":
    unittest.main()
