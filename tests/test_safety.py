import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
from datetime import datetime
from types import SimpleNamespace

from safety import (
    BUDGET_CODES,
    BUDGET_HOURLY_EXHAUSTED,
    BUDGET_TARGET_EXHAUSTED,
    COOLDOWN_ACTIVE,
    CONFIDENCE_TOO_LOW,
    EMERGENCY_STOP,
    LABEL_NOT_ALLOWED,
    NO_FRAME,
    NO_TARGET,
    NOT_ARMED,
    OK,
    OUTSIDE_SPRAY_ZONE,
    OUTSIDE_TIME_WINDOW,
    SPRAY_IN_PROGRESS,
    TARGET_NOT_CONFIRMED,
    TARGET_TOO_LARGE,
    TARGET_TOO_SMALL,
    VETO_PRIORITY,
    SafetyDecision,
    SafetyGuard,
    SafetyStats,
)
from turret_config import SafetyLimits


FRAME_SHAPE = (240, 320)  # rows (height), columns (width)


class FakeClock:
    """Deterministic stand-in for time.monotonic."""

    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


class FixedWallClock:
    """Deterministic stand-in for datetime.now."""

    def __init__(self, hour=12, minute=0):
        self.hour = hour
        self.minute = minute

    def __call__(self):
        return datetime(2024, 1, 1, self.hour, self.minute)

    def set_hour(self, hour):
        self.hour = hour


def target(label="cat", center=(160, 120), box=(140, 100, 40, 40),
           confidence=0.9, frames_confirmed=3):
    return SimpleNamespace(label=label, center=center, box=box,
                           confidence=confidence, frames_confirmed=frames_confirmed)


def armed_guard(**overrides):
    """Armed guard with one fresh frame registered; returns (guard, clock)."""
    clock = FakeClock()
    guard = SafetyGuard(SafetyLimits(armed=True, **overrides),
                        clock=clock, wall_clock=FixedWallClock())
    guard.register_frame()
    return guard, clock


class ArmingTest(unittest.TestCase):
    def test_default_guard_uses_safe_limits(self):
        guard = SafetyGuard()

        self.assertEqual(guard.limits, SafetyLimits())
        self.assertFalse(guard.armed)
        self.assertFalse(guard.stopped)

    def test_disarmed_guard_vetoes_aiming_and_spraying(self):
        guard, _ = armed_guard()
        guard.disarm()
        guard.register_frame()

        for check in (guard.check_aim, guard.check_spray):
            with self.subTest(check=check.__name__):
                decision = check(target(), FRAME_SHAPE)
                self.assertFalse(decision.allowed)
                self.assertEqual(decision.code, NOT_ARMED)

    def test_armed_guard_releases_aiming_and_spraying(self):
        guard, _ = armed_guard()

        self.assertTrue(guard.check_aim(target(), FRAME_SHAPE).allowed)
        self.assertEqual(guard.check_spray(target(), FRAME_SHAPE).code, OK)

    def test_emergency_stop_overrides_arming(self):
        guard, _ = armed_guard()

        guard.emergency_stop("operator hit the button")

        decision = guard.check_spray(target(), FRAME_SHAPE)
        self.assertEqual(decision.code, EMERGENCY_STOP)
        self.assertIn("operator hit the button", decision.message)
        self.assertTrue(guard.stopped)
        self.assertFalse(guard.armed)

    def test_clearing_the_stop_requires_arming_again(self):
        guard, _ = armed_guard()
        guard.emergency_stop()

        guard.clear_emergency_stop()

        self.assertFalse(guard.stopped)
        self.assertFalse(guard.armed)
        self.assertEqual(guard.check_spray(target(), FRAME_SHAPE).code, NOT_ARMED)

        guard.arm()
        self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)


class FrameFreshnessTest(unittest.TestCase):
    def test_guard_without_frame_vetoes(self):
        guard = SafetyGuard(SafetyLimits(armed=True))

        decision = guard.check_spray(target(), FRAME_SHAPE)

        self.assertEqual(decision.code, NO_FRAME)

    def test_stale_frame_vetoes(self):
        guard, clock = armed_guard()

        clock.advance(1.01)
        self.assertEqual(guard.check_spray(target(), FRAME_SHAPE).code, NO_FRAME)

        guard.register_frame()
        self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)

    def test_frame_age_at_the_limit_is_still_fresh(self):
        guard, clock = armed_guard()

        clock.advance(1.0)

        self.assertTrue(guard.check_aim(target(), FRAME_SHAPE).allowed)


class TargetClassTest(unittest.TestCase):
    def test_missing_target_vetoes(self):
        guard, _ = armed_guard()

        self.assertEqual(guard.check_aim(None, FRAME_SHAPE).code, NO_TARGET)
        self.assertEqual(guard.check_spray(None, FRAME_SHAPE).code, NO_TARGET)

    def test_foreign_label_vetoes(self):
        guard, _ = armed_guard()

        decision = guard.check_spray(target(label="dog"), FRAME_SHAPE)

        self.assertEqual(decision.code, LABEL_NOT_ALLOWED)
        self.assertIn("dog", decision.message)

    def test_several_labels_can_be_allowed(self):
        guard, _ = armed_guard(target_labels=("cat", "kitten"))

        self.assertTrue(guard.check_spray(target(label="kitten"), FRAME_SHAPE).allowed)

    def test_unconfirmed_target_vetoes(self):
        guard, _ = armed_guard()

        decision = guard.check_spray(target(frames_confirmed=2), FRAME_SHAPE)

        self.assertEqual(decision.code, TARGET_NOT_CONFIRMED)
        self.assertIn("2/3", decision.message)

    def test_low_confidence_vetoes(self):
        guard, _ = armed_guard()

        decision = guard.check_spray(target(confidence=0.45), FRAME_SHAPE)

        self.assertEqual(decision.code, CONFIDENCE_TOO_LOW)
        self.assertIn("0.45", decision.message)

    def test_missing_attributes_count_as_unconfirmed(self):
        guard, _ = armed_guard()

        decision = guard.check_spray(SimpleNamespace(label="cat"), FRAME_SHAPE)

        self.assertEqual(decision.code, TARGET_NOT_CONFIRMED)


class TargetPositionTest(unittest.TestCase):
    def test_centre_is_derived_from_the_box(self):
        guard, _ = armed_guard()

        decision = guard.check_spray(target(center=None), FRAME_SHAPE)

        self.assertTrue(decision.allowed)

    def test_outside_zone_vetoes_on_all_edges(self):
        guard, _ = armed_guard()
        cases = {"left": (2, 120), "right": (318, 120), "top": (160, 2), "bottom": (160, 238)}

        for edge, center in cases.items():
            with self.subTest(edge=edge):
                decision = guard.check_spray(target(center=center), FRAME_SHAPE)
                self.assertEqual(decision.code, OUTSIDE_SPRAY_ZONE)

    def test_target_without_position_vetoes(self):
        guard, _ = armed_guard()
        flat = SimpleNamespace(label="cat", confidence=0.9, frames_confirmed=3)

        self.assertEqual(guard.check_aim(flat, FRAME_SHAPE).code, NO_TARGET)

    def test_unknown_frame_shape_vetoes(self):
        guard, _ = armed_guard()

        for shape in (None, (), (0, 0)):
            with self.subTest(shape=shape):
                self.assertEqual(guard.check_aim(target(), shape).code, NO_FRAME)

    def test_too_small_and_too_large_targets_veto(self):
        guard, _ = armed_guard()

        small = guard.check_spray(target(box=(150, 110, 20, 2)), FRAME_SHAPE)
        large = guard.check_spray(target(box=(140, 100, 40, 225)), FRAME_SHAPE)

        self.assertEqual(small.code, TARGET_TOO_SMALL)
        self.assertEqual(large.code, TARGET_TOO_LARGE)


class SprayBudgetTest(unittest.TestCase):
    def test_cooldown_blocks_the_second_spray_but_not_aiming(self):
        guard, clock = armed_guard(cooldown_s=3.0)
        guard.register_spray_start()
        guard.register_spray_end()

        self.assertEqual(guard.check_spray(target(), FRAME_SHAPE).code, COOLDOWN_ACTIVE)
        self.assertTrue(guard.check_aim(target(), FRAME_SHAPE).allowed)

        clock.advance(3.0)
        guard.register_frame()
        self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)

    def test_running_spray_blocks_another_spray(self):
        guard, _ = armed_guard(cooldown_s=0.0)

        guard.register_spray_start()
        self.assertTrue(guard.spraying)
        self.assertEqual(guard.check_spray(target(), FRAME_SHAPE).code, SPRAY_IN_PROGRESS)

        guard.register_spray_end()
        self.assertFalse(guard.spraying)
        self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)

    def test_per_target_budget_is_exhausted_and_reset(self):
        guard, _ = armed_guard(max_sprays_per_target=2, cooldown_s=0.0)

        for _ in range(2):
            self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)
            guard.register_spray_start()
            guard.register_spray_end()
        self.assertEqual(guard.check_spray(target(), FRAME_SHAPE).code, BUDGET_TARGET_EXHAUSTED)

        guard.target_left()

        self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)

    def test_hourly_budget_slides_with_time(self):
        guard, clock = armed_guard(max_sprays_per_hour=2, max_sprays_per_target=99, cooldown_s=0.0)

        for _ in range(2):
            guard.register_spray_start()
            guard.register_spray_end()
        self.assertEqual(guard.check_spray(target(), FRAME_SHAPE).code, BUDGET_HOURLY_EXHAUSTED)

        clock.advance(3601.0)
        guard.register_frame()

        self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)

    def test_time_window_blocks_outside_the_allowed_hours(self):
        wall = FixedWallClock(hour=3)
        guard = SafetyGuard(SafetyLimits(armed=True, allowed_hours=(6, 22)),
                            clock=FakeClock(), wall_clock=wall)
        guard.register_frame()

        self.assertEqual(guard.check_spray(target(), FRAME_SHAPE).code, OUTSIDE_TIME_WINDOW)

        wall.set_hour(6)
        self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)

    def test_time_window_can_be_disabled(self):
        guard = SafetyGuard(SafetyLimits(armed=True, allowed_hours=None),
                            clock=FakeClock(), wall_clock=FixedWallClock(hour=3))
        guard.register_frame()

        self.assertTrue(guard.check_spray(target(), FRAME_SHAPE).allowed)

    def test_time_window_does_not_block_aiming(self):
        guard = SafetyGuard(SafetyLimits(armed=True, allowed_hours=(6, 22)),
                            clock=FakeClock(), wall_clock=FixedWallClock(hour=3))
        guard.register_frame()

        self.assertTrue(guard.check_aim(target(), FRAME_SHAPE).allowed)


class SprayDurationTest(unittest.TestCase):
    def test_overrun_is_reported_and_cleared(self):
        guard, clock = armed_guard(max_spray_s=0.4)

        self.assertFalse(guard.spray_overrun())

        guard.register_spray_start()
        clock.advance(0.4)
        self.assertFalse(guard.spray_overrun())

        clock.advance(0.01)
        self.assertTrue(guard.spray_overrun())

        guard.register_spray_end()
        self.assertFalse(guard.spray_overrun())


class StatsTest(unittest.TestCase):
    def test_vetoes_are_counted_by_code(self):
        guard, _ = armed_guard()

        guard.check_spray(target(label="dog"), FRAME_SHAPE)
        guard.check_spray(target(label="dog"), FRAME_SHAPE)
        guard.check_spray(None, FRAME_SHAPE)

        self.assertEqual(guard.stats().veto_counts, {LABEL_NOT_ALLOWED: 2, NO_TARGET: 1})

    def test_spray_counters_and_last_end(self):
        guard, clock = armed_guard(max_sprays_per_hour=5)

        guard.register_spray_start()
        clock.advance(0.3)
        guard.register_spray_end()

        stats = guard.stats()

        self.assertEqual(stats.sprays_total, 1)
        self.assertEqual(stats.sprays_last_hour, 1)
        self.assertEqual(stats.sprays_for_target, 1)
        self.assertEqual(stats.last_spray_end, 0.3)

    def test_stats_are_a_snapshot(self):
        guard, _ = armed_guard()

        stats = guard.stats()
        guard.check_spray(target(label="dog"), FRAME_SHAPE)

        self.assertEqual(stats.veto_counts, {})
        self.assertEqual(guard.stats().veto_counts, {LABEL_NOT_ALLOWED: 1})

    def test_decision_helpers(self):
        allowed = SafetyDecision.allow("go")
        vetoed = SafetyDecision.veto(NOT_ARMED, "no")

        self.assertEqual((allowed.allowed, allowed.code, allowed.message), (True, OK, "go"))
        self.assertEqual((vetoed.allowed, vetoed.code, vetoed.message), (False, NOT_ARMED, "no"))

    def test_stats_defaults(self):
        stats = SafetyStats()

        self.assertEqual(stats.sprays_total, 0)
        self.assertIsNone(stats.last_spray_end)
        self.assertEqual(stats.veto_counts, {})


ALL_CODES = (
    EMERGENCY_STOP, NOT_ARMED, NO_FRAME, NO_TARGET, LABEL_NOT_ALLOWED, TARGET_NOT_CONFIRMED,
    CONFIDENCE_TOO_LOW, OUTSIDE_SPRAY_ZONE, TARGET_TOO_SMALL, TARGET_TOO_LARGE, SPRAY_IN_PROGRESS,
    BUDGET_TARGET_EXHAUSTED, BUDGET_HOURLY_EXHAUSTED, COOLDOWN_ACTIVE, OUTSIDE_TIME_WINDOW,
)


class VetoDocumentationTest(unittest.TestCase):
    def test_priority_list_covers_every_code_exactly_once(self):
        self.assertEqual(set(VETO_PRIORITY), set(ALL_CODES))
        self.assertEqual(len(VETO_PRIORITY), len(ALL_CODES))

    def test_budget_codes_are_the_budget_vetoes(self):
        self.assertEqual(BUDGET_CODES, {BUDGET_TARGET_EXHAUSTED, BUDGET_HOURLY_EXHAUSTED})

    def test_first_matching_veto_wins_in_documented_order(self):
        observed = []
        wall = FixedWallClock(hour=3)
        clock = FakeClock()
        guard = SafetyGuard(
            SafetyLimits(armed=True, target_labels=("cat",), min_confidence=0.6, confirm_frames=3,
                         max_sprays_per_target=1, max_sprays_per_hour=1, cooldown_s=10.0,
                         allowed_hours=(6, 22)),
            clock=clock, wall_clock=wall,
        )
        guard.register_frame()

        guard.emergency_stop("test")
        observed.append(guard.check_spray(target(), FRAME_SHAPE).code)
        guard.clear_emergency_stop()
        guard.arm()

        guard.disarm()
        observed.append(guard.check_spray(target(), FRAME_SHAPE).code)
        guard.arm()

        clock.advance(2.0)
        observed.append(guard.check_spray(target(), FRAME_SHAPE).code)
        guard.register_frame()

        observed.append(guard.check_spray(None, FRAME_SHAPE).code)
        observed.append(guard.check_spray(target(label="dog"), FRAME_SHAPE).code)
        observed.append(guard.check_spray(SimpleNamespace(label="cat"), FRAME_SHAPE).code)
        observed.append(guard.check_spray(target(confidence=0.1), FRAME_SHAPE).code)
        observed.append(guard.check_spray(target(center=(315, 120)), FRAME_SHAPE).code)
        observed.append(guard.check_spray(target(box=(150, 110, 20, 2)), FRAME_SHAPE).code)
        observed.append(guard.check_spray(target(box=(140, 100, 40, 225)), FRAME_SHAPE).code)

        guard.register_spray_start()
        observed.append(guard.check_spray(target(), FRAME_SHAPE).code)
        guard.register_spray_end()
        observed.append(guard.check_spray(target(), FRAME_SHAPE).code)
        guard.target_left()
        observed.append(guard.check_spray(target(), FRAME_SHAPE).code)
        guard.limits.max_sprays_per_hour = 5
        observed.append(guard.check_spray(target(), FRAME_SHAPE).code)
        guard.limits.cooldown_s = 0.0
        observed.append(guard.check_spray(target(), FRAME_SHAPE).code)

        self.assertEqual(tuple(observed), VETO_PRIORITY)


if __name__ == "__main__":
    unittest.main()