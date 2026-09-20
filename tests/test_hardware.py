import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

from hardware import (
    DryRunWaterPump,
    GpioWaterPump,
    PanTiltActuator,
    Position,
    SerialPanTilt,
    SimulatedPanTilt,
    SimulatedWaterPump,
    WaterPump,
    create_hardware,
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


class SimulatedPanTiltTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.head = SimulatedPanTilt(deg_per_s=60.0, clock=self.clock)

    def test_starts_at_the_origin(self):
        self.assertEqual(self.head.position(), Position(0.0, 0.0))
        self.assertTrue(self.head.settled)

    def test_reaches_the_commanded_position_over_time(self):
        self.head.point_at(30.0, 0.0)

        self.clock.advance(0.5)

        self.assertEqual(self.head.position(), Position(30.0, 0.0))
        self.assertTrue(self.head.settled)

    def test_moves_at_the_configured_rate(self):
        self.head.point_at(60.0, 0.0)

        self.clock.advance(0.1)
        self.assertAlmostEqual(self.head.position().pan_deg, 6.0)

        self.clock.advance(0.1)
        self.assertAlmostEqual(self.head.position().pan_deg, 12.0)

    def test_never_overshoots_the_target(self):
        head = SimulatedPanTilt(deg_per_s=1000.0, clock=self.clock)
        head.point_at(10.0, 0.0)

        self.clock.advance(10.0)

        self.assertEqual(head.position(), Position(10.0, 0.0))

    def test_moving_in_both_axes_at_once(self):
        self.head.point_at(-15.0, 9.0)

        self.clock.advance(1.0)

        self.assertEqual(self.head.position(), Position(-15.0, 9.0))

    def test_targets_are_clamped_to_the_limits(self):
        head = SimulatedPanTilt(pan_limits=(-10.0, 10.0), tilt_limits=(-5.0, 5.0), clock=self.clock)

        head.point_at(100.0, -100.0)

        self.assertEqual(head.target, Position(10.0, -5.0))

    def test_start_position_is_clamped(self):
        head = SimulatedPanTilt(pan_limits=(-10.0, 10.0), tilt_limits=(-5.0, 5.0),
                                clock=self.clock, start=(100.0, -100.0))

        self.assertEqual(head.position(), Position(10.0, -5.0))

    def test_snap_jumps_to_the_target(self):
        self.head.point_at(30.0, 10.0)

        self.assertEqual(self.head.snap(), Position(30.0, 10.0))
        self.assertTrue(self.head.settled)

    def test_center_commands_the_origin(self):
        head = SimulatedPanTilt(clock=self.clock, start=(10.0, 5.0))

        head.center()

        self.assertEqual(head.target, Position(0.0, 0.0))

    def test_close_is_idempotent(self):
        self.head.close()
        self.head.close()

        self.assertTrue(self.head.closed)

    def test_invalid_settings_are_rejected(self):
        cases = {
            "unordered pan limits": {"pan_limits": (10.0, -10.0)},
            "incomplete tilt limits": {"tilt_limits": (0.0,)},
            "zero rate": {"deg_per_s": 0.0},
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    SimulatedPanTilt(**overrides)


class SimulatedWaterPumpTest(unittest.TestCase):
    def setUp(self):
        self.clock = FakeClock()
        self.pump = SimulatedWaterPump(clock=self.clock)

    def test_starts_off_without_history(self):
        self.assertFalse(self.pump.is_on)
        self.assertEqual(self.pump.history, [])
        self.assertEqual(self.pump.switches, 0)

    def test_on_and_off_are_recorded_with_timestamps(self):
        self.pump.on()
        self.clock.advance(0.2)
        self.pump.off()

        self.assertEqual(self.pump.history, [(0.0, "on"), (0.2, "off")])
        self.assertFalse(self.pump.is_on)

    def test_repeated_switches_are_ignored(self):
        self.pump.on()
        self.pump.on()
        self.pump.off()
        self.pump.off()

        self.assertEqual(self.pump.switches, 2)

    def test_on_seconds_totals_all_bursts(self):
        self.pump.on()
        self.clock.advance(0.3)
        self.pump.off()
        self.pump.on()
        self.clock.advance(0.2)

        self.assertAlmostEqual(self.pump.on_seconds(), 0.5)

    def test_on_seconds_includes_a_running_burst(self):
        self.pump.on()

        self.clock.advance(1.5)

        self.assertAlmostEqual(self.pump.on_seconds(), 1.5)

    def test_on_seconds_accepts_an_explicit_time(self):
        self.pump.on()

        self.assertAlmostEqual(self.pump.on_seconds(now=2.0), 2.0)

    def test_close_switches_off_and_marks_the_device(self):
        self.pump.on()

        self.pump.close()

        self.assertFalse(self.pump.is_on)
        self.assertTrue(self.pump.closed)

    def test_off_without_a_running_burst_stays_silent(self):
        self.pump.off()

        self.assertEqual(self.pump.history, [])


class DryRunWaterPumpTest(unittest.TestCase):
    def test_never_switches_the_inner_pump_on(self):
        inner = SimulatedWaterPump()
        pump = DryRunWaterPump(inner)

        pump.on()
        pump.on()

        self.assertEqual(pump.requests, 2)
        self.assertFalse(pump.is_on)
        self.assertFalse(pump.inner_is_on)
        self.assertEqual(inner.history, [])

    def test_off_reaches_the_inner_pump(self):
        inner = SimulatedWaterPump()
        inner.on()
        pump = DryRunWaterPump(inner)

        pump.off()

        self.assertFalse(inner.is_on)

    def test_close_closes_both(self):
        inner = SimulatedWaterPump()
        pump = DryRunWaterPump(inner)

        pump.close()

        self.assertTrue(pump.closed)
        self.assertTrue(inner.closed)


class DeviceSkeletonTest(unittest.TestCase):
    def test_serial_head_names_what_is_missing(self):
        with self.assertRaises(NotImplementedError) as context:
            SerialPanTilt("COM3")

        self.assertIn("protocol", str(context.exception))
        self.assertIn("TURRET.md", str(context.exception))

    def test_gpio_pump_names_what_is_missing(self):
        with self.assertRaises(NotImplementedError) as context:
            GpioWaterPump(pin=17)

        self.assertIn("wiring", str(context.exception))
        self.assertIn("TURRET.md", str(context.exception))

    def test_skeletons_keep_the_interfaces(self):
        self.assertTrue(issubclass(SerialPanTilt, PanTiltActuator))
        self.assertTrue(issubclass(GpioWaterPump, WaterPump))


class HardwareFactoryTest(unittest.TestCase):
    def test_simulated_pair_is_built(self):
        actuator, pump = create_hardware("simulated", deg_per_s=30.0)

        self.assertIsInstance(actuator, SimulatedPanTilt)
        self.assertIsInstance(pump, SimulatedWaterPump)
        self.assertEqual(actuator.deg_per_s, 30.0)

    def test_simulated_kwargs_reach_the_actuator(self):
        clock = FakeClock()
        actuator, _ = create_hardware("simulated", clock=clock, deg_per_s=90.0, nonsense=1)

        actuator.point_at(9.0, 0.0)
        clock.advance(0.1)

        self.assertAlmostEqual(actuator.position().pan_deg, 9.0)

    def test_dry_run_wraps_the_pump(self):
        _, pump = create_hardware("simulated", dry_run=True)

        pump.on()

        self.assertIsInstance(pump, DryRunWaterPump)
        self.assertEqual(pump.requests, 1)
        self.assertFalse(pump.inner_is_on)

    def test_device_kind_raises_with_a_hint(self):
        with self.assertRaises(NotImplementedError):
            create_hardware("device", port="COM3", pump_pin=17)

    def test_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            create_hardware("warp")

        self.assertIn("warp", str(context.exception))


if __name__ == "__main__":
    unittest.main()

