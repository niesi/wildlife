import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from aim_controller import pixel_to_angle
from calibration import Calibration, Calibrator, default_calibration, field_of_view_deg
from hardware import Position, SimulatedPanTilt


DUMMY_FRAME = np.zeros((4, 4), dtype=np.uint8)


class CalibrationDefaultsTest(unittest.TestCase):
    def test_defaults_are_consistent(self):
        calibration = Calibration()

        self.assertEqual(calibration.pan_limits, (-90.0, 90.0))
        self.assertEqual(calibration.tilt_limits, (-45.0, 45.0))
        self.assertEqual(calibration.pan_center, 0.0)
        self.assertEqual(calibration.settle_frames, 3)
        self.assertTrue(0.0 < calibration.hfov_deg < 180.0)
        self.assertEqual(calibration.created, "")

    def test_default_factory_stamps_the_creation_time(self):
        calibration = default_calibration(wall_clock=lambda: datetime(2024, 5, 4, 12, 30, 15))

        self.assertEqual(calibration.created, "2024-05-04T12:30:15")

    def test_lists_become_tuples(self):
        calibration = Calibration(pan_limits=[-10, 10], tilt_limits=[-5, 5])

        self.assertEqual(calibration.pan_limits, (-10.0, 10.0))
        self.assertEqual(calibration.tilt_limits, (-5.0, 5.0))

    def test_clamp_uses_the_soft_limits(self):
        calibration = Calibration(pan_limits=(-10.0, 10.0), tilt_limits=(-5.0, 5.0))

        self.assertEqual(calibration.clamp(100.0, -100.0), (10.0, -5.0))
        self.assertEqual(calibration.clamp(3.0, -1.0), (3.0, -1.0))

    def test_invalid_settings_are_rejected(self):
        cases = {
            "unordered pan limits": {"pan_limits": (10.0, -10.0)},
            "incomplete tilt limits": {"tilt_limits": (5.0,)},
            "pan centre outside limits": {"pan_center": 200.0},
            "tilt centre outside limits": {"tilt_limits": (-5.0, 5.0), "tilt_center": 9.0},
            "zero rate": {"deg_per_s": 0.0},
            "zero horizontal fov": {"hfov_deg": 0.0},
            "vertical fov too wide": {"vfov_deg": 200.0},
            "negative tolerance": {"aim_tolerance_px": -1.0},
            "no settle frame": {"settle_frames": 0},
            "negative warmup": {"warmup_frames": -1},
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    Calibration(**overrides)


class FieldOfViewTest(unittest.TestCase):
    def test_is_the_inverse_of_the_pixel_angle(self):
        for fov_deg in (30.0, 60.0, 90.0):
            with self.subTest(fov_deg=fov_deg):
                angle = pixel_to_angle(80.0, 320.0, fov_deg)

                self.assertAlmostEqual(field_of_view_deg(angle, 80.0, 320.0), fov_deg, places=6)

    def test_negative_measurements_are_accepted(self):
        self.assertAlmostEqual(field_of_view_deg(-20.0, -100.0, 320.0),
                               field_of_view_deg(20.0, 100.0, 320.0))

    def test_invalid_values_are_rejected(self):
        cases = {"zero offset": (20.0, 0, 320.0), "no frame": (20.0, 10, 0), "right angle": (90.0, 10, 320.0)}

        for label, args in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    field_of_view_deg(*args)


class CalibrationFileTest(unittest.TestCase):
    def test_dict_round_trip(self):
        calibration = Calibration(pan_center=5.0, tilt_center=-2.0, pan_limits=(-40.0, 40.0),
                                  tilt_limits=(-20.0, 20.0), deg_per_s=45.0, hfov_deg=68.0,
                                  vfov_deg=39.0, aim_tolerance_px=18.0, settle_frames=2,
                                  warmup_frames=15, created="2024-01-01T08:00:00",
                                  notes="garden south")

        self.assertEqual(Calibration.from_dict(calibration.to_dict()), calibration)

    def test_unknown_keys_are_rejected(self):
        for data in ({"pan_degree_center": 0}, {"pan_limits": [0.0, 1.0], "extra": 1}, []):
            with self.subTest(data=data):
                with self.assertRaises(ValueError):
                    Calibration.from_dict(data)

    def test_save_and_load_round_trip(self):
        calibration = default_calibration(wall_clock=lambda: datetime(2024, 6, 1, 10, 0, 0))

        with tempfile.TemporaryDirectory() as folder:
            path = calibration.save(Path(folder) / "calibration.json")
            loaded = Calibration.load(path)
            raw = path.read_text(encoding="utf-8")

        self.assertEqual(loaded, calibration)
        self.assertEqual(json.loads(raw)["pan_limits"], [-90.0, 90.0])

    def test_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            Calibration.load("does-not-exist.json")


class FakeSource:
    """Frame source stand-in that hands out a fixed number of frames."""

    def __init__(self, frames):
        self.frames = list(frames)
        self.pulled = 0

    def get_frame(self):
        if not self.frames:
            return False, None
        self.pulled += 1
        return True, self.frames.pop(0)


class FakeMotion:
    """Motion detector stand-in counting how often it was fed."""

    def __init__(self):
        self.calls = 0

    def detect(self, frame):
        self.calls += 1
        return []


class FakeHead:
    """Head stand-in that jumps to every commanded position."""

    def __init__(self, settle=True):
        self.commands = []
        self.settle = settle

    def point_at(self, pan_deg, tilt_deg):
        self.commands.append((pan_deg, tilt_deg))

    def position(self):
        if not self.commands:
            return Position(0.0, 0.0)
        return Position(*self.commands[-1])

    def wait_until_settled(self, timeout_s=2.0):
        return self.settle


class WarmUpTest(unittest.TestCase):
    def test_frames_are_pulled_into_the_motion_model(self):
        motion = FakeMotion()
        calibrator = Calibrator(FakeHead(), Calibration(warmup_frames=4), motion=motion)

        pulled = calibrator.warm_up(FakeSource([DUMMY_FRAME] * 10))

        self.assertEqual(pulled, 4)
        self.assertEqual(motion.calls, 4)

    def test_stops_when_the_source_runs_out(self):
        motion = FakeMotion()
        calibrator = Calibrator(FakeHead(), Calibration(warmup_frames=10), motion=motion)

        pulled = calibrator.warm_up(FakeSource([DUMMY_FRAME] * 3))

        self.assertEqual(pulled, 3)
        self.assertEqual(motion.calls, 3)

    def test_works_without_a_motion_detector(self):
        calibrator = Calibrator(FakeHead(), Calibration(warmup_frames=2))

        self.assertEqual(calibrator.warm_up(FakeSource([DUMMY_FRAME] * 5)), 2)

    def test_explicit_frame_count_wins(self):
        calibrator = Calibrator(FakeHead(), Calibration(warmup_frames=30))

        self.assertEqual(calibrator.warm_up(FakeSource([DUMMY_FRAME] * 5), frames=2), 2)


class SweepTest(unittest.TestCase):
    def test_visits_centre_and_both_limits(self):
        head = FakeHead()
        calibrator = Calibrator(head, Calibration(pan_limits=(-10.0, 10.0), tilt_limits=(-4.0, 4.0)))

        visited = calibrator.sweep()

        self.assertEqual(visited, [Position(0.0, 0.0), Position(-10.0, 0.0), Position(10.0, 0.0),
                                   Position(0.0, -4.0), Position(0.0, 4.0), Position(0.0, 0.0)])
        self.assertEqual(head.commands[-1], (0.0, 0.0))

    def test_a_head_that_does_not_arrive_fails_the_calibration(self):
        calibrator = Calibrator(FakeHead(settle=False), Calibration())

        with self.assertRaises(RuntimeError) as context:
            calibrator.sweep()

        self.assertIn("sweep", str(context.exception))

    def test_run_returns_a_stamped_calibration(self):
        calibrator = Calibrator(FakeHead(), Calibration(pan_limits=(-30.0, 30.0)),
                                wall_clock=lambda: datetime(2024, 7, 1, 9, 0, 0))

        calibration = calibrator.run(hfov_deg=64.0)

        self.assertEqual(calibration.created, "2024-07-01T09:00:00")
        self.assertEqual(calibration.pan_limits, (-30.0, 30.0))
        self.assertEqual(calibration.hfov_deg, 64.0)

    def test_run_pulls_frames_when_a_source_is_given(self):
        motion = FakeMotion()
        calibrator = Calibrator(FakeHead(), Calibration(warmup_frames=3), motion=motion)

        calibrator.run(source=FakeSource([DUMMY_FRAME] * 3))

        self.assertEqual(motion.calls, 3)

    def test_run_without_a_source_skips_the_warmup(self):
        motion = FakeMotion()
        calibrator = Calibrator(FakeHead(), Calibration(warmup_frames=3), motion=motion)

        calibrator.run()

        self.assertEqual(motion.calls, 0)

    def test_the_simulated_head_completes_a_sweep(self):
        limits = dict(pan_limits=(-20.0, 20.0), tilt_limits=(-10.0, 10.0))
        head = SimulatedPanTilt(deg_per_s=200000.0, **limits)
        calibrator = Calibrator(head, Calibration(**limits))

        visited = calibrator.sweep()

        self.assertEqual(visited[0], Position(0.0, 0.0))
        self.assertEqual(visited[1], Position(-20.0, 0.0))
        self.assertEqual(visited[-1], Position(0.0, 0.0))


if __name__ == "__main__":
    unittest.main()

