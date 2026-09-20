import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math
import unittest
from types import SimpleNamespace

from aim_controller import AimController, AimError, AimResult, PanTilt, pixel_to_angle


FRAME = (240, 320)  # height, width


class PixelToAngleTest(unittest.TestCase):
    def test_centre_pixel_means_zero_angle(self):
        self.assertEqual(pixel_to_angle(0, 320, 60.0), 0.0)

    def test_half_frame_offset_means_half_the_field_of_view(self):
        self.assertAlmostEqual(pixel_to_angle(160, 320, 60.0), 30.0)
        self.assertAlmostEqual(pixel_to_angle(-160, 320, 60.0), -30.0)

    def test_quarter_offset_is_between_a_quarter_and_a_half_of_the_fov(self):
        angle = pixel_to_angle(80, 320, 60.0)

        self.assertGreater(angle, 15.0)
        self.assertLess(angle, 30.0)

    def test_vertical_axis_uses_its_own_frame_size(self):
        self.assertAlmostEqual(pixel_to_angle(120, 240, 36.0), 18.0)

    def test_invalid_arguments_are_rejected(self):
        cases = {"no frame size": (10, 0, 60.0), "zero fov": (10, 320, 0.0), "wide fov": (10, 320, 180.0)}

        for label, args in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    pixel_to_angle(*args)


class AimErrorTest(unittest.TestCase):
    def setUp(self):
        self.controller = AimController()

    def test_centred_target_needs_no_movement(self):
        error = self.controller.error((160, 120), FRAME)

        self.assertEqual((error.dx_px, error.dy_px), (0.0, 0.0))
        self.assertEqual((error.pan_deg, error.tilt_deg), (0.0, 0.0))
        self.assertEqual(error.distance_px, 0.0)
        self.assertTrue(error.aimed)

    def test_target_to_the_right_needs_positive_pan(self):
        error = self.controller.error((240, 120), FRAME)

        self.assertEqual(error.dx_px, 80.0)
        self.assertGreater(error.pan_deg, 0.0)
        self.assertEqual(error.tilt_deg, 0.0)
        self.assertFalse(error.aimed)

    def test_target_to_the_left_needs_negative_pan(self):
        self.assertLess(self.controller.error((80, 120), FRAME).pan_deg, 0.0)

    def test_target_below_the_centre_needs_negative_tilt(self):
        error = self.controller.error((160, 180), FRAME)

        self.assertEqual(error.dy_px, 60.0)
        self.assertLess(error.tilt_deg, 0.0)
        self.assertEqual(error.pan_deg, 0.0)

    def test_target_above_the_centre_needs_positive_tilt(self):
        self.assertGreater(self.controller.error((160, 60), FRAME).tilt_deg, 0.0)

    def test_small_offsets_are_already_aimed(self):
        error = self.controller.error((170, 125), FRAME)

        self.assertLess(error.distance_px, 24.0)
        self.assertTrue(error.aimed)

    def test_distance_is_the_pixel_hypotenuse(self):
        error = self.controller.error((220, 160), FRAME)

        self.assertAlmostEqual(error.distance_px, math.hypot(60.0, 40.0))

    def test_mounting_can_be_inverted(self):
        upright = self.controller.error((240, 180), FRAME)
        flipped = AimController(invert_pan=True, invert_tilt=True).error((240, 180), FRAME)

        self.assertAlmostEqual(flipped.pan_deg, -upright.pan_deg)
        self.assertAlmostEqual(flipped.tilt_deg, -upright.tilt_deg)
        self.assertEqual(flipped.distance_px, upright.distance_px)

    def test_bad_frame_shape_is_rejected(self):
        for shape in (None, (), (0, 0)):
            with self.subTest(shape=shape):
                with self.assertRaises(ValueError):
                    self.controller.error((10, 10), shape)


class CommandTest(unittest.TestCase):
    def test_step_is_limited_by_the_maximum_rate(self):
        controller = AimController(gain=1.0, max_rate_deg_s=60.0)
        error = controller.error((320, 120), FRAME)
        self.assertAlmostEqual(error.pan_deg, 30.0)

        command = controller.command(error, PanTilt(0.0, 0.0), dt_s=0.1)

        self.assertAlmostEqual(command.pan_deg, 6.0)
        self.assertEqual(command.tilt_deg, 0.0)

    def test_gain_scales_a_small_step(self):
        controller = AimController(gain=0.5, max_rate_deg_s=1000.0)
        error = controller.error((240, 120), FRAME)

        command = controller.command(error, PanTilt(0.0, 0.0), dt_s=1.0)

        self.assertAlmostEqual(command.pan_deg, error.pan_deg * 0.5)

    def test_command_stays_inside_the_limits(self):
        controller = AimController(gain=10.0, max_rate_deg_s=1000.0,
                                   pan_limits=(-5.0, 5.0), tilt_limits=(-2.0, 2.0))
        error = controller.error((320, 240), FRAME)

        command = controller.command(error, PanTilt(0.0, 0.0), dt_s=1.0)

        self.assertEqual(command.pan_deg, 5.0)
        self.assertEqual(command.tilt_deg, -2.0)

    def test_zero_dt_does_not_move(self):
        controller = AimController()
        error = controller.error((320, 120), FRAME)

        self.assertEqual(controller.command(error, PanTilt(3.0, -4.0), dt_s=0.0), PanTilt(3.0, -4.0))

    def test_the_rate_limit_applies_to_every_frame(self):
        controller = AimController(gain=1.0, max_rate_deg_s=60.0)
        error = controller.error((320, 120), FRAME)
        position = PanTilt(0.0, 0.0)

        for expected in (2.0, 4.0, 6.0):
            position = controller.command(error, position, dt_s=1 / 30)
            self.assertAlmostEqual(position.pan_deg, expected)

    def test_the_same_error_reaches_the_angle_in_one_step_without_a_limit(self):
        controller = AimController(gain=1.0, max_rate_deg_s=1000.0)
        error = controller.error((320, 120), FRAME)

        command = controller.command(error, PanTilt(0.0, 0.0), dt_s=1.0)

        self.assertAlmostEqual(command.pan_deg, error.pan_deg)


class SettleTest(unittest.TestCase):
    def test_settled_after_the_configured_number_of_frames(self):
        controller = AimController(settle_frames=3)

        results = [controller.update((160, 120), FRAME, PanTilt(0.0, 0.0), 1 / 30) for _ in range(3)]

        self.assertEqual([result.aimed_frames for result in results], [1, 2, 3])
        self.assertEqual([result.settled for result in results], [False, False, True])

    def test_a_single_settle_frame_is_possible(self):
        controller = AimController(settle_frames=1)

        self.assertTrue(controller.update((160, 120), FRAME, PanTilt(0.0, 0.0), 1 / 30).settled)

    def test_missing_the_tolerance_resets_the_counter(self):
        controller = AimController(settle_frames=2)
        controller.update((160, 120), FRAME, PanTilt(0.0, 0.0), 1 / 30)

        controller.update((300, 120), FRAME, PanTilt(0.0, 0.0), 1 / 30)
        result = controller.update((160, 120), FRAME, PanTilt(0.0, 0.0), 1 / 30)

        self.assertEqual(result.aimed_frames, 1)
        self.assertFalse(result.settled)

    def test_reset_clears_the_progress(self):
        controller = AimController(settle_frames=2)
        controller.update((160, 120), FRAME, PanTilt(0.0, 0.0), 1 / 30)

        controller.reset()

        self.assertEqual(controller.aimed_frames, 0)
        self.assertFalse(controller.update((160, 120), FRAME, PanTilt(0.0, 0.0), 1 / 30).settled)

    def test_update_returns_the_command_for_the_current_frame(self):
        controller = AimController(settle_frames=1)

        result = controller.update((240, 120), FRAME, PanTilt(0.0, 0.0), 0.05)

        self.assertIsInstance(result, AimResult)
        self.assertGreater(result.command.pan_deg, 0.0)
        self.assertIsInstance(result.error, AimError)


class SettingsTest(unittest.TestCase):
    def test_invalid_settings_are_rejected(self):
        cases = {
            "unordered pan limits": {"pan_limits": (10.0, -10.0)},
            "incomplete tilt limits": {"tilt_limits": (0.0,)},
            "zero horizontal fov": {"hfov_deg": 0.0},
            "vertical fov too wide": {"vfov_deg": 200.0},
            "zero gain": {"gain": 0.0},
            "negative rate": {"max_rate_deg_s": -1.0},
            "negative tolerance": {"tolerance_px": -1.0},
            "no settle frame": {"settle_frames": 0},
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    AimController(**overrides)

    def test_controller_can_be_built_from_a_calibration(self):
        calibration = SimpleNamespace(pan_limits=(-30.0, 30.0), tilt_limits=(-15.0, 15.0),
                                      hfov_deg=70.0, vfov_deg=40.0, aim_tolerance_px=10.0,
                                      settle_frames=2, deg_per_s=45.0)

        controller = AimController.from_calibration(calibration)

        self.assertEqual(controller.pan_limits, (-30.0, 30.0))
        self.assertEqual(controller.tilt_limits, (-15.0, 15.0))
        self.assertEqual(controller.hfov_deg, 70.0)
        self.assertEqual(controller.vfov_deg, 40.0)
        self.assertEqual(controller.tolerance_px, 10.0)
        self.assertEqual(controller.settle_frames, 2)
        self.assertEqual(controller.max_rate_deg_s, 45.0)

    def test_overrides_win_over_the_calibration(self):
        calibration = SimpleNamespace(pan_limits=(-30.0, 30.0), tilt_limits=(-15.0, 15.0),
                                      hfov_deg=70.0, vfov_deg=40.0, aim_tolerance_px=10.0,
                                      settle_frames=2, deg_per_s=45.0)

        controller = AimController.from_calibration(calibration, tolerance_px=99.0, invert_pan=True)

        self.assertEqual(controller.tolerance_px, 99.0)
        self.assertTrue(controller.invert_pan)


if __name__ == "__main__":
    unittest.main()

