import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
import unittest
from pathlib import Path

from turret_config import SafetyLimits, StateTimeouts, TurretConfig


class StateTimeoutsTest(unittest.TestCase):
    def test_defaults_are_positive(self):
        timeouts = StateTimeouts()

        self.assertEqual(
            (timeouts.aim_s, timeouts.spray_s, timeouts.verify_s, timeouts.target_confirm_s),
            (2.0, 0.3, 5.0, 1.0),
        )

    def test_zero_or_negative_duration_is_rejected(self):
        for name in ("aim_s", "spray_s", "verify_s", "target_confirm_s"):
            for value in (0.0, -1.0):
                with self.subTest(field=name, value=value):
                    with self.assertRaises(ValueError) as context:
                        StateTimeouts(**{name: value})
                    self.assertIn(name, str(context.exception))


class SafetyLimitsTest(unittest.TestCase):
    def test_safe_defaults(self):
        limits = SafetyLimits()

        self.assertFalse(limits.armed)
        self.assertEqual(limits.target_labels, ("cat",))
        self.assertEqual(limits.spray_zone, (0.05, 0.05, 0.95, 0.95))
        self.assertEqual(limits.allowed_hours, (6, 22))
        self.assertLess(limits.min_target_height, limits.max_target_height)

    def test_no_cooldown_is_allowed(self):
        self.assertEqual(SafetyLimits(cooldown_s=0.0).cooldown_s, 0.0)

    def test_sequence_fields_become_tuples(self):
        limits = SafetyLimits(target_labels=["cat", "dog"],
                              spray_zone=[0.1, 0.2, 0.8, 0.9],
                              allowed_hours=[8, 20])

        self.assertEqual(limits.target_labels, ("cat", "dog"))
        self.assertEqual(limits.spray_zone, (0.1, 0.2, 0.8, 0.9))
        self.assertEqual(limits.allowed_hours, (8, 20))

    def test_sprinkling_around_the_clock_is_allowed(self):
        self.assertIsNone(SafetyLimits(allowed_hours=None).allowed_hours)

    def test_invalid_values_are_rejected(self):
        cases = {
            "empty labels": {"target_labels": ()},
            "confidence above one": {"min_confidence": 1.2},
            "negative confidence": {"min_confidence": -0.1},
            "zero confirm frames": {"confirm_frames": 0},
            "short spray zone": {"spray_zone": (0.1, 0.1, 0.9)},
            "reversed spray zone": {"spray_zone": (0.9, 0.1, 0.1, 0.9)},
            "spray zone above one": {"spray_zone": (0.1, 0.1, 1.5, 0.9)},
            "negative spray zone": {"spray_zone": (-0.1, 0.1, 0.9, 0.9)},
            "reversed heights": {"min_target_height": 0.5, "max_target_height": 0.2},
            "height above one": {"max_target_height": 1.5},
            "negative cooldown": {"cooldown_s": -1.0},
            "zero max spray": {"max_spray_s": 0.0},
            "no per target budget": {"max_sprays_per_target": 0},
            "no hourly budget": {"max_sprays_per_hour": 0},
            "inverted hours": {"allowed_hours": (22, 6)},
            "hours beyond the day": {"allowed_hours": (6, 25)},
            "zero frame timeout": {"frame_timeout_s": 0.0},
        }
        for label, overrides in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    SafetyLimits(**overrides)


class TurretConfigTest(unittest.TestCase):
    def test_defaults_are_consistent(self):
        config = TurretConfig()

        self.assertEqual(config.state, StateTimeouts())
        self.assertEqual(config.safety, SafetyLimits())
        self.assertLessEqual(config.state.spray_s, config.safety.max_spray_s)

    def test_nominal_spray_must_fit_the_safety_cap(self):
        with self.assertRaises(ValueError) as context:
            TurretConfig(state=StateTimeouts(spray_s=0.9))

        self.assertIn("max_spray_s", str(context.exception))

    def test_dict_round_trip_keeps_every_value(self):
        config = TurretConfig(state=StateTimeouts(aim_s=1.5, spray_s=0.2, verify_s=2.0, target_confirm_s=0.5),
                              safety=SafetyLimits(armed=True, target_labels=("cat", "kitten"),
                                                  min_confidence=0.75, confirm_frames=2,
                                                  spray_zone=(0.2, 0.2, 0.8, 0.8),
                                                  min_target_height=0.05, max_target_height=0.7,
                                                  cooldown_s=1.0, max_spray_s=0.25,
                                                  max_sprays_per_target=1, max_sprays_per_hour=5,
                                                  allowed_hours=(7, 21), frame_timeout_s=0.5))

        self.assertEqual(TurretConfig.from_dict(config.to_dict()), config)

    def test_json_arrays_become_tuples(self):
        config = TurretConfig.from_dict({
            "safety": {"target_labels": ["cat"], "spray_zone": [0.1, 0.1, 0.9, 0.9], "allowed_hours": [8, 20]},
        })

        self.assertEqual(config.safety.target_labels, ("cat",))
        self.assertEqual(config.safety.spray_zone, (0.1, 0.1, 0.9, 0.9))
        self.assertEqual(config.safety.allowed_hours, (8, 20))

    def test_partial_section_keeps_the_other_defaults(self):
        config = TurretConfig.from_dict({"safety": {"cooldown_s": 1.5}})

        self.assertEqual(config.safety.cooldown_s, 1.5)
        self.assertEqual(config.state, StateTimeouts())
        self.assertFalse(config.safety.armed)

    def test_empty_dict_yields_defaults(self):
        self.assertEqual(TurretConfig.from_dict({}), TurretConfig())

    def test_allowed_hours_none_survives_the_round_trip(self):
        config = TurretConfig(safety=SafetyLimits(allowed_hours=None))

        self.assertIsNone(TurretConfig.from_dict(config.to_dict()).safety.allowed_hours)

    def test_unknown_keys_are_rejected(self):
        cases = {
            "top level": {"turret": {}},
            "state section": {"state": {"aim_ms": 2000}},
            "safety section": {"safety": {"spray_seconds": 1}},
        }
        for label, data in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(ValueError):
                    TurretConfig.from_dict(data)

    def test_wrong_section_type_is_rejected(self):
        with self.assertRaises(ValueError) as context:
            TurretConfig.from_dict({"safety": 5})

        self.assertIn("safety", str(context.exception))

    def test_non_object_config_is_rejected(self):
        with self.assertRaises(ValueError):
            TurretConfig.from_dict(["state"])


class ConfigFileTest(unittest.TestCase):
    def test_save_and_load_round_trip(self):
        config = TurretConfig(safety=SafetyLimits(armed=True, cooldown_s=2.5,
                                                  target_labels=("cat",),
                                                  allowed_hours=(6, 22)))

        with tempfile.TemporaryDirectory() as folder:
            path = config.save(Path(folder) / "turret.json")
            self.assertTrue(path.exists())
            loaded = TurretConfig.load(path)

        self.assertEqual(loaded, config)

    def test_file_is_utf8_json_with_readable_keys(self):
        config = TurretConfig()

        with tempfile.TemporaryDirectory() as folder:
            path = config.save(Path(folder) / "turret.json")
            raw = path.read_text(encoding="utf-8")

        self.assertIn("max_spray_s", raw)
        self.assertIn("max_spray_s", json.loads(raw)["safety"])

    def test_broken_json_raises_decode_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.json"
            path.write_text("{not json", encoding="utf-8")

            with self.assertRaises(json.JSONDecodeError):
                TurretConfig.load(path)


if __name__ == "__main__":
    unittest.main()