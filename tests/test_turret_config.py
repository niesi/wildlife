import json
import tempfile
import unittest
from pathlib import Path

from turret_config import AppConfig, PerceptionConfig, VideoConfig, reject_unknown_keys


class VideoConfigTest(unittest.TestCase):
    def test_safe_defaults(self):
        video = VideoConfig()
        self.assertEqual(video.source, "synthetic")
        self.assertIsNone(video.path)
        self.assertEqual(video.webcam_index, 0)
        self.assertIsNone(video.sprite)
        self.assertTrue(video.loop)

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            VideoConfig(source="rtsp")

    def test_video_file_requires_a_path(self):
        with self.assertRaises(ValueError):
            VideoConfig(source="video_file", path=None)
        with self.assertRaises(ValueError):
            VideoConfig(source="video_file", path="")


class PerceptionConfigTest(unittest.TestCase):
    def test_safe_defaults(self):
        perception = PerceptionConfig()
        self.assertEqual(perception.min_area, 1200)
        self.assertEqual(perception.max_area, 50000)
        self.assertEqual(perception.min_animal_area, 1500)
        self.assertEqual(perception.reclassify_every, 60)
        self.assertEqual(perception.downscale_width, 320)

    def test_areas_must_be_positive(self):
        for name in ("min_area", "max_area", "min_animal_area"):
            with self.assertRaises(ValueError):
                PerceptionConfig(**{name: 0})

    def test_min_area_must_not_exceed_max_area(self):
        with self.assertRaises(ValueError):
            PerceptionConfig(min_area=600, max_area=500)

    def test_reclassify_interval_must_be_positive(self):
        with self.assertRaises(ValueError):
            PerceptionConfig(reclassify_every=0)

    def test_downscale_width_must_not_be_negative(self):
        with self.assertRaises(ValueError):
            PerceptionConfig(downscale_width=-1)
        PerceptionConfig(downscale_width=0)  # 0 means full resolution


class AppConfigTest(unittest.TestCase):
    def test_defaults_are_consistent(self):
        config = AppConfig()
        self.assertEqual(config.to_dict(), {
            "video": {"source": "synthetic", "path": None, "webcam_index": 0,
                      "sprite": None, "loop": True, "show_motion": True},
            "perception": {"min_area": 1200, "max_area": 50000,
                           "min_animal_area": 1500, "reclassify_every": 60,
                           "downscale_width": 320},
        })

    def test_empty_dict_yields_defaults(self):
        config = AppConfig.from_dict({})
        self.assertIsInstance(config.video, VideoConfig)
        self.assertIsInstance(config.perception, PerceptionConfig)

    def test_unknown_top_level_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            AppConfig.from_dict({"safety": {}})

    def test_unknown_section_keys_are_rejected(self):
        with self.assertRaises(ValueError):
            AppConfig.from_dict({"video": {"fps": 30}})
        with self.assertRaises(ValueError):
            AppConfig.from_dict({"perception": {"min_confidence": 0.5}})

    def test_sections_must_be_objects(self):
        with self.assertRaises(ValueError):
            AppConfig.from_dict({"video": "webcam"})

    def test_config_must_be_an_object(self):
        with self.assertRaises(ValueError):
            AppConfig.from_dict([])

    def test_nested_validation_applies(self):
        with self.assertRaises(ValueError):
            AppConfig.from_dict({"video": {"source": "video_file"}})  # missing path


class ConfigFileTest(unittest.TestCase):
    def test_round_trip_from_file(self):
        config = AppConfig.from_dict({
            "video": {"source": "video_file", "path": "video/youtube.mp4", "loop": False},
            "perception": {"min_area": 800},
        })
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")
            loaded = AppConfig.load(path)
        self.assertEqual(loaded.to_dict(), config.to_dict())

    def test_broken_json_raises_decode_error(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "broken.json"
            path.write_text("{not json", encoding="utf-8")
            with self.assertRaises(json.JSONDecodeError):
                AppConfig.load(path)


class RejectUnknownKeysTest(unittest.TestCase):
    def test_rejects_unknown_keys_with_names(self):
        with self.assertRaises(ValueError) as ctx:
            reject_unknown_keys("video", {"path": "x", "resolution": "high"}, VideoConfig)
        self.assertIn("resolution", str(ctx.exception))

    def test_rejects_non_object_sections(self):
        with self.assertRaises(ValueError):
            reject_unknown_keys("video", ["x"], VideoConfig)


if __name__ == "__main__":
    unittest.main()
