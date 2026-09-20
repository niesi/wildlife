# Wildlife Turret (prototype)

Single app: video input -> motion detection -> one-shot classification ->
tracking -> aiming of a (simulated) pan/tilt head, developed on the PC with
simulated/video frames before the RV1106 camera hardware is available.

## Installation

```bash
pip install opencv-python numpy
```

## Running

```bash
python turret.py                             # reads config.json (defaults if missing)
python turret.py --config config.json        # explicit config file
python turret.py --calibration calibration.json
```

The window shows the video with the tracked target box and a status row
(label, confidence, confirmation count, aim error, head position). Press `q`
to quit.

### config.json (video input and perception)

Hand-edited, never written by the app. Unknown keys are rejected:

```json
{
  "video": {
    "source": "video_file",      // video_file | webcam | synthetic
    "path": "video/youtube.mp4", // required for video_file
    "webcam_index": 0,           // used for webcam
    "sprite": null,              // transparent PNG/JPG for synthetic
    "loop": true,
    "show_motion": true          // tint detected motion pixels red
  },
  "perception": {
    "min_area": 1200,            // smallest accepted motion region (px)
    "max_area": 50000,           // largest accepted motion region (px)
    "min_animal_area": 1500,     // crop area the mock classifier accepts
    "reclassify_every": 60,      // tracker re-classification interval (frames)
    "downscale_width": 320       // motion detection width (0 = full resolution;
                                 // keep small for high-res cameras: RAM of the MOG2
                                 // model scales with pixels, ~66 bytes/px grey)
  }
}
```

All sources deliver grayscale frames; motion detection, classification and
tracking work on unchanged grayscale images, only the display converts to
colour. During the warm-up phase (the first `history` frames of the motion
detector, default 500 ≈ 17 s at 30 FPS) the background model is trained but
no detection results are produced - the status row shows the progress.

The video playback uses the FPS metadata of the file and accounts for
decoding and processing time in the wait. With invalid FPS metadata 30 FPS
are used. If processing is slower than the frame interval, playback runs
slower; frames are never skipped.

### calibration.json (camera and servo data)

Hand-edited, never written by the app: pan/tilt soft limits and centre,
servo speed, camera field of view (h/v) and the aim tolerance. If the file
is missing, the built-in defaults are used. Values:

| Key | Meaning |
| --- | --- |
| `pan_center`, `tilt_center` | neutral position (degrees) |
| `pan_limits`, `tilt_limits` | soft limits the head never leaves |
| `deg_per_s` | maximum servo speed, caps the aim rate |
| `hfov_deg`, `vfov_deg` | horizontal/vertical field of view |
| `aim_tolerance_px` | how close (pixels) counts as "aimed" |
| `settle_frames` | consecutive frames inside tolerance before aiming counts as settled |
| `warmup_frames` | frames to feed a motion model before searching (used by the calibrator) |

## Structure

- `turret.py` – the single entry point: builds everything from the JSON files and runs the frame loop with display.
- `frame_source.py` – interchangeable frame sources (file/webcam/synthetic, later CSI camera via the same interface).
- `motion_detector.py` – MOG2 motion detection, returns bounding boxes.
- `classifier.py` – classification interface. `MockClassifier` and `MockCatClassifier` are placeholders without a real model. `RknnClassifier` is the scaffold for the later NPU classification (RKNN-Toolkit2).
- `tracker.py` – KCF tracking after a confirmed classification, with periodic re-classification.
- `target_detector.py` – motion -> confirmed target (`CatTargetDetector`), one target at a time.
- `aim_controller.py` – pixels -> pan/tilt angles, rate limited, deadband, settle counting.
- `calibration.py` – camera geometry and servo data plus the `Calibrator` procedure (warm-up, limit sweep).
- `hardware.py` – actuator interfaces, simulated devices and the (not yet implemented) serial/GPIO drivers.
- `turret_config.py` – `AppConfig` with `video` and `perception` sections, loaded from `config.json`.
- `config.json` – video input and perception configuration (hand-edited).
- `calibration.json` – default calibration data (hand-edited).
- `tests/` – unit tests, run with `python -m unittest discover` from the project directory.

## Next steps

1. Use real wildlife videos/images as test material (e.g. the LILA-BC dataset)
   instead of only `synthetic`.
2. Train a small classification model and replace the mock classifiers
   (initially PC-side TensorFlow/TFLite).
3. Once a `.rknn` model exists: implement `RknnClassifier`, `turret.py`
   stays unchanged (only the classifier is swapped).
4. Once the CSI camera is available: add `CsiFrameSource` to
   `frame_source.py`; the rest of the app stays unchanged.
5. Re-introduce a safety layer (arming, spray zone, budgets) before any real
   water is connected.
