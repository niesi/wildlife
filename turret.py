"""
Single entry point: video input -> motion -> classification -> tracking -> aim.

One frame loop, no safety layer for now:

    python turret.py                       # reads config.json (defaults if missing)
    python turret.py --config config.json --calibration calibration.json

The app never writes its JSON files; edit ``config.json`` (video input,
perception thresholds) and ``calibration.json`` (camera geometry, servo
limits) by hand. The pan/tilt head is simulated on the PC.
"""

import argparse
import math
import time

import cv2

from aim_controller import AimController
from calibration import Calibration
from classifier import MockCatClassifier
from frame_source import create_source, to_grayscale
from hardware import SimulatedPanTilt
from motion_detector import MotionDetector
from target_detector import CatTargetDetector
from tracker import AnimalTracker
from turret_config import AppConfig

TARGET_COLOR = (0, 220, 0)
STATUS_COLOR = (255, 200, 120)


def playback_delay_ms(fps, frame_started):
    """Remaining frame time; at least 1 ms so the GUI gets its events."""
    if fps is None:
        return 1
    remaining = 1.0 / fps - (time.perf_counter() - frame_started)
    return max(1, math.ceil(remaining * 1000))


def draw_status(frame, status_text):
    """Status row below the image; the scene itself is not modified."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = max(frame.shape[1], cv2.getTextSize(status_text, font, 0.55, 1)[0][0] + 24)
    display = cv2.copyMakeBorder(frame, 0, 34, 0, width - frame.shape[1],
                                 cv2.BORDER_CONSTANT, value=(20, 20, 20))
    cv2.putText(display, status_text, (12, frame.shape[0] + 24), font, 0.55,
                STATUS_COLOR, 1, cv2.LINE_AA)
    return display


def build(config, calibration):
    """Create source, perception chain, aim controller and simulated head."""
    video = config.video
    source = create_source(video.source, path=video.path, index=video.webcam_index,
                           sprite_path=video.sprite, loop=video.loop)
    motion = MotionDetector(min_area=config.perception.min_area,
                            max_area=config.perception.max_area,
                            downscale_width=config.perception.downscale_width)
    classifier = MockCatClassifier(min_area_for_cat=config.perception.min_animal_area)
    tracker = AnimalTracker(reclassify_interval_frames=config.perception.reclassify_every)
    detector = CatTargetDetector(detector=motion, classifier=classifier, tracker=tracker)
    aimer = AimController.from_calibration(calibration)
    head = SimulatedPanTilt(pan_limits=calibration.pan_limits,
                            tilt_limits=calibration.tilt_limits,
                            deg_per_s=calibration.deg_per_s)
    return source, motion, detector, aimer, head


def run(config, calibration):
    source, motion, detector, aimer, head = build(config, calibration)
    playback_fps = source.fps if config.video.source == "video_file" else None
    last_frame_started = time.perf_counter()
    try:
        while True:
            frame_started = time.perf_counter()
            dt_s = frame_started - last_frame_started
            last_frame_started = frame_started

            ok, frame = source.get_frame()
            if not ok:
                print("No more frames available, exiting.")
                break
            frame = to_grayscale(frame)

            now = time.monotonic()
            target = detector.update(frame, now)

            display = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if config.video.show_motion:
                display = motion.overlay_motion_mask(display)
            status = (f"warming up {motion.frames_seen}/{motion.warmup_frames}"
                      if motion.warming_up else "no target")
            if target is not None:
                aim = aimer.error(target.center, frame.shape)
                command = aimer.command(aim, head.position(), dt_s)
                head.point_at(command.pan_deg, command.tilt_deg)

                x, y, w, h = target.box
                cv2.rectangle(display, (x, y), (x + w, y + h), TARGET_COLOR, 2)
                status = (f"{target.label} {target.confidence:.2f} | confirmed "
                          f"{target.frames_confirmed}f | aim dx={aim.dx_px:.0f} "
                          f"dy={aim.dy_px:.0f}px | head pan={head.position().pan_deg:.1f} "
                          f"tilt={head.position().tilt_deg:.1f}")

            cv2.imshow("Turret", draw_status(display, status))
            if cv2.waitKey(playback_delay_ms(playback_fps, frame_started)) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        head.close()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Turret app: video input -> motion -> classify -> track -> aim")
    p.add_argument("--config", default="config.json", help="JSON config file (never written)")
    p.add_argument("--calibration", default="calibration.json", help="JSON calibration file (never written)")
    return p.parse_args()


def main():
    args = parse_args()
    from pathlib import Path

    config = AppConfig.load(args.config) if Path(args.config).exists() else AppConfig()
    if Path(args.calibration).exists():
        calibration = Calibration.load(args.calibration)
    else:
        print(f"{args.calibration} not found, using default calibration.")
        calibration = Calibration()
    run(config, calibration)


if __name__ == "__main__":
    main()
