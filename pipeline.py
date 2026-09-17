"""
Hauptpipeline: Bewegungserkennung -> einmalige Klassifikation bei
neuer Bewegung -> Tracking bei bestätigtem Tier -> periodische
Re-Klassifikation.

Läuft aktuell mit simulierten/Video-Frames (siehe frame_source.py).
Sobald die echte CSI-Kamera verfügbar ist, wird hier nur die Quelle
ausgetauscht (create_source("csi", ...) statt "video_file"/"synthetic"),
der Rest bleibt unverändert.

Start:
    python pipeline.py --source synthetic
    python pipeline.py --source video_file --path mein_clip.mp4
    python pipeline.py --source webcam --index 0
"""

import argparse
import math
import time
import cv2

from frame_source import create_source
from motion_detector import MotionDetector, to_grayscale
from classifier import MockClassifier
from tracker import AnimalTracker


def expand_box(box, frame_shape, margin=20):
    x, y, w, h = box
    fh, fw = frame_shape[:2]
    x0 = max(0, x - margin)
    y0 = max(0, y - margin)
    x1 = min(fw, x + w + margin)
    y1 = min(fh, y + h + margin)
    return x0, y0, x1, y1


def draw_result(frame, box, label, confidence, color):
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = f"{label} {confidence:.2f}" if confidence is not None else label
    cv2.putText(frame, text, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def playback_delay_ms(fps, frame_started):
    """Restliche Framezeit; mindestens 1 ms für die GUI-Ereignisse."""
    if fps is None:
        return 1
    remaining = 1.0 / fps - (time.perf_counter() - frame_started)
    return max(1, math.ceil(remaining * 1000))


def run(args):
    source = create_source(args.source, path=args.path, index=args.index, sprite_path=args.sprite)
    motion = MotionDetector(min_area=args.min_area, max_area=args.max_area)
    classifier = MockClassifier(min_area_for_animal=args.min_animal_area)
    tracker = AnimalTracker(reclassify_interval_frames=args.reclassify_every)

    playback_fps = source.fps if args.source == "video_file" else None
    frame_count = 0
    try:
        while True:
            frame_started = time.perf_counter()
            ok, frame = source.get_frame()
            if not ok:
                print("Kein Frame mehr verfügbar, beende.")
                break
            frame = to_grayscale(frame)
            frame_count += 1
            boxes = motion.detect(frame) if (not tracker.active or args.show_motion_mask) else []
            display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if args.show_motion_mask:
                display_frame = motion.overlay_motion_mask(display_frame)

            if tracker.active:
                ok_track, box = tracker.update(frame)
                if ok_track:
                    draw_result(display_frame, box, tracker.label, None, (0, 200, 255))
                    if tracker.needs_reclassification():
                        x0, y0, x1, y1 = expand_box(box, frame.shape)
                        crop = frame[y0:y1, x0:x1]
                        if crop.size > 0:
                            result = classifier.classify(crop)
                            if result.is_animal:
                                tracker.label = result.label
                                tracker.frames_since_start = 0
                            else:
                                tracker.stop()
                else:
                    tracker.stop()

            if not tracker.active:
                for box in boxes:
                    x0, y0, x1, y1 = expand_box(box, frame.shape)
                    crop = frame[y0:y1, x0:x1]
                    if crop.size == 0:
                        continue
                    result = classifier.classify(crop)
                    color = (0, 220, 0) if result.is_animal else (0, 0, 220)
                    draw_result(display_frame, box, result.label, result.confidence, color)
                    if result.is_animal:
                        tracker.start(frame, box, result.label)
                        break  # nur ein Objekt gleichzeitig verfolgen (Start)

            cv2.imshow("Wildlife Pipeline (Prototyp)", display_frame)
            delay = playback_delay_ms(playback_fps, frame_started)
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Wildlife-Erkennungspipeline (Prototyp, simulierte Bilder)")
    p.add_argument("--source", choices=["video_file", "webcam", "synthetic"], default="synthetic")
    p.add_argument("--path", type=str, default=None, help="Pfad zur Videodatei bei --source video_file")
    p.add_argument("--index", type=int, default=0, help="Webcam-Index bei --source webcam")
    p.add_argument(
        "--sprite",
        type=str,
        default=None,
        help="Transparentes PNG/JPG als bewegtes Tierbild bei --source synthetic",
    )
    p.add_argument("--min-area", type=int, default=1200, dest="min_area")
    p.add_argument("--max-area", type=int, default=50000, dest="max_area")
    p.add_argument("--min-animal-area", type=int, default=1500, dest="min_animal_area")
    p.add_argument("--reclassify-every", type=int, default=60, dest="reclassify_every")
    p.add_argument(
        "--show-motion-mask",
        action="store_true",
        help="Zeigt erkannte MOG2-Bewegungspixel als rote Überlagerung",
    )
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
