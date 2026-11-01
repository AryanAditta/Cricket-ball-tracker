"""Per-frame ball candidates from video.

Two detectors, same output: a list of (frame_idx, t, u, v, score) candidates,
several per frame allowed -- picking the real ball is the tracker's job.

* MotionDetector -- no training needed. Works because OUR cameras are fixed:
  the ball is a small, fast, roughly round blob in a frame difference.
* YoloDetector   -- a trained cricket-ball detector (ultralytics). Fine-tune it
  on frames from our own cameras; public weights trained on broadcast footage
  are a starting point only.
"""
import cv2
import numpy as np


def read_frames(path, start=0, end=None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    i = start
    while end is None or i < end:
        ok, frame = cap.read()
        if not ok:
            break
        # prefer container timestamps; fall back to frame index / fps
        ms = cap.get(cv2.CAP_PROP_POS_MSEC)
        yield i, (ms / 1000.0 if ms > 0 else i / fps), frame
        i += 1
    cap.release()


class MotionDetector:
    """Foreground blobs against a learned background (fixed camera).

    Background subtraction rather than frame differencing, because in the
    end-on views the ball moves only a few pixels per frame (it is coming
    straight at the camera) and frame differences cancel it out.
    """

    def __init__(self, min_area=3, max_area=900, min_circularity=0.4, var_threshold=24,
                 history=120, roi=None, max_per_frame=8, max_aspect=3.5,
                 ball_colour="red", min_redness=0.04):
        self.bg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=var_threshold,
                                                     detectShadows=False)
        self.min_area, self.max_area = min_area, max_area
        self.min_circ, self.max_aspect = min_circularity, max_aspect
        self.roi = roi                      # optional binary mask (h, w)
        self.max_per_frame = max_per_frame
        # colour gate: rejects shadows, whites (players, pads, sight screen)
        # "red" for the red ball; None to disable (e.g. white-ball cricket --
        # use a trained detector there)
        self.ball_colour, self.min_redness = ball_colour, min_redness

    def __call__(self, frame):
        m = self.bg.apply(cv2.GaussianBlur(frame, (3, 3), 0))
        if self.roi is not None:
            m &= self.roi
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bgimg = self.bg.getBackgroundImage() if self.ball_colour else None
        out = []
        for cn in cnts:
            area = cv2.contourArea(cn)
            if not (self.min_area <= area <= self.max_area):
                continue
            per = cv2.arcLength(cn, True) + 1e-6
            circ = 4 * np.pi * area / per ** 2       # motion blur lowers this
            x, y, w, h = cv2.boundingRect(cn)
            if circ < self.min_circ or max(w, h) / max(min(w, h), 1) > self.max_aspect:
                continue
            score = circ
            if self.ball_colour == "red":
                mask = np.zeros((h, w), np.uint8)
                cv2.drawContours(mask, [cn - (x, y)], -1, 255, -1)
                def red(img):
                    b_, g_, r_, _ = cv2.mean(img[y:y + h, x:x + w], mask=mask)
                    return (r_ - max(g_, b_)) / 255.0
                # redder than what is normally there (pitch is slightly red too;
                # shadows and white kit are LESS red than the background)
                redness = red(frame) - red(bgimg)
                if redness < self.min_redness:
                    continue
                score = circ * redness
            M = cv2.moments(cn)
            out.append((M["m10"] / M["m00"], M["m01"] / M["m00"], float(score)))
        out.sort(key=lambda x: -x[2])
        return out[: self.max_per_frame]


class YoloDetector:
    def __init__(self, weights, conf=0.2, imgsz=1280, classes=None):
        from ultralytics import YOLO          # optional dependency
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.classes = conf, imgsz, classes

    def __call__(self, frame):
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                               classes=self.classes, verbose=False)[0]
        out = []
        for (x1, y1, x2, y2), s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            out.append(((x1 + x2) / 2, (y1 + y2) / 2, float(s)))
        return out


def detect_video(path, detector, start=0, end=None):
    """Run a detector over a video. Returns array rows (frame, t, u, v, score)."""
    rows = []
    for i, t, frame in read_frames(path, start, end):
        for u, v, s in detector(frame):
            rows.append((i, t, u, v, s))
    return np.array(rows, float).reshape(-1, 5)
