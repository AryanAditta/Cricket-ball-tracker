"""Suggest a broadcast speed-gun reading from a user-selected graphic."""
from collections import Counter
import re
import shutil
import subprocess

import cv2
import numpy as np

from .broadcast import read_frames, video_info


def suggest_speed(video, shot, roi):
    exe = shutil.which("tesseract")
    if not exe:
        raise RuntimeError("Speed OCR requires Tesseract on PATH. You can enter --speed-kmh manually.")
    x, y, w, h = roi
    width, height = video_info(video)["size"]
    if min(x, y) < 0 or min(w, h) < 4 or x + w > width or y + h > height:
        raise ValueError("speed OCR region is outside the original video")
    indices = np.linspace(*shot, num=min(12, shot[1] - shot[0] + 1), dtype=int)
    readings = Counter()
    for _, frame in read_frames(video, indices):
        crop = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
        crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, image = cv2.imencode(".png", crop)
        try:
            result = subprocess.run(
                [exe, "stdin", "stdout", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789."],
                input=image.tobytes(), capture_output=True, timeout=10, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError("Speed OCR could not read the graphic. You can enter --speed-kmh manually.") from exc
        match = re.fullmatch(r"(\d{2,3}(?:\.\d)?)", result.stdout.decode().strip())
        if match and 40 <= float(match[1]) <= 175:
            readings[float(match[1])] += 1
    top = readings.most_common(2)
    if top and top[0][1] >= 2 and (len(top) == 1 or top[0][1] > top[1][1]):
        return top[0][0]
    return None
