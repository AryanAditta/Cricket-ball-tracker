"""Register small pans and zooms to a fixed view before broadcast calibration."""
import os

import cv2
import numpy as np

from .broadcast import read_frames, video_info


def _registration(reference, frame, points):
    moved, found, _ = cv2.calcOpticalFlowPyrLK(reference, frame, points, None,
                                                winSize=(31, 31), maxLevel=4)
    if moved is None:
        raise RuntimeError("too few static features to stabilize the shot")
    back, returned, _ = cv2.calcOpticalFlowPyrLK(frame, reference, moved, None,
                                                  winSize=(31, 31), maxLevel=4)
    if back is None:
        raise RuntimeError("too few static features to stabilize the shot")
    keep = found.ravel().astype(bool) & returned.ravel().astype(bool)
    keep &= np.linalg.norm((back - points).reshape(-1, 2), axis=1) < 1.5
    src, dst = moved[keep].reshape(-1, 2), points[keep].reshape(-1, 2)
    if len(src) < 16:
        raise RuntimeError("too few static features to stabilize the shot")
    warp, inliers = cv2.estimateAffinePartial2D(src, dst, ransacReprojThreshold=1.5)
    if warp is None or inliers.sum() < max(16, 0.4 * len(src)):
        raise RuntimeError("camera motion could not be estimated reliably")
    height, width = reference.shape
    if np.any(np.ptp(dst[inliers.ravel().astype(bool)], axis=0) < [0.2 * width, 0.2 * height]):
        raise RuntimeError("static features do not cover enough of the pitch view")
    scale = np.hypot(warp[0, 0], warp[1, 0])
    centre = np.array([width / 2, height / 2])
    shift = warp[:, :2] @ centre + warp[:, 2] - centre
    if not 0.88 <= scale <= 1.12 or np.linalg.norm(shift) > 0.10 * width:
        raise RuntimeError("camera moved too far for small-motion stabilization")
    return warp


def stabilize_shot(video, shot, output):
    first, last = shot
    mid = (first + last) // 2
    fps = video_info(video)["fps"]
    ref = next(read_frames(video, [mid]))[1]
    height, width = ref.shape[:2]
    down = min(1.0, 960 / width)

    def gray(frame):
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(image, None, fx=down, fy=down, interpolation=cv2.INTER_AREA)

    reference = gray(ref)
    points = cv2.goodFeaturesToTrack(reference, 700, 0.012, 6)
    if points is None or len(points) < 16:
        raise RuntimeError("too few reference features to stabilize the shot")
    tmp = os.fspath(output) + ".part.mp4"
    writer = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    try:
        if not writer.isOpened():
            raise RuntimeError("could not write stabilized footage")
        for index, frame in read_frames(video, range(first, last + 1)):
            try:
                warp = _registration(reference, gray(frame), points)
            except RuntimeError as exc:
                raise RuntimeError(f"Frame {index}: {exc}. Select a shorter, continuous replay shot.") from exc
            warp[:, 2] /= down
            writer.write(cv2.warpAffine(frame, warp, (width, height), borderMode=cv2.BORDER_REPLICATE))
        writer.release()
        os.replace(tmp, output)
    finally:
        writer.release()
        if os.path.exists(tmp):
            os.remove(tmp)
