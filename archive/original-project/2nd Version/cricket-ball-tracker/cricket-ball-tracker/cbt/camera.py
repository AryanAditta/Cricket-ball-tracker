"""Pinhole camera model, pose calibration from pitch landmarks, JSON I/O."""
import json
import numpy as np
import cv2


class Camera:
    """x_img ~ K [R | t] X_world, with optional OpenCV lens distortion."""

    def __init__(self, K, R, t, size=(1920, 1080), dist=None, name="cam", fps=120.0,
                 time_offset=0.0):
        self.K = np.asarray(K, float)
        self.R = np.asarray(R, float)
        self.t = np.asarray(t, float).reshape(3)
        self.size = tuple(size)
        self.dist = np.zeros(5) if dist is None else np.asarray(dist, float)
        self.name = name
        self.fps = fps
        self.time_offset = time_offset  # camera clock minus master clock (s)

    # ---------- construction ----------
    @classmethod
    def look_at(cls, position, target, field_width, size=(1920, 1080), **kw):
        """Place a camera at `position` aimed at `target`, zoomed so that
        `field_width` metres span the image width at the target distance."""
        position, target = np.asarray(position, float), np.asarray(target, float)
        fwd = target - position
        dist = np.linalg.norm(fwd)
        fwd /= dist
        right = np.cross(fwd, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        R = np.stack([right, down, fwd])          # rows: cam x, y, z in world
        f = (size[0] / 2) / ((field_width / 2) / dist)
        K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
        return cls(K, R, -R @ position, size=size, **kw)

    @property
    def center(self):
        return -self.R.T @ self.t

    # ---------- geometry ----------
    def project(self, X):
        """World points (N,3) -> pixels (N,2). Points behind camera -> nan."""
        X = np.atleast_2d(X)
        Xc = X @ self.R.T + self.t
        if np.any(self.dist):
            rvec, _ = cv2.Rodrigues(self.R)
            uv, _ = cv2.projectPoints(X.astype(np.float64), rvec, self.t, self.K, self.dist)
            uv = uv.reshape(-1, 2)
        else:
            p = Xc @ self.K.T
            uv = p[:, :2] / p[:, 2:3]
        uv[Xc[:, 2] <= 0] = np.nan
        return uv

    def in_view(self, uv, margin=0):
        w, h = self.size
        return (uv[:, 0] >= margin) & (uv[:, 0] < w - margin) & \
               (uv[:, 1] >= margin) & (uv[:, 1] < h - margin)

    def undistort(self, uv):
        if not np.any(self.dist):
            return np.asarray(uv, float)
        pts = np.asarray(uv, float).reshape(-1, 1, 2)
        return cv2.undistortPoints(pts, self.K, self.dist, P=self.K).reshape(-1, 2)

    def ray(self, uv):
        """Back-project pixel(s) to world rays: returns (origin, unit directions)."""
        uv = self.undistort(np.atleast_2d(uv))
        h = np.c_[uv, np.ones(len(uv))]
        d = (np.linalg.inv(self.K) @ h.T).T @ self.R   # = R^T K^-1 x
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        return self.center, d

    def ray_plane(self, uv, z=0.0):
        """Intersect pixel ray(s) with the horizontal plane at height z."""
        o, d = self.ray(uv)
        s = (z - o[2]) / d[:, 2]
        return o + s[:, None] * d

    # ---------- I/O ----------
    def to_dict(self):
        return dict(name=self.name, K=self.K.tolist(), R=self.R.tolist(), t=self.t.tolist(),
                    size=list(self.size), dist=self.dist.tolist(), fps=self.fps,
                    time_offset=self.time_offset)

    @classmethod
    def from_dict(cls, d):
        return cls(d["K"], d["R"], d["t"], size=d["size"], dist=d.get("dist"),
                   name=d.get("name", "cam"), fps=d.get("fps", 120.0),
                   time_offset=d.get("time_offset", 0.0))


def save_cameras(cams, path):
    with open(path, "w") as f:
        json.dump([c.to_dict() for c in cams], f, indent=2)


def load_cameras(path):
    with open(path) as f:
        return [Camera.from_dict(d) for d in json.load(f)]


def calibrate_pose(K, image_points, world_points, dist=None, **kw):
    """Camera pose from >= 6 clicked pitch landmarks (see pitch.landmarks()).

    Uses the stump tops as out-of-plane points; returns (Camera, rms_px).
    """
    obj = np.asarray(world_points, np.float64).reshape(-1, 3)
    img = np.asarray(image_points, np.float64).reshape(-1, 2)
    d = np.zeros(5) if dist is None else np.asarray(dist, float)
    ok, rvec, tvec = cv2.solvePnP(obj, img, np.asarray(K, float), d,
                                  flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, np.asarray(K, float), d, rvec, tvec)
    R, _ = cv2.Rodrigues(rvec)
    cam = Camera(K, R, tvec.ravel(), dist=d, **kw)
    rms = float(np.sqrt(np.nanmean(np.sum((cam.project(obj) - img) ** 2, axis=1))))
    return cam, rms


def calibrate_intrinsics(image_files, board=(9, 6), square=0.025):
    """Lens intrinsics from checkerboard photos (do once per camera+zoom)."""
    objp = np.zeros((board[0] * board[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board[0], 0:board[1]].T.reshape(-1, 2) * square
    objs, imgs, size = [], [], None
    for fn in image_files:
        gray = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, board)
        if ok:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                       (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
            objs.append(objp)
            imgs.append(corners)
    if len(objs) < 8:
        raise RuntimeError(f"only {len(objs)} usable checkerboard views; need >= 8")
    rms, K, dist, _, _ = cv2.calibrateCamera(objs, imgs, size, None, None)
    return K, dist.ravel(), rms, size
