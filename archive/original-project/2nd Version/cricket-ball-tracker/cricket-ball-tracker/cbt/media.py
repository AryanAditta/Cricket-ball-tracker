"""ffmpeg helpers. Uses a system ffmpeg if present, otherwise the one bundled
with the `imageio-ffmpeg` pip package (works on Windows without installing
ffmpeg separately)."""
import shutil
import subprocess


def ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def to_h264(src, dst, crf=23, all_intra=False, fps=None):
    """Re-encode to browser-playable H.264. all_intra makes every frame a
    keyframe so a player can jump to any frame instantly (frame stepping)."""
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("no ffmpeg found: pip install imageio-ffmpeg")
    cmd = [exe, "-y", "-loglevel", "error", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", str(crf), "-movflags", "+faststart"]
    if all_intra:
        cmd += ["-g", "1"]
    if fps:
        cmd += ["-r", str(fps)]
    subprocess.run(cmd + [dst], check=True)
    return dst


def to_webm(src, dst, crf=36, gop=8):
    """VP9/WebM: plays in Chrome, Edge and Firefox (and in headless Chromium,
    which lacks H.264). A short keyframe interval keeps frame stepping fast."""
    exe = ffmpeg_exe()
    if exe is None:
        raise RuntimeError("no ffmpeg found: pip install imageio-ffmpeg")
    subprocess.run([exe, "-y", "-loglevel", "error", "-i", src, "-c:v", "libvpx-vp9", "-b:v", "0",
                    "-crf", str(crf), "-g", str(gop), "-deadline", "good", "-cpu-used", "4",
                    "-row-mt", "1", "-an", dst], check=True)
    return dst
