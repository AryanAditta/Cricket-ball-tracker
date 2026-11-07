"""Windows clip picker for the analysis command."""
import datetime
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, ttk


def main():
    root = tk.Tk()
    root.title("Cricket Ball Tracker")
    root.minsize(630, 420)
    panel = ttk.Frame(root, padding=18)
    panel.pack(fill="both", expand=True)
    panel.columnconfigure(1, weight=1)
    panel.rowconfigure(8, weight=1)
    events = Queue()
    output = None
    clip = tk.StringVar()
    handed = tk.StringVar(value="right")
    call = tk.StringVar(value="not-out")
    ball = tk.StringVar(value="red")
    exports = tk.StringVar(value="Review only")
    modes = {
        "Review only": ["--review-only"],
        "Video with tracking overlay": ["--no-angles"],
        "Video with 3D angles": ["--angle-height", "480"],
    }
    status = tk.StringVar(value="Choose a replay to get started.")

    def choose():
        path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.mov *.avi *.mkv *.webm"),
                                                     ("All files", "*.*")])
        if path:
            clip.set(path)

    def worker(cmd, env):
        try:
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, encoding="utf-8", errors="replace", env=env,
                                  creationflags=subprocess.CREATE_NO_WINDOW) as proc:
                for line in proc.stdout:
                    events.put(line)
                events.put(proc.wait())
        except OSError as exc:
            events.put(str(exc) + "\n")
            events.put(1)

    def run():
        nonlocal output
        path = Path(clip.get())
        if not clip.get() or not path.is_file():
            status.set("Choose an existing video file.")
            return
        from .media import ffmpeg_exe
        if not ffmpeg_exe():
            status.set("Install the packages in requirements.txt to enable video export.")
            return
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        output = Path.cwd() / "results" / f"{path.stem} {stamp}"
        python = Path(sys.executable).with_name("python.exe")
        cmd = [str(python), "-X", "utf8", "-u", "-m", "cbt", "analyse", str(path.resolve()),
               "--handed", handed.get(), "--on-field", call.get(), "--ball", ball.get(),
               "--outdir", str(output), *modes[exports.get()]]
        env = os.environ.copy()
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")
        launch.configure(state="disabled")
        folder.configure(state="disabled")
        status.set("Analysing replay…")
        threading.Thread(target=worker, args=(cmd, env), daemon=True).start()

    def poll():
        while True:
            try:
                event = events.get_nowait()
            except Empty:
                break
            if isinstance(event, int):
                launch.configure(state="normal")
                folder.configure(state="normal" if output and output.is_dir() else "disabled")
                status.set("Review complete." if event == 0 else "Analysis stopped. See the details below.")
            else:
                log.configure(state="normal")
                log.insert("end", event)
                log.see("end")
                log.configure(state="disabled")
        root.after(100, poll)

    ttk.Label(panel, text="Review a delivery", font=("Segoe UI", 16, "bold")).grid(
        row=0, columnspan=3, sticky="w", pady=(0, 12))
    ttk.Label(panel, text="Replay clip").grid(row=1, column=0, sticky="w")
    ttk.Entry(panel, textvariable=clip).grid(row=1, column=1, sticky="ew", padx=8)
    ttk.Button(panel, text="Browse…", command=choose).grid(row=1, column=2)
    for row, (label, var, values) in enumerate((
        ("Batter", handed, ("right", "left")),
        ("On-field call", call, ("not-out", "out")),
        ("Ball colour", ball, ("red", "white")),
        ("Output", exports, tuple(modes)),
    ), 2):
        ttk.Label(panel, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(panel, textvariable=var, values=values, state="readonly").grid(
            row=row, column=1, sticky="ew", padx=8)
    launch = ttk.Button(panel, text="Analyse clip", command=run)
    launch.grid(row=6, column=0, sticky="w", pady=8)
    folder = ttk.Button(panel, text="Open results", command=lambda: os.startfile(output), state="disabled")
    folder.grid(row=6, column=1, sticky="w", padx=8)
    ttk.Label(panel, textvariable=status, wraplength=575).grid(row=7, columnspan=3, sticky="w", pady=(0, 6))
    log = tk.Text(panel, height=10, wrap="word", state="disabled", padx=8, pady=7)
    log.grid(row=8, columnspan=3, sticky="nsew")
    root.after(100, poll)
    root.mainloop()


if __name__ == "__main__":
    main()
