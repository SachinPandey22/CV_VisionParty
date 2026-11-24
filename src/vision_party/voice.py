"""Offline text-to-speech that never blocks the video loop.

Lines are spoken one at a time on a background thread. If the game produces
lines faster than they can be spoken, the oldest waiting lines are dropped
(VOICE_MAX_QUEUE), so speech stays in sync with what's on screen.

Backends:
- macOS: the built-in `say` command. pyttsx3 on macOS needs the main thread's
  run loop, which the OpenCV loop owns, so it can't be used from a thread there.
- Elsewhere: pyttsx3 (espeak on Linux, SAPI5 on Windows).
- If neither works, lines are printed instead.
"""

import queue
import shutil
import subprocess
import sys
import threading
from collections import deque

from . import config


class Voice:
    def __init__(self, enabled: bool = True, rate: int = config.VOICE_RATE):
        self.enabled = enabled
        self.rate = rate
        self._pending: deque[str] = deque()
        self._wake = queue.Queue()            # signals the worker that a line is waiting
        self._lock = threading.Lock()
        self._backend = self._pick_backend() if enabled else "print"
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def backend(self) -> str:
        return self._backend

    def _pick_backend(self) -> str:
        if sys.platform == "darwin" and shutil.which("say"):
            return "say"
        try:
            import pyttsx3  # noqa: F401
            return "pyttsx3"
        except Exception:
            return "print"

    def say(self, text: str) -> None:
        """Queue a line to be spoken. Returns immediately."""
        print(f"[voice] {text}")
        if self._backend == "print":
            return
        with self._lock:
            self._pending.append(text)
            while len(self._pending) > config.VOICE_MAX_QUEUE:
                self._pending.popleft()       # drop stale lines
        self._wake.put(None)

    def close(self) -> None:
        self._wake.put("stop")

    def _next(self) -> str | None:
        with self._lock:
            return self._pending.popleft() if self._pending else None

    def _run(self) -> None:
        engine = None
        if self._backend == "pyttsx3":
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty("rate", self.rate)
            except Exception as exc:
                print(f"[voice] pyttsx3 failed ({exc}); printing only")
                self._backend = "print"
        while True:
            if self._wake.get() == "stop":
                return
            while (text := self._next()) is not None:
                try:
                    if self._backend == "say":
                        subprocess.run(["say", "-r", str(self.rate), text], check=False)
                    elif engine is not None:
                        engine.say(text)
                        engine.runAndWait()
                except Exception as exc:
                    print(f"[voice] could not speak: {exc}")
