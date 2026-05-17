"""
Tellur — push-to-talk voice dictation with local Whisper.

Hold Ctrl+Win to talk. Release to transcribe and paste into the focused window.
Quit: Ctrl+Win+Q.

See README.md for setup, configuration, and troubleshooting.
"""

from __future__ import annotations

__version__ = "1.1.0"
APP_NAME = "Tellur"


# ===========================================================================
# stdlib
# ===========================================================================
import argparse
import inspect
import json
import logging
import logging.handlers
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections import deque
from math import sin
from pathlib import Path


# ===========================================================================
# CUDA DLL bootstrap — must run before `import faster_whisper`
# ===========================================================================
def _register_cuda_dlls() -> None:
    """Ensure CT2 can locate cuBLAS / cuDNN / NVRTC from the nvidia-* pip packages.

    CT2's pip wheel on Windows doesn't bundle CUDA libs. We prepend the
    nvidia-*/bin paths to PATH so CT2's internal loader can find them.
    """
    if sys.platform != "win32":
        return
    try:
        import nvidia
    except ImportError:
        return
    site = Path(nvidia.__path__[0])
    added: list[str] = []
    for sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin", "cuda_runtime/bin"):
        d = site / sub
        if d.is_dir():
            os.add_dll_directory(str(d))
            added.append(str(d))
    if added:
        os.environ["PATH"] = os.pathsep.join(added + [os.environ.get("PATH", "")])


_register_cuda_dlls()


# ===========================================================================
# third-party
# ===========================================================================
import numpy as np
import sounddevice as sd
import pyperclip
import keyboard
from faster_whisper import WhisperModel

from PyQt6.QtCore import Qt, QTimer, QObject, QRectF, pyqtSignal
from PyQt6.QtGui import QAction, QColor, QGuiApplication, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QPushButton, QSlider, QSystemTrayIcon,
    QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)


# ===========================================================================
# configuration
# ===========================================================================
# whisper
MODEL_NAME = "large-v3-turbo"
PREFERRED_DEVICE = "cuda"
CUDA_COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
LANGUAGE = "en"

# audio
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 0.25

# hotkey
HOTKEY_POLL_MS = 20

# overlay
OVERLAY_WIDTH = 64
OVERLAY_HEIGHT = 16
OVERLAY_BOTTOM_MARGIN = 6
BAR_COUNT = 8
BAR_WIDTH = 4
BAR_GAP = 4
BAR_MIN_HEIGHT = 2
BAR_LEVEL_SCALE = 120.0      # RMS multiplier — full bar at ~0.008 RMS
LEVEL_PUSH_HZ = 24
RESET_AFTER_DONE_SEC = 1.0

# behavior
PASTE_AFTER_TRANSCRIBE = True
PASTE_KEYSTROKE_DELAY_SEC = 0.06
CLIPBOARD_RESTORE_DELAY_SEC = 0.8

# personalization
REPLACEMENTS_FILE = "replacements.json"
HISTORY_SIZE = 1                  # transcripts fed back as Whisper context
PROMPT_MAX_CHARS = 900

# decoder robustness — guards against repetition loops on the user's own audio.
BEAM_SIZE = 5
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.05

# auto-update — checks GitHub releases on startup, offers one-click install.
UPDATE_REPO = "EddyThePro/tellur"
UPDATE_API_URL = f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest"
UPDATE_TIMEOUT_SEC = 6
UPDATE_HELPER_PARENT_WAIT_SEC = 30

# persistence — defaults to per-user app data; override with TELLUR_HOME env var.
def _default_data_dir() -> Path:
    override = os.environ.get("TELLUR_HOME")
    if override:
        return Path(override)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "Tellur"
        return Path.home() / "AppData" / "Local" / "Tellur"
    return Path.home() / ".tellur"


DATA_DIR = _default_data_dir()
HISTORY_FILE = DATA_DIR / "history.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
MAX_HISTORY_ENTRIES = 500

# logging
DEFAULT_LOG_DIR = DATA_DIR / "logs"              # TELLUR_LOG_DIR env var overrides
LOG_FILE_MAX_BYTES = 2 * 1024 * 1024
LOG_FILE_BACKUP_COUNT = 3


# ===========================================================================
# logging — asynchronous via QueueListener so I/O never blocks callers
# ===========================================================================
def setup_logging(debug: bool) -> logging.handlers.QueueListener:
    """Configure async logging. All handlers run on a background thread; the
    cost on the caller side is just a `Queue.put()` (microseconds). Returns the
    listener so callers can stop it on shutdown.
    """
    log_dir_env = os.environ.get("TELLUR_LOG_DIR")
    candidates = [Path(log_dir_env)] if log_dir_env else [DEFAULT_LOG_DIR]
    candidates.append(Path(__file__).resolve().parent / "logs")  # always-writable fallback

    log_dir: Path | None = None
    for c in candidates:
        try:
            c.mkdir(parents=True, exist_ok=True)
            # quick write probe
            probe = c / ".write_probe"
            probe.touch()
            probe.unlink()
            log_dir = c
            break
        except OSError:
            continue
    if log_dir is None:
        log_dir = Path.cwd()

    log_file = log_dir / "tellur.log"

    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=LOG_FILE_MAX_BYTES,
        backupCount=LOG_FILE_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-7s %(name)-9s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug else logging.INFO)
    console_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s %(levelname)-5s %(name)-7s %(message)s",
        datefmt="%H:%M:%S",
    ))

    log_queue: queue.Queue = queue.Queue(-1)
    queue_handler = logging.handlers.QueueHandler(log_queue)
    queue_handler.setLevel(logging.DEBUG)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(queue_handler)

    # noisy third-party libs — silence at WARNING+
    for noisy in (
        "urllib3", "huggingface_hub", "filelock", "httpcore", "httpx",
        "faster_whisper",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    listener = logging.handlers.QueueListener(
        log_queue, file_handler, console_handler, respect_handler_level=True,
    )
    listener.start()

    # capture uncaught exceptions from the main thread
    def _excepthook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_tb)
            return
        logging.getLogger("crash").critical(
            "uncaught exception in main thread",
            exc_info=(exc_type, exc_value, exc_tb),
        )
    sys.excepthook = _excepthook

    # capture uncaught exceptions from worker threads (Python 3.8+)
    def _thread_excepthook(args):
        if args.exc_type is KeyboardInterrupt:
            return
        logging.getLogger("crash").critical(
            "uncaught exception in thread '%s'",
            getattr(args.thread, "name", "?"),
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )
    threading.excepthook = _thread_excepthook

    logging.getLogger("startup").info("log file: %s", log_file)
    return listener


# Module-level loggers — kept tight so we can scan-grep easily.
log_app = logging.getLogger("app")
log_audio = logging.getLogger("audio")
log_engine = logging.getLogger("engine")
log_hotkey = logging.getLogger("hotkey")
log_paste = logging.getLogger("paste")
log_repl = logging.getLogger("repl")


# ===========================================================================
# replacements (custom dictionary, hot-reloaded)
# ===========================================================================
class Replacements:
    """User-editable dictionary that post-processes Whisper output.

    JSON format: {"wrong phrase": "right phrase", ...}. Matching is
    case-insensitive, whitespace-tolerant, and word-boundary aware so "pi"
    won't replace inside "pickle". Reloads on the fly when the file's mtime
    changes — no app restart needed.
    """

    def __init__(self, path: Path):
        self.path = path
        self._mtime = 0.0
        self._compiled: list[tuple[re.Pattern, str]] = []
        self._values: list[str] = []
        self.reload_if_changed()

    def reload_if_changed(self) -> bool:
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            if self._compiled:
                self._compiled = []
                self._values = []
                self._mtime = 0.0
                log_repl.warning("%s disappeared; using empty dictionary", self.path.name)
                return True
            return False
        if m == self._mtime:
            return False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            log_repl.exception("failed to load %s", self.path.name)
            return False
        if not isinstance(raw, dict):
            log_repl.error("%s must be a JSON object", self.path.name)
            return False

        items = sorted(raw.items(), key=lambda kv: -len(kv[0]))
        compiled: list[tuple[re.Pattern, str]] = []
        for key, val in items:
            if not isinstance(key, str) or not isinstance(val, str) or not key:
                continue
            parts = key.split()
            body = r"\s+".join(re.escape(p) for p in parts)
            pat = re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)
            compiled.append((pat, val))
        self._compiled = compiled
        self._values = [v for _, v in compiled if v.strip()]
        self._mtime = m
        log_repl.info("loaded %d replacement(s) from %s", len(compiled), self.path.name)
        return True

    def apply(self, text: str) -> str:
        self.reload_if_changed()
        if not text:
            return text
        for pat, repl in self._compiled:
            text = pat.sub(repl, text)
        return text

    @property
    def vocab(self) -> list[str]:
        seen, out = set(), []
        for v in self._values:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out


class TranscriptLog(QObject):
    """Persistent record of every transcription. Backs the History tab and
    the tray's 'Copy last' action; also provides Whisper-context for prompts.

    File writes happen on a worker thread so the transcribe path is never
    blocked by disk I/O.
    """

    entry_added = pyqtSignal(dict)
    cleared = pyqtSignal()

    def __init__(self, path: Path, max_entries: int = MAX_HISTORY_ENTRIES):
        super().__init__()
        self.path = path
        self.max_entries = max_entries
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._entries = [
                        e for e in data
                        if isinstance(e, dict) and isinstance(e.get("text"), str)
                    ]
                    log_app.info("loaded %d history entries from %s",
                                 len(self._entries), self.path.name)
        except Exception:
            log_app.exception("failed to load %s", self.path.name)

    def _save_async(self) -> None:
        snapshot = list(self._entries)
        def _w():
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(snapshot, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self.path)
            except Exception:
                log_app.exception("failed to save history")
        threading.Thread(target=_w, daemon=True, name="history-save").start()

    def add(self, text: str, raw: str | None = None) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None
        entry: dict = {"text": text, "ts": time.time()}
        if raw and raw != text:
            entry["raw"] = raw
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]
        self._save_async()
        self.entry_added.emit(entry)
        return entry

    def last(self) -> dict | None:
        with self._lock:
            return self._entries[-1] if self._entries else None

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self._save_async()
        self.cleared.emit()
        log_app.info("history cleared")

    def context_for_prompt(self, n: int = HISTORY_SIZE) -> str:
        with self._lock:
            recent = self._entries[-n:]
        return " ".join(e["text"] for e in recent if e.get("text"))


class Settings(QObject):
    """User preferences. Persisted to settings.json, live-applied via signal."""

    changed = pyqtSignal()

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        # defaults
        self.auto_paste = PASTE_AFTER_TRANSCRIBE
        self.sensitivity = BAR_LEVEL_SCALE
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self.auto_paste = bool(data.get("auto_paste", self.auto_paste))
                    self.sensitivity = float(data.get("sensitivity", self.sensitivity))
                    log_app.info("loaded settings: auto_paste=%s sensitivity=%.1f",
                                 self.auto_paste, self.sensitivity)
        except Exception:
            log_app.exception("failed to load settings")

    def save(self) -> None:
        data = {"auto_paste": self.auto_paste, "sensitivity": self.sensitivity}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            log_app.exception("failed to save settings")
        self.changed.emit()


def build_initial_prompt(log: TranscriptLog) -> str | None:
    """Construct Whisper's initial_prompt from recent transcript context only.

    Dictionary vocabulary is NOT injected here — that's what `hotwords` is for.
    Stuffing a comma-separated vocab list into initial_prompt makes Whisper
    hallucinate from those tokens (e.g. "HTML1, HTML2, HTML3...") or return
    empty results on real audio.
    """
    ctx = log.context_for_prompt().strip()
    if not ctx:
        return None
    if len(ctx) > PROMPT_MAX_CHARS:
        ctx = ctx[-PROMPT_MAX_CHARS:]
    return ctx


def humanize_time(ts: float) -> str:
    if not ts:
        return "—"
    diff = time.time() - ts
    if diff < 60:
        return "just now"
    if diff < 3600:
        return f"{int(diff / 60)}m ago"
    if diff < 86400:
        return f"{int(diff / 3600)}h ago"
    days = int(diff / 86400)
    if days < 7:
        return f"{days}d ago"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(ts))


# ===========================================================================
# audio capture
# ===========================================================================
class AudioRecorder:
    """Captures mono float32 audio at SAMPLE_RATE while held. Audio callback is
    kept lean — no logging or heavy computation — so it doesn't stall."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._level = 0.0

    def _callback(self, indata, _frames, _time_info, _status) -> None:
        chunk = indata.copy().reshape(-1).astype(np.float32)
        self._frames.append(chunk)
        self._level = float(np.sqrt(np.mean(chunk * chunk)))

    def start(self) -> None:
        self._frames = []
        self._level = 0.0
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            blocksize=0,
        )
        self._stream.start()
        log_audio.debug("recording started")

    def stop(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        if not self._frames:
            log_audio.debug("recording stopped — no frames captured")
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._frames).astype(np.float32)
        log_audio.debug("recording stopped — %d samples (%.2fs)",
                        audio.size, audio.size / self.sample_rate)
        return audio

    @property
    def level(self) -> float:
        return self._level


# ===========================================================================
# whisper engine
# ===========================================================================
class WhisperEngine:
    """Wraps faster-whisper. Single-threaded transcribe; callers serialize."""

    def __init__(self, name: str = MODEL_NAME):
        self.name = name
        self.device = PREFERRED_DEVICE
        self.compute_type = CUDA_COMPUTE_TYPE
        self._model: WhisperModel | None = None
        self._lock = threading.Lock()
        self._supports_hotwords = False

    def load(self) -> WhisperModel:
        with self._lock:
            if self._model is not None:
                return self._model
            t0 = time.monotonic()
            try:
                self._model = WhisperModel(
                    self.name, device="cuda", compute_type=CUDA_COMPUTE_TYPE,
                )
                self.device, self.compute_type = "cuda", CUDA_COMPUTE_TYPE
            except Exception:
                log_engine.exception("CUDA load failed — falling back to CPU")
                self._model = WhisperModel(
                    self.name, device="cpu", compute_type=CPU_COMPUTE_TYPE,
                )
                self.device, self.compute_type = "cpu", CPU_COMPUTE_TYPE
            try:
                sig = inspect.signature(self._model.transcribe)
                self._supports_hotwords = "hotwords" in sig.parameters
            except (TypeError, ValueError):
                self._supports_hotwords = False
            log_engine.info(
                "model loaded name=%s device=%s compute=%s hotwords=%s in %.2fs",
                self.name, self.device, self.compute_type,
                self._supports_hotwords, time.monotonic() - t0,
            )
            return self._model

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
    ) -> str:
        if audio.size < int(SAMPLE_RATE * MIN_AUDIO_SECONDS):
            log_engine.debug("audio too short (%d samples); skipping", audio.size)
            return ""
        model = self.load()
        kwargs = dict(
            language=LANGUAGE,
            beam_size=BEAM_SIZE,
            no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
            repetition_penalty=REPETITION_PENALTY,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 250},
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
        )
        if self._supports_hotwords and hotwords:
            kwargs["hotwords"] = hotwords
        t0 = time.monotonic()
        segments, _info = model.transcribe(audio, **kwargs)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        log_engine.info(
            "transcribed audio=%.2fs in %.0fms text=%r",
            audio.size / SAMPLE_RATE, (time.monotonic() - t0) * 1000, text,
        )
        return text


# ===========================================================================
# text injection (clipboard + simulated Ctrl+V, race-safe restore)
# ===========================================================================
class TextInjector:
    """Pastes via clipboard + Ctrl+V. The previous clipboard contents are
    captured on the FIRST paste in a rapid-fire session and only restored
    after CLIPBOARD_RESTORE_DELAY_SEC of quiet, so consecutive dictations
    never lose the user's original clipboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self._saved: str | None = None
        self._gen = 0

    def paste(self, text: str) -> None:
        if not text:
            return
        with self._lock:
            if self._saved is None:
                try:
                    self._saved = pyperclip.paste()
                except Exception:
                    log_paste.debug("clipboard read failed", exc_info=True)
            self._gen += 1
            my_gen = self._gen

        try:
            pyperclip.copy(text)
        except Exception:
            log_paste.exception("clipboard copy failed")
            return
        time.sleep(PASTE_KEYSTROKE_DELAY_SEC)
        try:
            keyboard.send("ctrl+v")
        except Exception:
            log_paste.exception("paste keystroke failed")
            return
        log_paste.debug("pasted %d chars", len(text))

        def _restore():
            time.sleep(CLIPBOARD_RESTORE_DELAY_SEC)
            with self._lock:
                if my_gen != self._gen:
                    return  # newer paste pending — don't clobber
                saved, self._saved = self._saved, None
            if saved is None:
                return
            try:
                pyperclip.copy(saved)
                log_paste.debug("clipboard restored (%d chars)", len(saved))
            except Exception:
                log_paste.exception("clipboard restore failed")

        threading.Thread(target=_restore, daemon=True, name="clipboard-restore").start()


# ===========================================================================
# hotkey watcher — polls Ctrl + Win held; emits press / release transitions
# ===========================================================================
class HotkeyWatcher(QObject):
    pressed = pyqtSignal()
    released = pyqtSignal()
    quit_requested = pyqtSignal()

    def __init__(self, poll_ms: int = HOTKEY_POLL_MS):
        super().__init__()
        self._held = False
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)
        try:
            keyboard.add_hotkey("ctrl+windows+q", self.quit_requested.emit)
            log_hotkey.info("registered quit hotkey: ctrl+win+q")
        except Exception:
            log_hotkey.exception("failed to register quit hotkey")

    def start(self) -> None:
        self._timer.start()
        log_hotkey.info("watching ctrl+win as push-to-talk")

    def _all_held(self) -> bool:
        try:
            ctrl = keyboard.is_pressed("ctrl")
            win = keyboard.is_pressed("left windows") or keyboard.is_pressed("right windows")
            return bool(ctrl and win)
        except Exception:
            log_hotkey.exception("keyboard poll failed")
            return False

    def _poll(self) -> None:
        held = self._all_held()
        if held and not self._held:
            self._held = True
            self.pressed.emit()
        elif not held and self._held:
            self._held = False
            self.released.emit()


# ===========================================================================
# overlay — tiny indicator at bottom-center of the primary monitor
# ===========================================================================
class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.resize(OVERLAY_WIDTH, OVERLAY_HEIGHT)

        self._state = "idle"
        self._pulse_t = 0.0
        self._history: deque[float] = deque([0.0] * BAR_COUNT, maxlen=BAR_COUNT)
        self._scale = float(BAR_LEVEL_SCALE)

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick)
        self._anim.start(33)
        self._position()

    def set_scale(self, scale: float) -> None:
        self._scale = max(1.0, float(scale))
        if self._state == "recording":
            self.update()

    def _position(self) -> None:
        screen = QGuiApplication.primaryScreen().availableGeometry()
        x = screen.center().x() - self.width() // 2
        y = screen.bottom() - self.height() - OVERLAY_BOTTOM_MARGIN
        self.move(x, y)

    def set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            if state == "recording":
                self._history = deque([0.0] * BAR_COUNT, maxlen=BAR_COUNT)
            self.update()

    def push_level(self, level: float) -> None:
        self._history.append(level)
        if self._state == "recording":
            self.update()

    def _tick(self) -> None:
        self._pulse_t += 0.06
        if self._state == "transcribing":
            self.update()

    def paintEvent(self, _e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self._state == "recording":
            self._draw_bars(p)
        else:
            self._draw_dot(p)

    def _draw_bars(self, p: QPainter) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(240, 90, 90))
        total_w = BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_GAP
        x0 = (self.width() - total_w) // 2
        max_h = self.height()
        cy = self.height() / 2
        for i, level in enumerate(self._history):
            scaled = min(level * self._scale, 1.0)
            h = max(BAR_MIN_HEIGHT, scaled * max_h)
            x = x0 + i * (BAR_WIDTH + BAR_GAP)
            y = cy - h / 2
            p.drawRoundedRect(QRectF(x, y, BAR_WIDTH, h), BAR_WIDTH / 2, BAR_WIDTH / 2)

    def _draw_dot(self, p: QPainter) -> None:
        color, radius = self._dot_style()
        cx = self.width() / 2
        cy = self.height() / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))

    def _dot_style(self) -> tuple[QColor, float]:
        if self._state == "transcribing":
            r = 2.4 + (sin(self._pulse_t * 7) + 1) * 1.1
            return QColor(96, 168, 255), r
        if self._state == "done":
            return QColor(96, 200, 138), 2.6
        if self._state == "error":
            return QColor(230, 160, 80), 2.6
        return QColor(140, 144, 156, 170), 1.8


# ===========================================================================
# auto-update — one-click in-app upgrade from GitHub Releases
# ===========================================================================
log_update = logging.getLogger("update")


def _version_tuple(v: str) -> tuple:
    """Parse a SemVer-ish version string to a comparable tuple. Non-numeric
    parts are treated as -1 so pre-release suffixes sort below releases."""
    parts: list[int] = []
    for chunk in (v or "").lstrip("v").split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            # strip non-digits, fall back to 0
            digits = "".join(c for c in chunk if c.isdigit())
            parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _pid_alive(pid: int) -> bool:
    """Return True iff a process with the given PID is currently running.
    Used by the update helper to know when the old app has fully exited."""
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        import ctypes
        from ctypes import wintypes
    except Exception:
        return False
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    STILL_ACTIVE = 259
    h = ctypes.windll.kernel32.OpenProcess(
        PROCESS_QUERY_LIMITED_INFORMATION, False, pid,
    )
    if not h:
        return False
    try:
        code = wintypes.DWORD()
        ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        return code.value == STILL_ACTIVE
    finally:
        ctypes.windll.kernel32.CloseHandle(h)


def check_for_update() -> dict | None:
    """Query the GitHub releases API for the latest tagged release. Returns
    an info dict if a newer-than-current version is available, else None.

    Network failures, parse errors, and rate limits all map to None — silent
    so the app keeps running normally.
    """
    headers = {
        "User-Agent": f"{APP_NAME}/{__version__}",
        "Accept": "application/vnd.github+json",
    }
    req = urllib.request.Request(UPDATE_API_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=UPDATE_TIMEOUT_SEC) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        log_update.info("update check failed: %s", e)
        return None
    except Exception:
        log_update.exception("update check unexpectedly raised")
        return None

    tag = (data.get("tag_name") or "").strip()
    if not tag:
        return None
    remote_v = _version_tuple(tag)
    local_v = _version_tuple(__version__)
    if remote_v <= local_v:
        log_update.info("on latest version (%s)", __version__)
        return None

    # find the first .zip asset
    zip_url = None
    zip_size = 0
    for asset in data.get("assets", []) or []:
        name = asset.get("name") or ""
        if name.lower().endswith(".zip"):
            zip_url = asset.get("browser_download_url")
            zip_size = asset.get("size") or 0
            break
    if not zip_url:
        log_update.info("release %s has no zip asset; skipping", tag)
        return None

    body = data.get("body") or ""
    log_update.info("update available: %s (%d bytes)", tag, zip_size)
    return {
        "tag": tag,
        "version": tag.lstrip("v"),
        "zip_url": zip_url,
        "zip_size": zip_size,
        "notes": body[:1500],
        "html_url": data.get("html_url") or "",
    }


def _spawn_update_helper(zip_path: Path, install_dir: Path) -> None:
    """Launch a detached helper process that will replace install files
    after this process exits and then start the new version."""
    args = [
        sys.executable,
        str(install_dir / "tellur.py"),
        "--update-helper",
        "--zip", str(zip_path),
        "--install-dir", str(install_dir),
        "--parent-pid", str(os.getpid()),
    ]
    creationflags = 0
    if sys.platform == "win32":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        creationflags = 0x00000008 | 0x00000200 | 0x08000000
    subprocess.Popen(
        args,
        creationflags=creationflags,
        close_fds=True,
        cwd=str(install_dir),
    )
    log_update.info("spawned update helper for %s", zip_path)


# Files in the install dir we should NOT overwrite during update because the
# user may have customized them. Their data (settings.json / history.json /
# logs) lives in TELLUR_HOME, not the install dir, so they're not at risk.
_PRESERVE_FILES = frozenset({"replacements.json"})


def _update_helper_main(args) -> int:
    """Standalone entry point used when tellur.py is invoked with
    --update-helper. Lives in this file so it ships with every version going
    forward. Does the post-exit file replacement and relaunches the app.

    Logs to a side-channel file in %TEMP% so issues are debuggable even when
    the main log isn't writable yet.
    """
    log_dir = Path(tempfile.gettempdir())
    helper_log = log_dir / "tellur_update.log"

    def w(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
        try:
            with helper_log.open("a", encoding="utf-8") as f:
                f.write(line)
        except OSError:
            pass

    w(f"helper start: pid={os.getpid()} parent={args.parent_pid} zip={args.zip}")

    zip_path = Path(args.zip)
    install_dir = Path(args.install_dir)

    # 1) wait for parent to exit so its file handles release.
    deadline = time.monotonic() + UPDATE_HELPER_PARENT_WAIT_SEC
    while time.monotonic() < deadline and _pid_alive(args.parent_pid):
        time.sleep(0.2)
    if _pid_alive(args.parent_pid):
        w(f"parent {args.parent_pid} still alive after timeout — aborting")
        return 1
    w("parent exited; extracting")

    # 2) verify zip integrity, then extract to a staging dir.
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                w(f"zip integrity check failed on member: {bad}")
                return 2
    except (zipfile.BadZipFile, OSError) as e:
        w(f"zip open failed: {e}")
        return 2

    with tempfile.TemporaryDirectory(prefix="tellur_upd_") as staging:
        staging_dir = Path(staging)
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(staging_dir)
        except (zipfile.BadZipFile, OSError) as e:
            w(f"extract failed: {e}")
            return 3

        # 3) copy files over the install dir, skipping preserved user files.
        copied = 0
        skipped = 0
        for src in staging_dir.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(staging_dir)
            dst = install_dir / rel
            if rel.name in _PRESERVE_FILES and dst.exists():
                w(f"preserved: {rel}")
                skipped += 1
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
            except OSError as e:
                w(f"copy failed for {rel}: {e}")
                return 4
        w(f"install complete: copied={copied} skipped={skipped}")

    # 4) launch the new version via run.bat.
    run_bat = install_dir / "run.bat"
    if run_bat.exists():
        flags = 0
        if sys.platform == "win32":
            flags = 0x00000008 | 0x00000200  # DETACHED | NEW_GROUP
        try:
            subprocess.Popen(
                ["cmd", "/c", str(run_bat)],
                cwd=str(install_dir),
                creationflags=flags,
                close_fds=True,
            )
            w(f"relaunched via {run_bat}")
        except OSError as e:
            w(f"relaunch failed: {e}")
            return 5
    else:
        w(f"run.bat missing at {run_bat}; not relaunched")

    # 5) best-effort cleanup of the downloaded zip + its temp dir.
    try:
        zip_path.unlink(missing_ok=True)
        parent = zip_path.parent
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    except OSError:
        pass

    w("helper exit ok")
    return 0


class Updater(QObject):
    """Background update checker + one-click installer."""

    state_changed = pyqtSignal(str)        # human-readable status for UI
    update_found = pyqtSignal(dict)        # info dict from check_for_update
    install_started = pyqtSignal()
    install_failed = pyqtSignal(str)

    def __init__(self, install_dir: Path):
        super().__init__()
        self.install_dir = install_dir
        self._info: dict | None = None
        self._busy = False
        self._lock = threading.Lock()

    @property
    def latest_info(self) -> dict | None:
        return self._info

    def check_async(self) -> None:
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self.state_changed.emit("Checking for updates…")
        threading.Thread(target=self._check_worker, daemon=True, name="update-check").start()

    def _check_worker(self) -> None:
        try:
            info = check_for_update()
        finally:
            with self._lock:
                self._busy = False
        if info is None:
            self._info = None
            self.state_changed.emit(f"Tellur {__version__} — up to date")
        else:
            self._info = info
            self.state_changed.emit(f"Update available: v{info['version']}")
            self.update_found.emit(info)

    def install_async(self) -> None:
        info = self._info
        if not info:
            self.install_failed.emit("No update info — run a check first.")
            return
        with self._lock:
            if self._busy:
                return
            self._busy = True
        self.install_started.emit()
        threading.Thread(
            target=self._install_worker, args=(info,),
            daemon=False, name="update-install",
        ).start()

    def _install_worker(self, info: dict) -> None:
        try:
            self.state_changed.emit(f"Downloading v{info['version']}…")
            tmpdir = Path(tempfile.mkdtemp(prefix="tellur_dl_"))
            zip_path = tmpdir / f"Tellur-{info['version']}.zip"
            try:
                urllib.request.urlretrieve(info["zip_url"], zip_path)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                log_update.exception("download failed")
                self.install_failed.emit(f"Download failed: {e}")
                return

            # quick integrity check
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    bad = zf.testzip()
                    if bad is not None:
                        self.install_failed.emit(f"Downloaded archive is corrupt ({bad})")
                        return
            except zipfile.BadZipFile:
                self.install_failed.emit("Downloaded file is not a valid zip")
                return

            self.state_changed.emit("Installing…")
            try:
                _spawn_update_helper(zip_path, self.install_dir)
            except OSError as e:
                log_update.exception("helper spawn failed")
                self.install_failed.emit(f"Could not start updater: {e}")
                return

            self.state_changed.emit("Restarting Tellur…")
            # Give the helper a beat to start polling our PID, then quit.
            QTimer.singleShot(400, QApplication.instance().quit)
        finally:
            with self._lock:
                self._busy = False


# ===========================================================================
# system tray icon
# ===========================================================================
class TrayIcon(QSystemTrayIcon):
    """System-tray icon. Left-click or double-click opens the main panel; the
    right-click menu has Open / Copy-last / Quit."""

    show_panel = pyqtSignal()
    copy_last = pyqtSignal()
    quit_requested = pyqtSignal()
    install_update = pyqtSignal()

    def __init__(self, updater: "Updater | None" = None, parent: QObject | None = None):
        super().__init__(self._make_icon(), parent)
        self.setToolTip(f"{APP_NAME} {__version__}")
        self._updater = updater

        menu = QMenu()
        open_act = QAction(f"Open {APP_NAME}", self)
        f = open_act.font(); f.setBold(True); open_act.setFont(f)
        open_act.triggered.connect(self.show_panel)
        menu.addAction(open_act)

        copy_act = QAction("Copy last transcription", self)
        copy_act.triggered.connect(self.copy_last)
        menu.addAction(copy_act)

        # Update entry — hidden until an update is found, then becomes visible.
        self._update_act = QAction("Install update", self)
        fu = self._update_act.font(); fu.setBold(True); self._update_act.setFont(fu)
        self._update_act.setVisible(False)
        self._update_act.triggered.connect(self.install_update)
        menu.addAction(self._update_act)

        menu.addSeparator()

        quit_act = QAction("Quit  (Ctrl+Win+Q)", self)
        quit_act.triggered.connect(self.quit_requested)
        menu.addAction(quit_act)

        self.setContextMenu(menu)
        self.activated.connect(self._on_activated)

    def set_update_available(self, info: dict) -> None:
        """Reveal an 'Install update vX.Y.Z' entry in the tray menu."""
        self._update_act.setText(f"Install update — v{info.get('version','?')}")
        self._update_act.setVisible(True)
        self.setToolTip(f"{APP_NAME} {__version__}  ·  v{info.get('version','?')} available")

    def _on_activated(self, reason) -> None:
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_panel.emit()

    @staticmethod
    def _make_icon() -> QIcon:
        pix = QPixmap(64, 64)
        pix.fill(Qt.GlobalColor.transparent)
        p = QPainter(pix)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(38, 40, 48))
        p.drawEllipse(2, 2, 60, 60)
        p.setBrush(QColor(232, 80, 80))
        p.drawEllipse(20, 20, 24, 24)
        p.end()
        return QIcon(pix)


# ===========================================================================
# main panel (History + Settings tabs)
# ===========================================================================
PANEL_QSS = """
QWidget#mainPanel {
    background-color: #1a1b1f;
    color: #e6e8ee;
    font-family: "Segoe UI";
    font-size: 10pt;
}
QWidget#mainPanel QLabel { color: #e6e8ee; }
QWidget#mainPanel QListWidget {
    background-color: #15161a;
    border: 1px solid #2a2b30;
    border-radius: 4px;
    outline: 0;
}
QWidget#mainPanel QListWidget::item {
    padding: 8px 10px;
    border-bottom: 1px solid #22232a;
}
QWidget#mainPanel QListWidget::item:hover { background-color: #20212a; }
QWidget#mainPanel QListWidget::item:selected {
    background-color: #2c3a55;
    color: #ffffff;
}
QWidget#mainPanel QPushButton {
    background-color: #2a2b30;
    color: #e6e8ee;
    border: 1px solid #3a3b42;
    border-radius: 4px;
    padding: 6px 14px;
    min-width: 70px;
}
QWidget#mainPanel QPushButton:hover { background-color: #34353c; }
QWidget#mainPanel QPushButton:pressed { background-color: #4a4b54; }
QWidget#mainPanel QTextEdit {
    background-color: #15161a;
    color: #e6e8ee;
    border: 1px solid #2a2b30;
    border-radius: 4px;
    padding: 8px;
    selection-background-color: #5078ff;
}
QWidget#mainPanel QTabWidget::pane {
    border: none;
    background-color: #1a1b1f;
}
QWidget#mainPanel QTabBar::tab {
    background-color: transparent;
    color: #9a9da6;
    padding: 8px 18px;
    border: none;
}
QWidget#mainPanel QTabBar::tab:selected {
    color: #e6e8ee;
    border-bottom: 2px solid #6080ff;
}
QWidget#mainPanel QTabBar::tab:hover:!selected { color: #ccd0d8; }
QWidget#mainPanel QCheckBox { color: #e6e8ee; spacing: 8px; }
QWidget#mainPanel QSlider::groove:horizontal {
    background: #2a2b30; height: 4px; border-radius: 2px;
}
QWidget#mainPanel QSlider::handle:horizontal {
    background: #6080ff; width: 14px; margin: -5px 0; border-radius: 7px;
}
QWidget#mainPanel QSlider::handle:horizontal:hover { background: #7a98ff; }
"""


class MainPanel(QWidget):
    """The 'open from the tray' window. Two tabs: History and Settings."""

    def __init__(
        self,
        log: TranscriptLog,
        settings: Settings,
        updater: "Updater | None" = None,
        parent: QObject | None = None,
    ):
        super().__init__()
        self._log = log
        self._settings = settings
        self._updater = updater

        self.setObjectName("mainPanel")
        self.setWindowTitle(APP_NAME)
        self.setStyleSheet(PANEL_QSS)
        self.resize(620, 520)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(0)

        tabs = QTabWidget()
        tabs.addTab(self._build_history_tab(), "History")
        tabs.addTab(self._build_settings_tab(), "Settings")
        root.addWidget(tabs)

        # subscribe to log changes
        log.entry_added.connect(self._on_entry_added)
        log.cleared.connect(self._refresh_list)
        self._refresh_list()

        # refresh time labels every 30s so "just now" rolls to "1m ago" etc.
        self._tick = QTimer(self)
        self._tick.setInterval(30_000)
        self._tick.timeout.connect(self._refresh_visible_timestamps)
        self._tick.start()

    # --- history tab ----------------------------------------------------

    def _build_history_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(10)

        self._list = QListWidget()
        self._list.itemSelectionChanged.connect(self._on_selection)
        self._list.itemDoubleClicked.connect(lambda _i: self._copy_selected())
        layout.addWidget(self._list, stretch=2)

        layout.addWidget(QLabel("Selected:"))
        self._detail = QTextEdit()
        self._detail.setReadOnly(True)
        self._detail.setPlaceholderText("Click an entry above to view its full text.")
        layout.addWidget(self._detail, stretch=1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        copy_sel = QPushButton("Copy selected")
        copy_sel.clicked.connect(self._copy_selected)
        actions.addWidget(copy_sel)
        copy_last = QPushButton("Copy most recent")
        copy_last.clicked.connect(self._copy_last)
        actions.addWidget(copy_last)
        actions.addStretch()
        clear = QPushButton("Clear history")
        clear.clicked.connect(self._clear_history)
        actions.addWidget(clear)
        layout.addLayout(actions)

        return w

    def _format_item(self, entry: dict) -> str:
        when = humanize_time(entry.get("ts", 0))
        text = (entry.get("text") or "").replace("\n", " ")
        if len(text) > 70:
            text = text[:70] + "…"
        return f"{when}  ·  {text}"

    def _refresh_list(self) -> None:
        self._list.clear()
        for entry in reversed(self._log.entries()):
            item = QListWidgetItem(self._format_item(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self._list.addItem(item)
        if self._list.count():
            self._list.setCurrentRow(0)
        else:
            self._detail.setPlainText("")

    def _refresh_visible_timestamps(self) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            entry = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(entry, dict):
                item.setText(self._format_item(entry))

    def _on_entry_added(self, entry: dict) -> None:
        item = QListWidgetItem(self._format_item(entry))
        item.setData(Qt.ItemDataRole.UserRole, entry)
        self._list.insertItem(0, item)
        while self._list.count() > self._log.max_entries:
            self._list.takeItem(self._list.count() - 1)

    def _on_selection(self) -> None:
        items = self._list.selectedItems()
        if not items:
            self._detail.setPlainText("")
            return
        entry = items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self._detail.setPlainText(entry.get("text", ""))

    def _copy_selected(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        entry = items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self._copy_to_clipboard(entry.get("text", ""), source="selected")

    def _copy_last(self) -> None:
        last = self._log.last()
        if last:
            self._copy_to_clipboard(last.get("text", ""), source="most-recent")

    def _copy_to_clipboard(self, text: str, source: str) -> None:
        if not text:
            return
        try:
            pyperclip.copy(text)
            log_app.info("panel: copied %s (%d chars)", source, len(text))
        except Exception:
            log_app.exception("panel: clipboard copy failed")

    def _clear_history(self) -> None:
        reply = QMessageBox.question(
            self, "Clear history",
            "Clear all transcription history? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._log.clear()

    # --- settings tab ---------------------------------------------------

    def _build_settings_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 20, 0, 0)
        layout.setSpacing(22)

        self._auto_paste_cb = QCheckBox("Auto-paste transcription into focused window")
        self._auto_paste_cb.setChecked(self._settings.auto_paste)
        self._auto_paste_cb.toggled.connect(self._on_auto_paste_toggled)
        layout.addWidget(self._auto_paste_cb)

        # sensitivity
        sens_box = QVBoxLayout()
        sens_box.setSpacing(6)
        sens_box.addWidget(QLabel("Mic-meter sensitivity (visual only, doesn't affect transcription)"))
        row = QHBoxLayout()
        self._sens_slider = QSlider(Qt.Orientation.Horizontal)
        self._sens_slider.setRange(20, 400)
        self._sens_slider.setValue(int(self._settings.sensitivity))
        self._sens_slider.valueChanged.connect(self._on_sens_changed)
        row.addWidget(self._sens_slider)
        self._sens_label = QLabel(str(int(self._settings.sensitivity)))
        self._sens_label.setFixedWidth(40)
        self._sens_label.setStyleSheet("color: #9a9da6;")
        row.addWidget(self._sens_label)
        sens_box.addLayout(row)
        layout.addLayout(sens_box)

        # --- software updates section ---
        if self._updater is not None:
            update_box = QVBoxLayout()
            update_box.setSpacing(6)
            update_header = QLabel("Software updates")
            update_header.setStyleSheet("font-weight: 600; color: #d8dbe2;")
            update_box.addWidget(update_header)

            row = QHBoxLayout()
            row.setSpacing(10)
            self._update_status = QLabel(f"Tellur {__version__} — checking for updates…")
            self._update_status.setStyleSheet("color: #9a9da6;")
            self._update_status.setWordWrap(True)
            row.addWidget(self._update_status, stretch=1)

            self._update_btn = QPushButton("Check for updates")
            self._update_btn.clicked.connect(self._on_update_button)
            row.addWidget(self._update_btn)
            update_box.addLayout(row)
            layout.addLayout(update_box)

            self._updater.state_changed.connect(self._on_update_state_changed)
            self._updater.update_found.connect(self._on_update_found)
            self._updater.install_failed.connect(self._on_update_failed)

        layout.addStretch()

        footer = QLabel(
            f"{APP_NAME} {__version__}  ·  hold Ctrl+Win to talk  ·  Ctrl+Win+Q to quit"
        )
        footer.setStyleSheet("color: #6a6e78; font-size: 9pt;")
        layout.addWidget(footer)

        return w

    def _on_auto_paste_toggled(self, checked: bool) -> None:
        self._settings.auto_paste = bool(checked)
        self._settings.save()

    def _on_sens_changed(self, value: int) -> None:
        self._settings.sensitivity = float(value)
        self._sens_label.setText(str(value))
        # save lazily — slider can fire many events; debounce via QTimer
        if not hasattr(self, "_sens_save_timer"):
            self._sens_save_timer = QTimer(self)
            self._sens_save_timer.setSingleShot(True)
            self._sens_save_timer.timeout.connect(self._settings.save)
        self._sens_save_timer.start(400)
        # apply live regardless of save timing
        self._settings.changed.emit()

    # --- update UI handlers --------------------------------------------

    def _on_update_button(self) -> None:
        if self._updater is None:
            return
        info = self._updater.latest_info
        if info:
            # update is known — install it
            self._update_btn.setEnabled(False)
            self._update_btn.setText("Installing…")
            self._updater.install_async()
        else:
            # re-check
            self._update_btn.setEnabled(False)
            self._update_btn.setText("Checking…")
            self._updater.check_async()

    def _on_update_state_changed(self, msg: str) -> None:
        if hasattr(self, "_update_status"):
            self._update_status.setText(msg)
        # Re-enable the button if we landed in a stable resting state.
        if hasattr(self, "_update_btn"):
            if msg.startswith("Checking") or msg.startswith("Downloading") or \
               msg.startswith("Installing") or msg.startswith("Restarting"):
                self._update_btn.setEnabled(False)
            else:
                self._update_btn.setEnabled(True)
                # text is set by the more specific handlers below

    def _on_update_found(self, info: dict) -> None:
        if hasattr(self, "_update_btn"):
            self._update_btn.setText(f"Install v{info.get('version','?')}")
            self._update_btn.setEnabled(True)

    def _on_update_failed(self, reason: str) -> None:
        if hasattr(self, "_update_status"):
            self._update_status.setText(f"Update failed: {reason}")
        if hasattr(self, "_update_btn"):
            self._update_btn.setEnabled(True)
            self._update_btn.setText("Try again")

    # --- window behavior -----------------------------------------------

    def closeEvent(self, event) -> None:
        # Closing the window just hides it — tray quit (or Ctrl+Win+Q) actually exits.
        event.ignore()
        self.hide()


# ===========================================================================
# orchestrator
# ===========================================================================
class App(QObject):
    ui_state = pyqtSignal(str)
    ui_level = pyqtSignal(float)

    def __init__(self):
        super().__init__()
        self.recorder = AudioRecorder()
        self.engine = WhisperEngine()
        self.overlay = Overlay()
        self.hotkey = HotkeyWatcher()
        self.injector = TextInjector()

        here = Path(__file__).resolve().parent
        self.install_dir = here
        self.replacements = Replacements(here / REPLACEMENTS_FILE)

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.log = TranscriptLog(HISTORY_FILE)
        self.settings = Settings(SETTINGS_FILE)
        self.updater = Updater(self.install_dir)

        # Apply persisted settings to the overlay; react to live changes.
        self.overlay.set_scale(self.settings.sensitivity)
        self.settings.changed.connect(
            lambda: self.overlay.set_scale(self.settings.sensitivity)
        )

        # Tray + panel — panel is hidden by default; opened from the tray.
        self.panel = MainPanel(self.log, self.settings, self.updater)
        self.tray = TrayIcon(self.updater, self)
        if not QSystemTrayIcon.isSystemTrayAvailable():
            log_app.warning("system tray not available — tray icon will be inactive")
        self.tray.show_panel.connect(self._show_panel)
        self.tray.copy_last.connect(self._copy_last_to_clipboard)
        self.tray.quit_requested.connect(self.quit)
        self.tray.install_update.connect(self.updater.install_async)
        self.updater.update_found.connect(self.tray.set_update_available)

        self.hotkey.pressed.connect(self.on_press)
        self.hotkey.released.connect(self.on_release)
        self.hotkey.quit_requested.connect(self.quit)
        self.ui_state.connect(self.overlay.set_state)
        self.ui_level.connect(self.overlay.push_level)

        self._level_timer = QTimer(self)
        self._level_timer.setInterval(int(1000 / LEVEL_PUSH_HZ))
        self._level_timer.timeout.connect(self._tick_level)

        # serialize GPU calls; generation token lets stale state emits be skipped
        self._engine_lock = threading.Lock()
        self._latest_gen = 0

        threading.Thread(target=self._warm_up, daemon=True, name="model-warmup").start()

        # Kick off a one-shot update check ~3s after startup so it doesn't
        # compete with model loading for network/CPU.
        QTimer.singleShot(3000, self.updater.check_async)

    def start(self) -> None:
        self.overlay.show()
        self.tray.show()
        self.hotkey.start()
        log_app.info("started — tray icon visible")

    def quit(self) -> None:
        log_app.info("quit requested")
        # Hide tray icon explicitly so it disappears immediately rather than waiting
        # for Windows tray cleanup on process exit.
        try:
            self.tray.hide()
        except Exception:
            pass
        QApplication.instance().quit()

    def _show_panel(self) -> None:
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def _copy_last_to_clipboard(self) -> None:
        last = self.log.last()
        if not last:
            log_app.info("tray: copy-last requested but no transcription yet")
            return
        text = last.get("text", "")
        try:
            pyperclip.copy(text)
            log_app.info("tray: copied last transcription (%d chars)", len(text))
        except Exception:
            log_app.exception("tray: clipboard copy failed")

    def _warm_up(self) -> None:
        try:
            self.engine.load()
        except Exception:
            log_engine.exception("model load failed")
            self.ui_state.emit("error")

    def on_press(self) -> None:
        # Bump gen FIRST so any pending reset timer from a prior release becomes
        # stale and skips its emit — otherwise it can clobber our "recording".
        self._latest_gen += 1
        log_hotkey.debug("press (gen=%d)", self._latest_gen)
        try:
            self.recorder.start()
        except Exception:
            log_audio.exception("mic start failed")
            self.ui_state.emit("error")
            return
        self.ui_state.emit("recording")
        self._level_timer.start()

    def on_release(self) -> None:
        self._level_timer.stop()
        try:
            audio = self.recorder.stop()
        except Exception:
            log_audio.exception("mic stop failed")
            self.ui_state.emit("error")
            return
        if audio.size == 0:
            log_hotkey.debug("release (no audio)")
            self.ui_state.emit("idle")
            return
        self._latest_gen += 1
        gen = self._latest_gen
        log_hotkey.debug("release → transcribe (gen=%d, %.2fs)",
                         gen, audio.size / SAMPLE_RATE)
        self.ui_state.emit("transcribing")
        threading.Thread(
            target=self._process_audio, args=(audio, gen),
            daemon=True, name=f"transcribe-{gen}",
        ).start()

    def _tick_level(self) -> None:
        self.ui_level.emit(self.recorder.level)

    def _process_audio(self, audio: np.ndarray, gen: int) -> None:
        prompt = build_initial_prompt(self.log)
        hotwords = ", ".join(self.replacements.vocab) or None
        with self._engine_lock:
            try:
                raw = self.engine.transcribe(audio, initial_prompt=prompt, hotwords=hotwords)
            except Exception:
                log_engine.exception("transcribe failed (gen=%d)", gen)
                if gen == self._latest_gen:
                    self.ui_state.emit("error")
                    self._schedule_reset(gen)
                return

        if not raw:
            if gen == self._latest_gen:
                self.ui_state.emit("idle")
            return

        text = self.replacements.apply(raw)
        if text != raw:
            log_app.info("rewrite %r -> %r", raw, text)
        if self.settings.auto_paste:
            self.injector.paste(text)
        self.log.add(text, raw=raw)

        if gen == self._latest_gen:
            self.ui_state.emit("done")
            self._schedule_reset(gen)

    def _schedule_reset(self, gen: int) -> None:
        def reset():
            time.sleep(RESET_AFTER_DONE_SEC)
            if gen == self._latest_gen:
                self.ui_state.emit("idle")
            else:
                log_app.debug("reset(gen=%d) stale (latest=%d) — skip",
                              gen, self._latest_gen)
        threading.Thread(target=reset, daemon=True, name=f"reset-{gen}").start()


# ===========================================================================
# entry point
# ===========================================================================
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=APP_NAME.lower(),
        description="Push-to-talk voice dictation. Hold Ctrl+Win to talk.",
    )
    parser.add_argument("--debug", action="store_true",
                        help="verbose console logging (DEBUG level)")
    parser.add_argument("--version", action="version",
                        version=f"{APP_NAME} {__version__}")
    # Internal: spawn-as-updater. Not user-facing.
    parser.add_argument("--update-helper", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--zip", help=argparse.SUPPRESS)
    parser.add_argument("--install-dir", help=argparse.SUPPRESS)
    parser.add_argument("--parent-pid", type=int, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def show_fatal(message: str) -> None:
    """Best-effort modal popup for startup failures."""
    try:
        app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, f"{APP_NAME} — error", message)
    except Exception:
        pass


def main() -> int:
    args = parse_args()

    # Update helper code path: runs as a detached process spawned by an older
    # Tellur instance. No Qt, no audio, no model — just file replacement.
    if args.update_helper:
        if not (args.zip and args.install_dir and args.parent_pid):
            return 64
        return _update_helper_main(args)

    listener = setup_logging(args.debug)
    startup = logging.getLogger("startup")
    startup.info("%s %s on python %s", APP_NAME, __version__, sys.version.split()[0])

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")

    qapp = QApplication(sys.argv)
    qapp.setQuitOnLastWindowClosed(False)

    try:
        app = App()
        app.start()
    except Exception as e:
        startup.exception("fatal startup error")
        show_fatal(f"{APP_NAME} failed to start:\n\n{e}\n\nSee log for details.")
        listener.stop()
        return 1

    exit_code = qapp.exec()
    startup.info("shutting down (exit=%d)", exit_code)
    listener.stop()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
