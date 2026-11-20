"""
Tellur — push-to-talk voice dictation with local Whisper.

Hold Ctrl+Win to talk. Release to transcribe and paste into the focused window.
Quit: Ctrl+Win+Q.

See README.md for setup, configuration, and troubleshooting.
"""

from __future__ import annotations

__version__ = "2.7.4"
APP_NAME = "Tellur"
# Windows Task Scheduler task that launches Tellur elevated (no UAC prompt).
# Optional: installs that never created the task keep the legacy Run-key path.
TELLUR_TASK_NAME = "Tellur"


# ===========================================================================
# stderr/stdout fix MUST run BEFORE any third-party imports.
# Under pythonw.exe, sys.stderr and sys.stdout are None. Libraries that try
# to write progress / warnings at import time (tqdm via huggingface_hub is
# the main offender) crash with AttributeError: 'NoneType' has no 'write'.
# Give them a silent sink so import-time writes are discarded harmlessly.
# ===========================================================================
import io
import sys

if sys.stderr is None:
    sys.stderr = io.StringIO()
if sys.stdout is None:
    sys.stdout = io.StringIO()


# ===========================================================================
# stdlib
# ===========================================================================
import argparse
import inspect
import itertools
import json
import logging
import logging.handlers
import os
import queue
import re
import shutil
import subprocess
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

from PyQt6.QtCore import Qt, QSize, QTimer, QObject, QRectF, pyqtSignal
from PyQt6.QtGui import (
    QAction, QColor, QFont, QGuiApplication, QIcon, QKeySequence, QPainter,
    QPainterPath, QPixmap, QPen,
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFrame, QHBoxLayout, QInputDialog,
    QKeySequenceEdit, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSlider,
    QSystemTrayIcon, QTabWidget, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)


# ===========================================================================
# configuration
# ===========================================================================
# whisper
DEFAULT_MODEL = "distil-large-v3.5"
PREFERRED_DEVICE = "cuda"
CUDA_COMPUTE_TYPE = "float16"
CPU_COMPUTE_TYPE = "int8"
LANGUAGE = "en"

# User-selectable Whisper models, shown in Settings. Models are downloaded on
# first switch (via faster-whisper / HuggingFace Hub) into the hf-cache dir.
KNOWN_MODELS: list[dict] = [
    {"name": "tiny",             "label": "Tiny",            "size_mb": 75,
     "summary": "fastest, low accuracy — for old/slow PCs"},
    {"name": "small",            "label": "Small",           "size_mb": 466,
     "summary": "balanced for modest hardware"},
    {"name": "distil-large-v3",  "label": "Distil-Large v3", "size_mb": 1500,
     "summary": "fast, English only"},
    {"name": "distil-large-v3.5", "label": "Distil-Large v3.5", "size_mb": 1500,
     "summary": "recommended — best short-form accuracy, ~1.5× faster than "
                "turbo, English only"},
    {"name": "large-v3-turbo",   "label": "Large v3 turbo",  "size_mb": 1620,
     "summary": "near-large-v3 quality, multilingual, best for long clips"},
    {"name": "large-v3",         "label": "Large v3",        "size_mb": 3094,
     "summary": "highest accuracy, slower on weaker GPUs"},
]

# Auto backend routing ("auto" stt_backend): local Whisper when the GPU is
# free, ElevenLabs when it's under load (a game, a render, a big LLM decode).
# Utilization is NVML's rolling sample — an idle desktop reads 0-15%, any real
# game reads 80%+, so the threshold has a wide safe band. The VRAM floor is
# the fp16 model (~1.6 GB) plus decode workspace; below it a local load would
# either fail or evict someone else's allocations mid-game.
AUTO_GPU_BUSY_UTIL_PCT = 50.0
AUTO_GPU_MIN_FREE_VRAM_MB = 2600
# Routing to cloud this many times in a row (GPU busy each time) drops the
# loaded local model to give its VRAM back to whatever is keeping the GPU
# busy. It reloads automatically (~1-2 s) on the next local decode.
AUTO_RELEASE_AFTER_BUSY = 3
# ...but only when VRAM is actually scarce. An idle model costs no GPU time,
# just memory, so unloading it while the card has room to spare buys the busy
# app nothing and charges the next local clip a ~1.7 s reload (100 clips paid
# that between Sep and Nov 2026, most with 20+ GB free). Below this much free
# VRAM the release still happens.
AUTO_RELEASE_BELOW_FREE_VRAM_MB = 6000
# One GPU sample answers both the record-start badge and the decode that
# follows a few seconds later; caching keeps those two decisions identical.
AUTO_GPU_CACHE_SEC = 2.0

# audio
SAMPLE_RATE = 16000
MIN_AUDIO_SECONDS = 0.25
# Local-decode rescue for quiet recordings (see WhisperEngine.transcribe):
# below SILENCE the buffer is a dead/muted mic and is decoded as-is (boosting
# pure noise 1000x just hallucinates text); below NORMALIZE it's linearly
# scaled so its peak hits TARGET before the decode.
QUIET_PEAK_SILENCE = 0.003
QUIET_PEAK_NORMALIZE = 0.30
QUIET_PEAK_TARGET = 0.70
MAX_RECORDING_SECONDS = 300   # 5 minutes — hard cap to prevent unbounded RAM

# Mic passthrough ("hear-through") — a live mic -> speakers monitor so the room
# stays audible without pulling an IEM out. Runs at 48 kHz rather than
# SAMPLE_RATE: 16 kHz is a Whisper *input* rate and sounds muffled played back.
# The block size sets the delay you hear yourself at; much above ~20 ms starts
# to disrupt speech (delayed auditory feedback), so it is kept small.
PASSTHROUGH_SAMPLE_RATE = 48000
PASSTHROUGH_BLOCK = 240              # 5 ms at 48 kHz
PASSTHROUGH_QUEUE_BLOCKS = 8         # ~40 ms ceiling on buffered audio
PASSTHROUGH_RATE_FALLBACKS = (48000, 44100)

# hotkey
HOTKEY_POLL_MS = 20
HOOK_HEALTH_POLL_MS = 5000   # cadence of the LL-hook death probe (HotkeyWatcher)
REHOOK_MIN_GAP_SEC = 30.0    # min spacing between hook reinstalls, anti-thrash

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
# Engine badge (v2.4): when dictation is being served by LOCAL Whisper instead
# of the cloud engine the user selected, the pill grows upward and paints a
# label under the bars. Silent degradation to local Whisper was invisible
# before this — the only clue was a transcript that read slightly worse.
OVERLAY_BADGE_HEIGHT = 11    # extra pill height when the badge is showing
OVERLAY_BADGE_WIDTH = 78     # pill widens to fit the label text
OVERLAY_BADGE_FONT_PT = 7
OVERLAY_BADGE_TEXT = "WHISPER"
OVERLAY_BADGE_COLOR = (255, 176, 64)   # amber — matches the "error" dot family

# behavior
PASTE_AFTER_TRANSCRIBE = True
PASTE_KEYSTROKE_DELAY_SEC = 0.06
CLIPBOARD_RESTORE_DELAY_SEC = 0.8

# Terminal-aware paste (v2.3). A simulated Ctrl+V only pastes if the focused
# window translates it into a paste — and console hosts often don't:
#   - Legacy conhost (a bare cmd.exe / PowerShell window): when a TUI app
#     (Claude Code, vim, etc.) holds the console in raw mode, conhost's
#     Ctrl+V / Shift+Insert paste shortcuts are disabled entirely
#     (they're gated on ENABLE_PROCESSED_INPUT, which raw mode clears).
#     No paste chord works; the only reliable injection is typing the
#     characters, which conhost turns into input records in any mode.
#   - A few terminal emulators reserve a different paste chord entirely
#     (mintty: Shift+Insert; Alacritty/WezTerm: Ctrl+Shift+V).
#   - Windows Terminal and the VS Code family bind Ctrl+V themselves
#     (delivered to the app as a bracketed paste), so the default path
#     already works there.
# Strategy is chosen per-paste from the foreground window's class/process.
TERMINAL_TYPE_KEY_DELAY_SEC = 0.002   # per-char gap when typing into consoles

# Legacy console host window class (cmd.exe / powershell.exe when NOT hosted
# by Windows Terminal, whose class is CASCADIA_HOSTING_WINDOW_CLASS).
CONSOLE_WINDOW_CLASS = "ConsoleWindowClass"

# Foreground process basename → paste chord overrides. mintty pastes on
# Shift+Insert; Alacritty and WezTerm reserve Ctrl+Shift+V. Anything
# unlisted keeps plain Ctrl+V — including the VS Code family (VS Code /
# Cursor / Windsurf), where Ctrl+V is bound everywhere that matters
# (editor, chat panels, AND the integrated terminal on Windows); a
# Shift+Insert override there broke chat-panel paste in practice (v2.3.1).
PASTE_CHORD_BY_PROCESS: dict[str, str] = {
    "mintty": "shift+insert",
    "alacritty": "ctrl+shift+v",
    "wezterm-gui": "ctrl+shift+v",
}

# Electron-based editors share one top-level window class, so we can't route by
# window/process alone (a Shift+Insert override for the whole app breaks the
# chat panel — see the v2.3.1 revert). Instead we inspect the focused element's
# class via UIA: an xterm.js integrated terminal focuses a textarea whose class
# contains this marker. Only then do we switch off plain Ctrl+V. Substring match
# keeps it robust to xterm.js markup changes.
XTERM_FOCUS_CLASS_MARKER = "xterm"
# Processes whose windows are Electron editors worth probing with UIA. Keeps the
# (small) UIA cost off the hot path for everything else.
ELECTRON_EDITOR_PROCS = frozenset({"cursor", "code", "windsurf", "codium", "vscodium"})

# personalization
REPLACEMENTS_FILE = "replacements.json"
HISTORY_SIZE = 1                  # transcripts fed back as Whisper context
PROMPT_MAX_CHARS = 900

# A paste must not start while Ctrl+Win is physically held: `keyboard.send`
# presses AND releases Ctrl, the PTT poll reads the same key state, so a paste
# landing mid-hold cut the user's NEXT recording short (14 cases in the logs,
# text landing in Windows shell windows). paste() waits for the release, up
# to this long, before touching the keyboard.
PASTE_WAIT_FOR_KEYS_SEC = 30.0
# ElevenLabs Scribe labels non-speech as `[pause]`, `[clicking]`, `[silence]`
# … — dictation wants those gone, not pasted (~40 pasted across the logs).
AUDIO_EVENT_TAG_RE = re.compile(r"\s*\[[^\[\]\n]{1,40}\]\s*")

# decoder robustness — guards against repetition loops on the user's own audio.
BEAM_SIZE = 5
NO_REPEAT_NGRAM_SIZE = 3
REPETITION_PENALTY = 1.05
# Models whose decoder is too shallow to survive a prompt. Distil-Whisper keeps
# 2 of large-v3's 32 decoder layers; anything placed in the <|startofprev|>
# context — the previous transcript via `initial_prompt`, or the dictionary via
# `hotwords` (faster-whisper puts both there) — makes it stop early or emit
# function-word salad ("I we , the to do and a , you know, it, I is").
# Measured 2026-08-23 on distil-large-v3.5 against 8 synthesized + 5 real
# clips: the prompted decode lost >70% of the words on 4 of them; the plain
# decode was near-verbatim on all 13. For these models the local decode sends
# nothing but the audio, and the repetition penalties (band-aids for loops the
# prompt itself induced) are dropped too.
PROMPT_FRAGILE_MODEL_PREFIXES = ("distil",)
# Truncation guard: a local decode that yields fewer than this many words per
# second of audio on a clip at least LOCAL_GUARD_MIN_SECONDS long is
# re-decoded once with the plain config and no VAD; the longer result wins.
# Conversational dictation runs ~2.8 words/s (median over 2,000 clips); 1.2 is
# below anything but long-pause speech, and the retry costs ~0.3 s.
LOCAL_GUARD_MIN_SECONDS = 5.0
LOCAL_GUARD_MIN_WORDS_PER_SEC = 1.2

# auto-update — checks GitHub releases on startup, offers one-click install.
UPDATE_REPO = "EddyThePro/tellur"
UPDATE_API_URL = f"https://api.github.com/repos/{UPDATE_REPO}/releases/latest"
UPDATE_TIMEOUT_SEC = 6
# Per-read socket timeout while streaming the release zip (a stalled CDN
# must fail, not hang the install thread forever).
UPDATE_DOWNLOAD_TIMEOUT_SEC = 30
UPDATE_HELPER_PARENT_WAIT_SEC = 30
UPDATE_MAX_ZIP_BYTES = 250 * 1024 * 1024   # refuse any release zip > 250 MB

# branding — the logo PNG ships next to tellur.py. Used for window icon,
# taskbar, alt-tab. The tray icon is intentionally a separate, simpler
# painted-in-code red dot (see TrayIcon._make_icon).
ICON_PATH = Path(__file__).resolve().parent / "tellur.png"

# Stable AppUserModelID — tells Windows to group our windows under this
# identity (with our icon) instead of pythonw.exe. Must be set BEFORE any
# windows are shown.
APP_USER_MODEL_ID = "Tellur.Dictation.PushToTalk"


def apply_app_branding(qapp: "QApplication") -> None:
    """Install the Tellur icon as the QApplication-wide window icon and
    register an explicit AppUserModelID on Windows so the taskbar groups
    the app under our identity (not under pythonw.exe / Python's feather)."""
    if ICON_PATH.exists():
        qapp.setWindowIcon(QIcon(str(ICON_PATH)))
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                APP_USER_MODEL_ID,
            )
        except Exception:
            # Non-fatal — taskbar grouping will fall back to pythonw.
            pass


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
NOTES_FILE = DATA_DIR / "notes.json"
SETTINGS_FILE = DATA_DIR / "settings.json"
DEFAULT_MARKDOWN_DIR = DATA_DIR / "transcripts"   # one .md per day
PER_CONTEXT_DICT_DIR = DATA_DIR / "replacements.d"  # v2.0 per-app overlays
# Rolling buffer of raw dictation recordings (16-bit WAV), newest
# `Settings.keep_recordings` kept. Off by default; turn it on when a clip
# transcribes wrong so the exact audio can be replayed through the decoder.
DEBUG_RECORDINGS_DIR = DATA_DIR / "debug_recordings"
MAX_HISTORY_ENTRIES = 500
MAX_NOTES_ENTRIES = 2000

# Flags that can be applied to a note. "" / None means "no flag" (the default).
# Order matters: the UI's right-click flag menu enumerates these.
NOTE_FLAGS = ["critical", "important", "followup", "random"]

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
    # comtypes logs an interface Release() at DEBUG for every COM pointer the
    # audio ducker touches: measured at 71% of all log lines, which shrank the
    # useful history in a 2 MB rotation to roughly a third of its span.
    for noisy in (
        "urllib3", "huggingface_hub", "filelock", "httpcore", "httpx",
        # huggingface_hub 2.x moved to httpx2/httpcore2, which narrate every
        # request phase at DEBUG (34 lines per model load).
        "httpcore2", "httpx2", "hpack", "h2", "hf_xet", "fsspec",
        "faster_whisper", "comtypes", "comtypes.client", "comtypes._comobject",
        "comtypes._post_coinit", "comtypes._post_coinit.unknwn",
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

    Per-context (v2.0): if `per_context_dir` is set, an additional JSON file
    matching the *active app's executable basename* (e.g. `code.json` for
    VS Code, `slack.json` for Slack) is overlaid on top of the base rules.
    Per-app entries WIN on conflict — the base dictionary is the fallback.
    """

    def __init__(self, path: Path, per_context_dir: Path | None = None):
        self.path = path
        self._mtime = 0.0
        self._raw: dict[str, str] = {}
        self._compiled: list[tuple[re.Pattern, str]] = []
        self._values: list[str] = []
        # Per-context cache: { context_key: (mtime, compiled, values) }.
        # Loaded lazily on first apply() with that context.
        self.per_context_dir = per_context_dir
        self._ctx_cache: dict[str, tuple[float, list[tuple[re.Pattern, str]], list[str]]] = {}
        # Last active context — used by `vocab` so hotwords reflect per-app rules too.
        self._last_context: str | None = None
        self.reload_if_changed()

    @staticmethod
    def _compile_rules(raw: dict) -> tuple[list[tuple[re.Pattern, object]], list[str]]:
        """Compile the dictionary into ONE alternation regex applied in a
        single pass. Keys are matched longest-first, case-insensitively, with
        any whitespace between words; the replacement is looked up by the
        normalized matched text.

        Single pass matters: the old per-rule loop fed each rule the output
        of the previous ones, so the shipped pair `"dot html": ".html"` +
        `"html": "HTML"` turned `dashboard.html` into `dashboard.HTML` (three
        real entries in the user's history). It also passed the user's value
        to `re.sub` as a template, so a backslash in a value (a Windows path)
        raised at substitution time and killed the transcribe thread. A
        lookup callback has neither problem."""
        items = sorted(raw.items(), key=lambda kv: -len(kv[0]))
        lookup: dict[str, str] = {}
        bodies: list[str] = []
        values: list[str] = []
        for key, val in items:
            if not isinstance(key, str) or not isinstance(val, str) or not key:
                continue
            parts = key.split()
            if not parts:
                continue
            norm = " ".join(parts).lower()
            if norm in lookup:
                continue
            lookup[norm] = val
            bodies.append(r"\s+".join(re.escape(p) for p in parts))
            if val.strip():
                values.append(val)
        if not bodies:
            return [], values
        # `(?<![\w.])`: a key glued to a preceding dot is part of a file name
        # or identifier (`dashboard.html`, `package.json`) — the bare
        # `"html": "HTML"` rule must not rewrite it. Group 1 captures the
        # whitespace before the key so a value that starts with "." can
        # attach to the previous word (`main dot pi` → `main.py`).
        pat = re.compile(
            r"(\s*)(?<![\w.])(?:" + "|".join(bodies) + r")(?!\w)", re.IGNORECASE)

        def _sub(m: "re.Match[str]") -> str:
            lead = m.group(1)
            key = " ".join(m.group(0)[len(lead):].split()).lower()
            val = lookup.get(key)
            if val is None:
                return m.group(0)
            if val.startswith(".") and lead:
                return val
            return lead + val

        return [(pat, _sub)], values

    def reload_if_changed(self) -> bool:
        try:
            m = self.path.stat().st_mtime
        except FileNotFoundError:
            if self._compiled:
                self._compiled = []
                self._values = []
                self._raw = {}
                self._mtime = 0.0
                log_repl.warning("%s disappeared; using empty dictionary", self.path.name)
                return True
            return False
        if m == self._mtime:
            return False
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            log_repl.exception("failed to load %s", self.path.name)
            return False
        if not isinstance(raw, dict):
            log_repl.error("%s must be a JSON object", self.path.name)
            return False
        self._raw = {str(k): str(v) for k, v in raw.items()
                     if isinstance(k, str) and isinstance(v, str)}
        self._compiled, self._values = self._compile_rules(self._raw)
        self._mtime = m
        log_repl.info("loaded %d replacement(s) from %s",
                      len(self._raw), self.path.name)
        return True

    def _load_context(self, context: str) -> tuple[list[tuple[re.Pattern, str]], list[str]] | None:
        """Load + compile a per-context overlay if present. Returns the
        merged-with-base compiled rules or None if no overlay exists.
        Caches by mtime so repeated dictations don't re-read or re-compile."""
        if not self.per_context_dir or not context:
            return None
        path = self.per_context_dir / f"{context.lower()}.json"
        try:
            m = path.stat().st_mtime
        except FileNotFoundError:
            self._ctx_cache.pop(context, None)
            return None
        cached = self._ctx_cache.get(context)
        if cached and cached[0] == m:
            return cached[1], cached[2]
        try:
            ctx_raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except (json.JSONDecodeError, OSError):
            log_repl.exception("failed to load per-context dict %s", path.name)
            return None
        if not isinstance(ctx_raw, dict):
            log_repl.error("%s must be a JSON object", path.name)
            return None
        # Merge with base: context wins on conflict.
        merged = dict(self._raw)
        for k, v in ctx_raw.items():
            if isinstance(k, str) and isinstance(v, str):
                merged[k] = v
        compiled, values = self._compile_rules(merged)
        self._ctx_cache[context] = (m, compiled, values)
        log_repl.info("loaded %d replacement(s) from %s (context=%s, %d total after merge)",
                      len(ctx_raw), path.name, context, len(merged))
        return compiled, values

    def apply(self, text: str, context: str | None = None) -> str:
        """Apply replacements. If `context` matches a per-app overlay file,
        merged rules (overlay wins on conflict) apply instead of the base
        rules only."""
        self.reload_if_changed()
        if not text:
            return text
        self._last_context = context
        compiled = self._compiled
        if context:
            ctx_result = self._load_context(context)
            if ctx_result is not None:
                compiled = ctx_result[0]
        for pat, repl in compiled:
            text = pat.sub(repl, text)
        return text

    @property
    def vocab(self) -> list[str]:
        """Vocabulary list used as Whisper hotwords. Reflects the most
        recently-applied context (if any) so per-app vocab also biases
        decoding, not just post-processing."""
        values = self._values
        if self._last_context:
            ctx = self._ctx_cache.get(self._last_context)
            if ctx is not None:
                values = ctx[2]
        seen, out = set(), []
        for v in values:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out


def _start_on_login_command() -> str:
    """Command to register in HKCU\\...\\Run for auto-start at login. Points
    at this install's run.bat so the same self-elevation + venv discovery
    + env-var resolution path runs regardless of how Tellur was launched
    historically. Quoted in case the install path contains spaces."""
    bat = Path(__file__).resolve().parent / "run.bat"
    return f'"{bat}"'


def _schtasks(*args: str) -> "subprocess.CompletedProcess":
    """Run schtasks.exe silently (CREATE_NO_WINDOW — we may be pythonw)."""
    return subprocess.run(
        ["schtasks", *args], capture_output=True,
        creationflags=0x08000000,
    )


def _tellur_task_xml() -> str | None:
    """XML of the elevated 'Tellur' scheduled task, or None if it doesn't
    exist (or schtasks failed). Existence of this task means the task owns
    autostart + elevation and the legacy Run-key path is retired."""
    if sys.platform != "win32":
        return None
    try:
        p = _schtasks("/query", "/tn", TELLUR_TASK_NAME, "/xml")
        if p.returncode != 0:
            return None
        return p.stdout.decode("utf-8", errors="replace")
    except Exception:
        log_app.debug("schtasks query failed", exc_info=True)
        return None


def _delete_legacy_run_key() -> None:
    """Remove the old HKCU\\...\\Run entry so login can't double-launch once
    the scheduled task owns autostart. Idempotent, best-effort."""
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0, winreg.KEY_SET_VALUE,
        ) as k:
            winreg.DeleteValue(k, APP_NAME)
        log_app.info("start-on-login: removed legacy Run key (task owns autostart)")
    except FileNotFoundError:
        pass  # already absent
    except Exception:
        log_app.debug("legacy Run key cleanup failed", exc_info=True)


def is_start_on_login_enabled() -> bool:
    """True if Tellur is registered for auto-start at user login.

    Preferred source of truth is the elevated 'Tellur' scheduled task
    (v2.3.3): enabled unless its XML carries an explicit <Enabled>false</>
    (which is what `schtasks /change /disable` writes). Without the task,
    falls back to reading the per-user Run key — never touches HKLM, never
    requires admin. Compares against the command we'd write, so a stale
    value from a moved install reads as 'not really us'."""
    if sys.platform != "win32":
        return False
    xml = _tellur_task_xml()
    if xml is not None:
        # Only the task-level switch under <Settings> is what /disable and
        # /enable flip. Each trigger carries its own <Enabled> too; reading
        # those as well (the old behaviour) reported a task with one disabled
        # trigger as "off" forever, which /enable can never fix — so every
        # settings save re-ran schtasks trying.
        block = re.search(r"<Settings>(.*?)</Settings>", xml, re.S | re.I)
        scope = block.group(1) if block else xml
        return "false" not in [
            m.lower() for m in re.findall(r"<Enabled>\s*(\w+)\s*</Enabled>", scope)
        ]
    try:
        import winreg
        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
        ) as k:
            try:
                value, _ = winreg.QueryValueEx(k, APP_NAME)
            except FileNotFoundError:
                return False
        return value == _start_on_login_command()
    except Exception:
        log_app.debug("is_start_on_login_enabled check failed", exc_info=True)
        return False


def apply_start_on_login(enabled: bool) -> bool:
    """Reconcile the autostart mechanism with the saved setting. Returns True
    on success. Idempotent.

    When the elevated 'Tellur' scheduled task exists it owns autostart:
    enable/disable the task (silent start at login, no UAC, paste into admin
    windows works) and keep the legacy Run key deleted so login can't launch
    twice. Toggling the task needs the elevated token Tellur normally runs
    with; a failure is logged and reported, never silently swallowed.
    Without the task, the v2.2.2 per-user Run-key behavior is unchanged."""
    if sys.platform != "win32":
        return False
    if _tellur_task_xml() is not None:
        try:
            p = _schtasks("/change", "/tn", TELLUR_TASK_NAME,
                          "/enable" if enabled else "/disable")
            if p.returncode == 0:
                _delete_legacy_run_key()
                log_app.info("start-on-login: scheduled task %s",
                             "enabled" if enabled else "disabled")
                return True
            log_app.warning(
                "start-on-login: schtasks /change failed (rc=%s): %s",
                p.returncode, p.stderr.decode(errors="replace").strip(),
            )
            return False
        except Exception:
            log_app.exception("start-on-login: task toggle failed")
            return False
    try:
        import winreg
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        if enabled:
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE,
            ) as k:
                winreg.SetValueEx(k, APP_NAME, 0, winreg.REG_SZ,
                                  _start_on_login_command())
            log_app.info("start-on-login: enabled (Run key set)")
        else:
            try:
                with winreg.OpenKey(
                    winreg.HKEY_CURRENT_USER, key_path, 0, winreg.KEY_SET_VALUE,
                ) as k:
                    winreg.DeleteValue(k, APP_NAME)
                log_app.info("start-on-login: disabled (Run key removed)")
            except FileNotFoundError:
                pass  # already absent
        return True
    except Exception:
        log_app.exception("apply_start_on_login failed (enabled=%s)", enabled)
        return False


def get_foreground_window_info() -> tuple[str | None, str | None]:
    """Return (window_class, process_basename) for the current foreground
    window — e.g. ("ConsoleWindowClass", "cmd"). The basename is lowercase,
    without extension. Windows-only — relies on ctypes Win32 calls. Either
    element is None on failure rather than raising; callers should treat
    None as "no context available"."""
    if sys.platform != "win32":
        return None, None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None, None
        cls_buf = ctypes.create_unicode_buffer(256)
        cls_name: str | None = None
        if user32.GetClassNameW(hwnd, cls_buf, len(cls_buf)):
            cls_name = cls_buf.value or None
        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if not pid.value:
            return cls_name, None
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(
            PROCESS_QUERY_LIMITED_INFORMATION, False, pid.value,
        )
        if not handle:
            return cls_name, None
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(len(buf))
            ok = kernel32.QueryFullProcessImageNameW(
                handle, 0, buf, ctypes.byref(size),
            )
            if not ok:
                return cls_name, None
            path = Path(buf.value)
            return cls_name, (path.stem.lower() or None)
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        log_app.debug("foreground-window detection failed", exc_info=True)
        return None, None


def get_foreground_process_basename() -> str | None:
    """Return the basename (without extension, lowercase) of the executable
    that owns the current foreground window. Used for per-context vocabulary
    selection. Returns None on any failure rather than raising."""
    return get_foreground_window_info()[1]


# Cached UI Automation object. Building it is expensive (COM + typelib), so we
# create it once, lazily, on the first paste into an Electron-class window. None
# means "not yet built"; False means "build failed, don't retry".
_uia_automation: object | None | bool = None


def get_focused_element_class() -> str | None:
    """Return the focused UI element's class name via UI Automation, or None.

    Electron apps (VS Code / Cursor / Windsurf) all share one opaque top-level
    window class (`Chrome_WidgetWin_1`), so the foreground *window* can't tell a
    terminal apart from the editor or a chat box. But UIA exposes the focused
    control's underlying CSS/DOM class, which IS distinct:
      - integrated terminal → 'xterm-helper-textarea' (xterm.js)
      - Cursor chat input    → 'messageInput_<hash>'
      - editor               → Monaco classes ('monaco-list', 'View', …)

    Windows-only. Best-effort: any failure (COM error, timeout, no focus)
    returns None so the caller falls back to window-class/process heuristics.
    This is called per-paste, so it must stay cheap and never raise."""
    global _uia_automation
    if sys.platform != "win32" or _uia_automation is False:
        return None
    try:
        if _uia_automation is None:
            import comtypes.client
            comtypes.client.GetModule("UIAutomationCore.dll")
            from comtypes.gen import UIAutomationClient as _U
            _uia_automation = comtypes.client.CreateObject(
                _U.CUIAutomation, interface=_U.IUIAutomation,
            )
        el = _uia_automation.GetFocusedElement()  # type: ignore[union-attr]
        return el.CurrentClassName or None
    except Exception:
        # First-time build failure → disable permanently; transient failure
        # (no focused element) → just return None and try again next paste.
        if _uia_automation is None:
            _uia_automation = False
            log_app.debug("UIA focused-element detection unavailable", exc_info=True)
        return None


def strip_audio_event_tags(text: str) -> str:
    """Remove Scribe-style `[pause]` / `[clicking]` tags; collapse the gaps
    they leave. A transcript that was ONLY tags comes back empty, which the
    callers already treat as 'nothing said'."""
    if not text or "[" not in text:
        return text
    out = AUDIO_EVENT_TAG_RE.sub(" ", text)
    return re.sub(r"\s{2,}", " ", out).strip()


def apply_smart_defaults(text: str) -> str:
    """Safe always-on normalization for every transcription, regardless of
    whether voice commands are enabled:

    - Trim leading/trailing whitespace
    - Capitalize the first letter
    - Strip trailing 'weak' punctuation (orphan comma/semicolon/colon that
      Whisper sometimes leaves on incomplete thoughts)
    - Ensure the transcript ends with terminal punctuation (. ! ?)

    Whisper already does most of the heavy lifting based on speech tonality
    and pauses; this is just the small polish pass on top.
    """
    if not text:
        return text
    text = text.strip()
    if not text:
        return text
    # Capitalize first alpha character (don't touch numbers/symbols leading).
    if text[0].isalpha():
        text = text[0].upper() + text[1:]
    # Strip trailing weak punctuation & whitespace.
    text = re.sub(r"[,;:\s]+$", "", text)
    # Ensure terminal punctuation — judged on the last character BEFORE any
    # closing quote/bracket, so `He said "hello."` and `(see above.)` don't
    # get a second period tacked on after the closer.
    core = text.rstrip("\"'’”»)]}")
    if core and core[-1] not in ".!?…":
        text = text + "."
    return text


class VoiceCommandProcessor:
    """Post-processes Whisper output to apply spoken punctuation, structural
    commands ('new line', 'new paragraph'), editing commands ('scratch that'),
    and auto-capitalization of sentence starts. All gated by a single setting
    so users can opt out and get raw transcripts.

    Pipeline order is significant — it runs AFTER the user dictionary
    (Replacements), so dictionary rewrites happen on raw Whisper text first,
    and then voice commands turn dictated punctuation into real punctuation.
    """

    # (phrase, replacement, direction)
    # direction controls how surrounding whitespace is consumed:
    #   "before" — eat preceding whitespace (e.g. comma attaches to prior word)
    #   "after"  — eat trailing whitespace (e.g. open paren attaches to next)
    #   "both"   — eat both (structural separators, newlines)
    #   "none"   — leave whitespace alone (e.g. em dash with spaces)
    PUNCTUATION: list[tuple[str, str, str]] = [
        # Multi-word phrases first so they win against single-word fallbacks.
        ("exclamation point", "!", "before"),
        ("exclamation mark",  "!", "before"),
        ("question mark",     "?", "before"),
        ("full stop",         ".", "before"),
        # Brackets / quotes — directional
        ("open parenthesis",  "(", "after"),
        ("close parenthesis", ")", "before"),
        ("open paren",        "(", "after"),
        ("close paren",       ")", "before"),
        ("open quotation",    "\"", "after"),
        ("close quotation",   "\"", "before"),
        ("open quote",        "\"", "after"),
        ("close quote",       "\"", "before"),
        ("open bracket",      "[", "after"),
        ("close bracket",     "]", "before"),
        ("open brace",        "{", "after"),
        ("close brace",       "}", "before"),
        # Em dash typically reads with spaces around it.
        ("em dash",           "—", "none"),
        # Single-word punctuation
        ("ellipsis",          "…", "before"),
        ("semicolon",         ";", "before"),
        ("period",            ".", "before"),
        ("comma",             ",", "before"),
        ("colon",             ":", "before"),
        ("dash",              "-", "both"),
        ("hyphen",            "-", "both"),
    ]

    STRUCTURAL: list[tuple[str, str, str]] = [
        ("new paragraph", "\n\n", "both"),
        ("new line",      "\n",   "both"),
        ("tab",           "\t",   "both"),
    ]

    # "scratch that" deletes everything from the most recent sentence
    # boundary (or start of utterance) up through the trigger phrase.
    SCRATCH_TRIGGERS: list[str] = [
        "scratch that",
        "delete that",
        "scratch it",
    ]

    # Priority for picking the survivor when multiple punctuation marks end
    # up adjacent (after we add ours on top of Whisper's auto-punctuation).
    # Higher = stronger.
    _PUNCT_PRIORITY: dict[str, int] = {
        ",": 1, ";": 2, ":": 3, ".": 4, "?": 5, "!": 5,
    }

    def process(self, text: str) -> str:
        if not text:
            return text
        text = self._apply_scratch(text)
        text = self._apply_replacements(text, self.STRUCTURAL)
        text = self._apply_replacements(text, self.PUNCTUATION)
        # Whisper does its own punctuation; combined with ours we end up with
        # runs like ",.." or ".,". Dedupe to the strongest single mark.
        text = self._collapse_adjacent_punctuation(text)
        # Orphan commas at the start of a new line (Whisper inserted one
        # right before a word we converted to "\n") look like noise.
        text = self._strip_leading_punct_after_newline(text)
        text = self._capitalize_sentences(text)
        text = self._normalize_whitespace(text)
        return text.strip()

    @classmethod
    def _collapse_adjacent_punctuation(cls, text: str) -> str:
        """Collapse runs of adjacent .,?!;: (with optional horizontal
        whitespace between) into the strongest single mark. Doesn't cross
        newlines, so paragraph structure is preserved."""
        priority = cls._PUNCT_PRIORITY

        def pick(m: "re.Match") -> str:
            chars = [c for c in m.group(0) if c in priority]
            if not chars:
                return m.group(0)
            return max(chars, key=lambda c: priority[c])

        # At least one punct + at least one more punct, separated only by
        # horizontal whitespace (no \n in between).
        return re.sub(r"[.,?!;:](?:[ \t]*[.,?!;:])+", pick, text)

    @staticmethod
    def _strip_leading_punct_after_newline(text: str) -> str:
        """Remove a leftover comma/semicolon/colon that ends up at the very
        start of a new line — typically Whisper's auto-punctuation that got
        stranded when an adjacent word was converted to '\\n'."""
        return re.sub(r"\n[ \t]*[,;:]+[ \t]*", "\n", text)

    @staticmethod
    def _phrase_pattern(phrase: str, direction: str) -> "re.Pattern":
        # Allow flexible internal whitespace ("new  line" → newline too).
        parts = phrase.split()
        body = r"\s+".join(re.escape(p) for p in parts)
        if direction == "before":
            return re.compile(r"\s*\b" + body + r"\b", re.IGNORECASE)
        if direction == "after":
            return re.compile(r"\b" + body + r"\b\s*", re.IGNORECASE)
        if direction == "both":
            return re.compile(r"\s*\b" + body + r"\b\s*", re.IGNORECASE)
        return re.compile(r"\b" + body + r"\b", re.IGNORECASE)

    def _apply_replacements(self, text: str, pairs: list[tuple[str, str, str]]) -> str:
        for phrase, replacement, direction in pairs:
            pat = self._phrase_pattern(phrase, direction)
            text = pat.sub(replacement, text)
        return text

    def _apply_scratch(self, text: str) -> str:
        for trigger in self.SCRATCH_TRIGGERS:
            pat = self._phrase_pattern(trigger, "after")
            while True:
                m = pat.search(text)
                if not m:
                    break
                prefix = text[:m.start()]
                # Find the most recent sentence boundary in the prefix.
                last_punct = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"))
                cut_start = last_punct + 1 if last_punct >= 0 else 0
                # Skip whitespace immediately after the boundary.
                while cut_start < len(prefix) and prefix[cut_start] in " \t\n":
                    cut_start += 1
                text = text[:cut_start] + text[m.end():]
        return text

    @staticmethod
    def _capitalize_sentences(text: str) -> str:
        # Uppercase the first letter of the utterance and any letter that
        # follows sentence-ending punctuation + whitespace.
        def _upper(m: "re.Match") -> str:
            return m.group(1) + m.group(2).upper()
        return re.sub(r"(^|[.!?]\s+)([a-z])", _upper, text)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        # Collapse runs of horizontal whitespace but preserve newlines.
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r" *\n *", "\n", text)
        return text


class TranscriptLog(QObject):
    """Persistent record of every transcription. Backs the History tab and
    the tray's 'Copy last' action; also provides Whisper-context for prompts.

    File writes happen on a worker thread so the transcribe path is never
    blocked by disk I/O.
    """

    entry_added = pyqtSignal(dict)
    entry_changed = pyqtSignal(dict)   # mutated in place (pin, edit) — UI re-renders that row
    entry_removed = pyqtSignal(dict)   # removed — UI drops the row
    cleared = pyqtSignal()

    def __init__(self, path: Path, max_entries: int = MAX_HISTORY_ENTRIES):
        super().__init__()
        self.path = path
        self.max_entries = max_entries
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        # Single dedicated saver thread fed by a queue. Each add() enqueues a
        # snapshot; the saver coalesces (only the most-recent snapshot is
        # actually written) so a burst of rapid dictations doesn't fan out
        # into N concurrent writers racing on the same .tmp filename.
        self._save_queue: "queue.Queue[list[dict] | None]" = queue.Queue()
        self._saver_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="history-saver",
        )
        self._saver_thread.start()
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8-sig"))
                if isinstance(data, list):
                    self._entries = [
                        e for e in data
                        if isinstance(e, dict) and isinstance(e.get("text"), str)
                    ]
                    log_app.info("loaded %d history entries from %s",
                                 len(self._entries), self.path.name)
        except Exception:
            log_app.exception("failed to load %s", self.path.name)

    def _save_worker(self) -> None:
        """Drain the save queue. If multiple snapshots are queued, only the
        latest is actually written (coalescing) — saves us redundant disk I/O
        when dictations are bursty."""
        while True:
            snap = self._save_queue.get()
            if snap is None:
                return  # shutdown signal (flush() at quit)
            stop = False
            # Coalesce: drain any newer snapshots that landed while waiting.
            while True:
                try:
                    newer = self._save_queue.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    stop = True
                    break
                snap = newer
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(snap, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self.path)
            except Exception:
                log_app.exception("failed to save history")
            if stop:
                return

    def _save_async(self) -> None:
        # Snapshot under the lock so we can't race a concurrent mutation.
        with self._lock:
            snapshot = list(self._entries)
        self._save_queue.put(snapshot)

    def flush(self, timeout: float = 2.0) -> None:
        """Write anything still queued and stop the saver. Called at quit so
        a dictation finished milliseconds before Ctrl+Win+Q isn't lost."""
        self._save_queue.put(None)
        self._saver_thread.join(timeout)

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

    def delete_entry(self, entry: dict) -> bool:
        """Remove a specific entry. First tries identity match (fast, exact);
        falls back to matching on (ts, text) if identity fails for any reason
        (defensive — shouldn't happen, but keeps the user's click from
        silently doing nothing). Returns True on success."""
        removed = False
        matched_by = None
        with self._lock:
            for i, e in enumerate(self._entries):
                if e is entry:
                    self._entries.pop(i)
                    removed = True
                    matched_by = "is"
                    break
            if not removed:
                target_ts = entry.get("ts")
                target_text = entry.get("text")
                for i, e in enumerate(self._entries):
                    if e.get("ts") == target_ts and e.get("text") == target_text:
                        self._entries.pop(i)
                        removed = True
                        matched_by = "ts+text"
                        break
        log_app.info(
            "delete_entry: removed=%s matched_by=%s text=%r",
            removed, matched_by, (entry.get("text") or "")[:40],
        )
        if removed:
            self._save_async()
            self.entry_removed.emit(entry)
        return removed

    def purge_older_than(self, max_age_seconds: float) -> int:
        """Drop entries older than `max_age_seconds` (entry.ts < now - age).
        Returns the count of removed entries. Emits `cleared` if anything was
        removed so listeners refresh wholesale."""
        if max_age_seconds <= 0:
            return 0
        cutoff = time.time() - max_age_seconds
        removed = 0
        with self._lock:
            before = len(self._entries)
            self._entries = [e for e in self._entries if e.get("ts", 0) >= cutoff]
            removed = before - len(self._entries)
        if removed:
            self._save_async()
            self.cleared.emit()
            log_app.info("history purge: removed %d entries older than %.1f days",
                         removed, max_age_seconds / 86400)
        return removed

    def update_text(self, entry: dict, new_text: str) -> bool:
        """Edit a transcript's text in place (used by inline-edit). Returns
        True if the entry was found and updated."""
        new_text = (new_text or "").strip()
        if not new_text:
            return False
        found = False
        with self._lock:
            for e in self._entries:
                if e is entry:
                    e["text"] = new_text
                    e["edited"] = True
                    found = True
                    break
        if found:
            self._save_async()
            self.entry_changed.emit(entry)
        return found

    def context_for_prompt(self, n: int = HISTORY_SIZE) -> str:
        with self._lock:
            recent = self._entries[-n:]
        return " ".join(e["text"] for e in recent if e.get("text"))


class NotesLog(QObject):
    """Persistent record of voice-captured notes (Ctrl+Win+N).

    Same JSON-on-disk pattern as TranscriptLog, but each entry also carries
    an `app` field (foreground process basename at capture time, auto-flag)
    and an optional `flag` field (one of NOTE_FLAGS, or "" for unflagged).

    Notes are *not* pasted into the focused window — they're captured for
    later review in the Notes tab.
    """

    entry_added = pyqtSignal(dict)
    entry_changed = pyqtSignal(dict)
    entry_removed = pyqtSignal(dict)
    cleared = pyqtSignal()

    def __init__(self, path: Path, max_entries: int = MAX_NOTES_ENTRIES):
        super().__init__()
        self.path = path
        self.max_entries = max_entries
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self._save_queue: "queue.Queue[list[dict] | None]" = queue.Queue()
        self._saver_thread = threading.Thread(
            target=self._save_worker, daemon=True, name="notes-saver",
        )
        self._saver_thread.start()
        self._load()

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8-sig"))
                if isinstance(data, list):
                    self._entries = [
                        e for e in data
                        if isinstance(e, dict) and isinstance(e.get("text"), str)
                    ]
                    log_app.info("loaded %d notes from %s",
                                 len(self._entries), self.path.name)
        except Exception:
            log_app.exception("failed to load %s", self.path.name)

    def _save_worker(self) -> None:
        while True:
            snap = self._save_queue.get()
            if snap is None:
                return
            stop = False
            while True:
                try:
                    newer = self._save_queue.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    stop = True
                    break
                snap = newer
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_suffix(self.path.suffix + ".tmp")
                tmp.write_text(
                    json.dumps(snap, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                tmp.replace(self.path)
            except Exception:
                log_app.exception("failed to save notes")
            if stop:
                return

    def _save_async(self) -> None:
        with self._lock:
            snapshot = list(self._entries)
        self._save_queue.put(snapshot)

    def flush(self, timeout: float = 2.0) -> None:
        """Write anything still queued and stop the saver (see
        TranscriptLog.flush)."""
        self._save_queue.put(None)
        self._saver_thread.join(timeout)

    def add(self, text: str, *, app: str | None = None,
            flag: str = "", raw: str | None = None) -> dict | None:
        text = (text or "").strip()
        if not text:
            return None
        entry: dict = {"text": text, "ts": time.time()}
        if app:
            entry["app"] = app
        if flag:
            entry["flag"] = flag
        if raw and raw != text:
            entry["raw"] = raw
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.max_entries:
                self._entries = self._entries[-self.max_entries:]
        self._save_async()
        self.entry_added.emit(entry)
        return entry

    def entries(self) -> list[dict]:
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
        self._save_async()
        self.cleared.emit()
        log_app.info("notes cleared")

    def delete_entry(self, entry: dict) -> bool:
        removed = False
        with self._lock:
            for i, e in enumerate(self._entries):
                if e is entry:
                    self._entries.pop(i)
                    removed = True
                    break
            if not removed:
                target_ts = entry.get("ts")
                target_text = entry.get("text")
                for i, e in enumerate(self._entries):
                    if e.get("ts") == target_ts and e.get("text") == target_text:
                        self._entries.pop(i)
                        removed = True
                        break
        if removed:
            self._save_async()
            self.entry_removed.emit(entry)
        return removed

    def set_flag(self, entry: dict, flag: str) -> bool:
        """Set or clear the flag on a note. flag="" means unflag.
        Returns True if the entry was found."""
        if flag and flag not in NOTE_FLAGS:
            log_app.warning("set_flag: ignoring unknown flag %r", flag)
            return False
        found = False
        with self._lock:
            for e in self._entries:
                if e is entry:
                    if flag:
                        e["flag"] = flag
                    else:
                        e.pop("flag", None)
                    found = True
                    break
        if found:
            self._save_async()
            self.entry_changed.emit(entry)
            log_app.info("note flag -> %r (text=%r)",
                         flag or "", (entry.get("text") or "")[:40])
        return found


def export_history(entries: list[dict], path: Path, fmt: str) -> int:
    """Export `entries` to `path` in the requested format. Returns the count
    of exported entries. Supported formats: txt, md, json, csv."""
    fmt = fmt.lower().lstrip(".")
    path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "json":
        path.write_text(json.dumps(entries, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    elif fmt == "csv":
        import csv
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp_iso", "ts", "text", "raw", "edited"])
            for e in entries:
                ts = e.get("ts", 0)
                iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts)) if ts else ""
                w.writerow([iso, ts, e.get("text", ""), e.get("raw", ""),
                            bool(e.get("edited", False))])
    elif fmt == "md":
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# Tellur transcripts — exported {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for e in entries:
                ts = e.get("ts", 0)
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "?"
                f.write(f"## {stamp}\n\n{e.get('text', '').strip()}\n\n")
    elif fmt == "txt":
        with path.open("w", encoding="utf-8") as f:
            for e in entries:
                ts = e.get("ts", 0)
                stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else "?"
                f.write(f"[{stamp}] {e.get('text', '').strip()}\n")
    else:
        raise ValueError(f"unsupported export format: {fmt}")
    return len(entries)


class Settings(QObject):
    """User preferences. Persisted to settings.json, live-applied via signal."""

    changed = pyqtSignal()

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        # defaults
        self.auto_paste = PASTE_AFTER_TRANSCRIBE
        self.sensitivity = BAR_LEVEL_SCALE
        self.model_name = DEFAULT_MODEL
        self.voice_commands = False   # opt-in: changes transcript output
        self.hotkey_mode = "hold"     # "hold" or "toggle"
        # Audio input config (v1.6):
        # - input_device_name: human-readable device name, persisted so the
        #   choice survives index reshuffles. None = system default.
        # - input_gain: linear multiplier (1.0 = no boost). Clamped 0.1..10.0.
        self.input_device_name: str | None = None
        self.input_gain: float = 1.0
        # LLM post-processing (v1.7) — opt-in; defaults aimed at LM Studio.
        self.llm_enabled: bool = False
        self.llm_base_url: str = "http://localhost:1234/v1"
        self.llm_model: str = ""
        self.llm_api_key: str = ""
        self.llm_default_prompt: str = "cleanup"
        self.llm_auto_apply: bool = False  # if True, every transcript runs through the LLM
        # Cloud transcription (v2.3) — opt-in alternative to local Whisper.
        # When stt_backend == "elevenlabs" and a key is set, dictation is sent
        # to ElevenLabs' /v1/speech-to-text (Scribe) instead of the on-device
        # model. This is NOT offline; default stays "local".
        self.stt_backend: str = "local"          # "local" | "elevenlabs"
        self.eleven_api_key: str = ""
        self.eleven_api_key_backup: str = ""     # auto-used when the primary key is out of credits
        self.eleven_model_id: str = "scribe_v3"
        # Ask Scribe to drop filler words and false starts ("um", "l- to
        # as--") instead of transcribing them literally. On by default: local
        # Whisper already writes that way, so both engines read the same.
        self.eleven_clean_transcript: bool = True
        # One-time move of a saved pre-v3 model id onto Scribe v3 (2.7.4).
        self._eleven_model_migrated: bool = False
        # Integration & automation (v1.8):
        # - send_after_paste: auto-press Enter after each paste (chat-app mode)
        # - markdown_save_enabled / markdown_folder: write each transcript to a daily .md file
        # - webhook_enabled / webhook_url / webhook_template: POST each transcript
        self.send_after_paste: bool = False
        # Terminal-aware paste (v2.3): pick the injection strategy from the
        # focused window — type into legacy console windows (no paste chord
        # works there under raw-mode TUIs like Claude Code), per-emulator
        # chords for mintty/Alacritty/WezTerm, plain Ctrl+V everywhere else.
        self.terminal_aware_paste: bool = True
        self.markdown_save_enabled: bool = False
        self.markdown_folder: str = ""        # empty → default under DATA_DIR
        self.webhook_enabled: bool = False
        self.webhook_url: str = ""
        self.webhook_template: str = '{"text": "{text}", "raw": "{raw}", "ts": {ts}}'
        # Privacy & data management (v1.9):
        # - save_history: master switch; if False, no transcripts are persisted
        # - history_retention_days: 0 = keep forever; >0 = auto-purge older entries
        self.save_history: bool = True
        self.history_retention_days: int = 0
        # Theme (v2.0) — "dark" or "light".
        self.theme: str = "dark"
        # Audio ducking (v2.2) — while a dictation/note is recording, lower
        # every other app's volume so background audio doesn't distract.
        self.duck_enabled: bool = False
        self.duck_level_pct: int = 5    # 0..100; level other apps drop to
        # Mic passthrough (v2.7) — echo the mic into the speakers at a level
        # that is independent of the Windows mic volume, so the room stays
        # audible through closed IEMs without making you quieter to everyone
        # else. passthrough_output_name is a device NAME (None = system
        # default) for the same reason input_device_name is: indexes reshuffle.
        self.passthrough_enabled: bool = False
        self.passthrough_level_pct: int = 25
        self.passthrough_output_name: str | None = None
        self.passthrough_mute_while_dictating: bool = True
        # Diagnostics (v2.7.1): keep the newest N raw recordings on disk
        # (DEBUG_RECORDINGS_DIR) so a mis-transcribed clip can be replayed.
        # 0 = off.
        self.keep_recordings: int = 0
        # Customizable hotkeys (v2.2) — secondary hotkeys are user-editable
        # via Settings; stored in the `keyboard` library's combo format
        # (e.g. "ctrl+windows+z"). The primary Ctrl+Win push-to-talk and the
        # Esc cancel are intentionally NOT in this list — the former uses
        # polling rather than add_hotkey, and the latter is universally Esc.
        self.hotkey_quit: str = "ctrl+windows+q"
        self.hotkey_repaste: str = "ctrl+windows+b"
        self.hotkey_llm: str = "ctrl+windows+l"
        self.hotkey_note: str = "ctrl+windows+z"
        # Start on Windows login (v2.2.2) — backed by an HKCU\...\Run entry
        # that points at this install's run.bat. App reconciles the registry
        # with this value on every save() so a manual registry edit (or a
        # moved install) is fixed up the next time the user touches Settings.
        self.start_on_login: bool = False
        # ElevenLabs failover (persisted): keys proven out of credits / invalid.
        # The active key is the first eleven_keys() entry NOT in this list, so the
        # choice survives restarts and marches forward as each key is used up.
        self.eleven_exhausted: list[str] = []
        # When each key was marked exhausted (unix ts). After
        # ELEVEN_EXHAUSTED_RETRY_SEC the pool probes the key again, so a
        # monthly quota reset brings it back without user action. Keys in
        # eleven_exhausted with no timestamp count as due for retry now.
        self.eleven_exhausted_at: dict[str, float] = {}
        # Absolute unix time each key may be retried. Set per-key so a transient
        # 429 costs ~a minute while a spent monthly quota costs hours. Missing
        # entries fall back to eleven_exhausted_at + ELEVEN_EXHAUSTED_RETRY_SEC,
        # which is how state written by <=v2.3.4 is interpreted.
        self.eleven_retry_after: dict[str, float] = {}
        # Last credit balance ElevenLabs reported when rejecting a key, and when
        # we learned it. Lets the pool still send SHORT clips to a key with a few
        # credits left instead of writing it off wholesale.
        self.eleven_credits_left: dict[str, float] = {}
        self.eleven_credits_at: dict[str, float] = {}
        # Third and subsequent ElevenLabs keys, tried in order after the primary
        # and backup. Unbounded because each key has its own monthly allowance.
        self.eleven_api_keys_extra: list[str] = []
        self._save_lock = threading.RLock()
        self._load()

    def eleven_keys(self) -> list[str]:
        """Ordered, de-duplicated ElevenLabs keys: primary, backup, then any
        extras from `eleven_api_keys_extra`, in the order given.

        The Scribe failover walks this list, so an out-of-credits key rotates
        to the next one. Empty slots are skipped. Each key carries its own
        monthly credit allowance (they can be on separate accounts), so adding
        keys is the only way to raise the monthly ceiling short of a bigger
        plan — hence an unbounded list rather than two fixed slots."""
        keys: list[str] = []
        for k in (self.eleven_api_key, self.eleven_api_key_backup,
                  *self.eleven_api_keys_extra):
            k = (k or "").strip()
            if k and k not in keys:
                keys.append(k)
        return keys

    def mark_eleven_exhausted(
        self,
        key: str,
        retry_after_sec: float | None = None,
        credits_left: float | None = None,
    ) -> None:
        """Record an ElevenLabs key as out of credits / invalid and persist it,
        so the failover skips it now and after every future restart. Always
        refreshes the timestamp so a failed retry re-arms the cooldown.

        `retry_after_sec` overrides the default cooldown for THIS key — a
        rate-limit (429) is transient and deserves ~a minute, while a spent
        monthly quota deserves hours. Blanket-benching a rate-limited key for
        6h was a real bug: one burst of dictation could cost a working key the
        rest of the evening.

        `credits_left` is the balance ElevenLabs reported in the rejection, so
        the pool can still route SHORT clips to a partly-drained key instead of
        writing it off entirely."""
        key = (key or "").strip()
        if not key:
            return
        now = time.time()
        cooldown = (ELEVEN_EXHAUSTED_RETRY_SEC if retry_after_sec is None
                    else max(0.0, float(retry_after_sec)))
        with self._save_lock:
            self.eleven_exhausted_at[key] = now
            self.eleven_retry_after[key] = now + cooldown
            if credits_left is None:
                self.eleven_credits_left.pop(key, None)
                self.eleven_credits_at.pop(key, None)
            else:
                self.eleven_credits_left[key] = float(credits_left)
                self.eleven_credits_at[key] = now
            if key not in self.eleven_exhausted:
                self.eleven_exhausted.append(key)
            self.save_quiet()

    def eleven_credits_known(self, key: str) -> float | None:
        """Last balance ElevenLabs reported for `key`, or None if unknown or
        too stale to trust. Balances go out of date the moment the quota
        resets, so anything older than ELEVEN_CREDITS_TTL_SEC is 'unknown'
        and the pool simply tries the key and lets the API decide."""
        key = (key or "").strip()
        seen = self.eleven_credits_at.get(key)
        if not key or seen is None:
            return None
        if time.time() - seen > ELEVEN_CREDITS_TTL_SEC:
            return None
        return self.eleven_credits_left.get(key)

    def clear_eleven_exhausted(self, key: str) -> None:
        """Re-enable a previously-exhausted key (e.g. after topping up credits)."""
        key = (key or "").strip()
        if not key:
            return
        with self._save_lock:
            if (key in self.eleven_exhausted or key in self.eleven_exhausted_at
                    or key in self.eleven_retry_after):
                if key in self.eleven_exhausted:
                    self.eleven_exhausted.remove(key)
                self.eleven_exhausted_at.pop(key, None)
                self.eleven_retry_after.pop(key, None)
                self.eleven_credits_left.pop(key, None)
                self.eleven_credits_at.pop(key, None)
                self.save_quiet()

    def eleven_exhausted_active(self) -> set[str]:
        """Exhausted keys still inside the retry cooldown. Keys past the
        cooldown (or with no recorded timestamp) are omitted, so the pool
        gives them another chance and re-marks them if they still fail.

        Honours the per-key expiry written by mark_eleven_exhausted() and falls
        back to the flat cooldown for state persisted by older versions."""
        now = time.time()
        active: set[str] = set()
        for k in self.eleven_exhausted:
            until = self.eleven_retry_after.get(k)
            if until is None:
                until = self.eleven_exhausted_at.get(k, 0.0) + ELEVEN_EXHAUSTED_RETRY_SEC
            if now < until:
                active.add(k)
        return active

    def active_stt_engine(self) -> tuple[str, str]:
        """Which engine will actually serve the next dictation, and why.

        Returns ("local" | "elevenlabs", human-readable reason). This is the
        same decision the pool makes at transcribe time, hoisted out so the
        overlay can badge the recording BEFORE the audio is sent — the user
        should know they're on local Whisper while they're still talking, not
        infer it afterwards from a worse transcript."""
        backend = (self.stt_backend or "local").lower()
        if backend == "auto":
            busy = GPU_MONITOR.busy_reason()
            if busy is None:
                return "local", "auto: GPU is free, local Whisper is fastest"
            # GPU busy — fall through to the cloud-key checks below, keeping
            # the reason so "cloud chosen but keys dead" still reads honestly.
            auto_reason = busy
        elif backend != "elevenlabs":
            return "local", "Local Whisper selected in Settings"
        else:
            auto_reason = None
        keys = self.eleven_keys()
        if not keys:
            return "local", "no ElevenLabs API key configured"
        usable = [k for k in keys if k not in self.eleven_exhausted_active()]
        if not usable:
            return "local", (
                "all %d ElevenLabs key(s) out of credits" % len(keys)
            )
        reason = "ElevenLabs Scribe (key #%d)" % (keys.index(usable[0]) + 1)
        if auto_reason:
            reason = f"auto: {auto_reason} — {reason}"
        return "elevenlabs", reason

    def _load(self) -> None:
        try:
            if self.path.exists():
                data = json.loads(self.path.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    self.auto_paste = bool(data.get("auto_paste", self.auto_paste))
                    self.sensitivity = float(data.get("sensitivity", self.sensitivity))
                    self.voice_commands = bool(data.get("voice_commands", self.voice_commands))
                    mode = str(data.get("hotkey_mode", self.hotkey_mode)).lower()
                    if mode not in ("hold", "toggle"):
                        mode = "hold"
                    self.hotkey_mode = mode
                    dev = data.get("input_device_name", None)
                    self.input_device_name = str(dev) if dev else None
                    try:
                        self.input_gain = float(data.get("input_gain", self.input_gain))
                    except (TypeError, ValueError):
                        self.input_gain = 1.0
                    self.input_gain = max(0.1, min(10.0, self.input_gain))
                    self.llm_enabled = bool(data.get("llm_enabled", self.llm_enabled))
                    self.llm_base_url = str(data.get("llm_base_url", self.llm_base_url))
                    self.llm_model = str(data.get("llm_model", self.llm_model))
                    self.llm_api_key = str(data.get("llm_api_key", self.llm_api_key))
                    self.llm_default_prompt = str(data.get("llm_default_prompt", self.llm_default_prompt))
                    self.llm_auto_apply = bool(data.get("llm_auto_apply", self.llm_auto_apply))
                    backend = str(data.get("stt_backend", self.stt_backend)).lower()
                    if backend not in ("local", "elevenlabs", "auto"):
                        backend = "local"
                    self.stt_backend = backend
                    self.eleven_api_key = str(data.get("eleven_api_key", self.eleven_api_key))
                    self.eleven_api_key_backup = str(
                        data.get("eleven_api_key_backup", self.eleven_api_key_backup)
                    )
                    extra = data.get("eleven_api_keys_extra", [])
                    if isinstance(extra, list):
                        self.eleven_api_keys_extra = [
                            str(k).strip() for k in extra if str(k).strip()
                        ]
                    ex = data.get("eleven_exhausted", [])
                    if isinstance(ex, list):
                        self.eleven_exhausted = [
                            str(k).strip() for k in ex if str(k).strip()
                        ]
                    ex_at = data.get("eleven_exhausted_at", {})
                    if isinstance(ex_at, dict):
                        self.eleven_exhausted_at = {}
                        for k, ts in ex_at.items():
                            try:
                                self.eleven_exhausted_at[str(k).strip()] = float(ts)
                            except (TypeError, ValueError):
                                pass
                    for field, target in (
                        ("eleven_retry_after", "eleven_retry_after"),
                        ("eleven_credits_left", "eleven_credits_left"),
                        ("eleven_credits_at", "eleven_credits_at"),
                    ):
                        raw = data.get(field, {})
                        if isinstance(raw, dict):
                            parsed: dict[str, float] = {}
                            for k, v in raw.items():
                                try:
                                    parsed[str(k).strip()] = float(v)
                                except (TypeError, ValueError):
                                    pass
                            setattr(self, target, parsed)
                    self.eleven_model_id = str(
                        data.get("eleven_model_id", self.eleven_model_id)
                    ) or ELEVEN_DEFAULT_MODEL
                    # Settings saved before 2.7.4 carry the old default model
                    # id. Move them to the current one once; after that a
                    # hand-edited id (pinning an older model) is respected.
                    self._eleven_model_migrated = bool(
                        data.get("eleven_model_migrated_v3", False))
                    if (not self._eleven_model_migrated
                            and self.eleven_model_id in ELEVEN_LEGACY_MODELS):
                        self.eleven_model_id = ELEVEN_DEFAULT_MODEL
                    self._eleven_model_migrated = True
                    self.eleven_clean_transcript = bool(data.get(
                        "eleven_clean_transcript", self.eleven_clean_transcript))
                    self.send_after_paste = bool(data.get("send_after_paste", self.send_after_paste))
                    self.terminal_aware_paste = bool(data.get("terminal_aware_paste", self.terminal_aware_paste))
                    self.markdown_save_enabled = bool(data.get("markdown_save_enabled", self.markdown_save_enabled))
                    self.markdown_folder = str(data.get("markdown_folder", self.markdown_folder))
                    self.webhook_enabled = bool(data.get("webhook_enabled", self.webhook_enabled))
                    self.webhook_url = str(data.get("webhook_url", self.webhook_url))
                    self.webhook_template = str(data.get("webhook_template", self.webhook_template))
                    self.save_history = bool(data.get("save_history", self.save_history))
                    try:
                        self.history_retention_days = int(data.get("history_retention_days", self.history_retention_days))
                    except (TypeError, ValueError):
                        self.history_retention_days = 0
                    if self.history_retention_days < 0:
                        self.history_retention_days = 0
                    th = str(data.get("theme", self.theme)).lower()
                    if th not in ("dark", "light"):
                        th = "dark"
                    self.theme = th
                    self.duck_enabled = bool(data.get("duck_enabled", self.duck_enabled))
                    try:
                        self.duck_level_pct = int(data.get("duck_level_pct", self.duck_level_pct))
                    except (TypeError, ValueError):
                        self.duck_level_pct = 5
                    self.duck_level_pct = max(0, min(100, self.duck_level_pct))
                    self.passthrough_enabled = bool(
                        data.get("passthrough_enabled", self.passthrough_enabled)
                    )
                    try:
                        self.passthrough_level_pct = int(
                            data.get("passthrough_level_pct", self.passthrough_level_pct)
                        )
                    except (TypeError, ValueError):
                        self.passthrough_level_pct = 25
                    self.passthrough_level_pct = max(0, min(100, self.passthrough_level_pct))
                    pt_out = data.get("passthrough_output_name", None)
                    self.passthrough_output_name = str(pt_out) if pt_out else None
                    self.passthrough_mute_while_dictating = bool(
                        data.get("passthrough_mute_while_dictating",
                                 self.passthrough_mute_while_dictating)
                    )
                    try:
                        self.keep_recordings = int(
                            data.get("keep_recordings", self.keep_recordings))
                    except (TypeError, ValueError):
                        self.keep_recordings = 0
                    self.keep_recordings = max(0, min(500, self.keep_recordings))
                    # Hotkeys — accept any non-empty string; HotkeyWatcher
                    # validates via keyboard.add_hotkey at register time and
                    # logs (rather than crashes) on bad combos.
                    for name in ("hotkey_quit", "hotkey_repaste",
                                 "hotkey_llm", "hotkey_note"):
                        v = data.get(name)
                        if isinstance(v, str) and v.strip():
                            setattr(self, name, v.strip().lower())
                    self.start_on_login = bool(data.get(
                        "start_on_login", self.start_on_login,
                    ))
                    requested = str(data.get("model_name", self.model_name))
                    valid_names = {m["name"] for m in KNOWN_MODELS}
                    if requested not in valid_names:
                        log_app.warning(
                            "settings.json has unknown model_name %r; "
                            "falling back to %s",
                            requested, DEFAULT_MODEL,
                        )
                        requested = DEFAULT_MODEL
                    self.model_name = requested
                    log_app.info(
                        "loaded settings: auto_paste=%s sensitivity=%.1f "
                        "model=%s voice_commands=%s",
                        self.auto_paste, self.sensitivity, self.model_name,
                        self.voice_commands,
                    )
        except Exception:
            log_app.exception("failed to load settings")

    def save(self) -> None:
        """Persist, reconcile autostart, and broadcast `changed` (live-apply
        of audio/hotkey/theme settings). For state that changes on its own in
        the background — the ElevenLabs key pool — use save_quiet()."""
        self._write()
        # Reconcile the HKCU\...\Run registry entry with the saved setting.
        # Done here (not in __init__) so a manual edit to settings.json picks
        # up only on the next user-driven Settings change, not on startup —
        # but each Settings save makes the registry match what's persisted.
        #
        # ONLY for the real settings file. apply_start_on_login() edits the
        # machine-wide 'Tellur' scheduled task, which is not scoped to
        # self.path — so a Settings built on any other path (tests, a scratch
        # TELLUR_HOME, a one-off script) used to silently disable the user's
        # autostart, because such instances default start_on_login to False.
        # That cost a real autostart outage; a Settings pointed somewhere else
        # must never reach out and touch global OS state.
        if self.path == SETTINGS_FILE:
            apply_start_on_login(self.start_on_login)
        self.changed.emit()

    def save_quiet(self) -> None:
        """Persist only — no autostart reconcile, no `changed` broadcast.

        The ElevenLabs pool persists key cooldowns/balances from the
        transcribe worker thread on every rejected request; routing that
        through save() meant two `schtasks` subprocesses, re-registering
        every chord hotkey and re-resolving audio devices per rejection,
        mid-paste (263 times across the logs). None of those consumers care
        about pool state — the Settings status line polls it on a timer."""
        self._write()

    def _write(self) -> None:
        # RLock: re-entrant from mark_eleven_exhausted(), and it keeps a
        # UI-thread save() from snapshotting the eleven_* dicts while the
        # worker thread is mutating them.
        with self._save_lock:
            self._write_locked()

    def _write_locked(self) -> None:
        data = {
            "auto_paste": self.auto_paste,
            "sensitivity": self.sensitivity,
            "model_name": self.model_name,
            "voice_commands": self.voice_commands,
            "hotkey_mode": self.hotkey_mode,
            "input_device_name": self.input_device_name,
            "input_gain": self.input_gain,
            "llm_enabled": self.llm_enabled,
            "llm_base_url": self.llm_base_url,
            "llm_model": self.llm_model,
            "llm_api_key": self.llm_api_key,
            "llm_default_prompt": self.llm_default_prompt,
            "llm_auto_apply": self.llm_auto_apply,
            "stt_backend": self.stt_backend,
            "eleven_api_key": self.eleven_api_key,
            "eleven_api_key_backup": self.eleven_api_key_backup,
            "eleven_api_keys_extra": list(self.eleven_api_keys_extra),
            "eleven_exhausted": list(self.eleven_exhausted),
            "eleven_exhausted_at": dict(self.eleven_exhausted_at),
            "eleven_retry_after": dict(self.eleven_retry_after),
            "eleven_credits_left": dict(self.eleven_credits_left),
            "eleven_credits_at": dict(self.eleven_credits_at),
            "eleven_model_id": self.eleven_model_id,
            "eleven_model_migrated_v3": True,
            "eleven_clean_transcript": self.eleven_clean_transcript,
            "send_after_paste": self.send_after_paste,
            "terminal_aware_paste": self.terminal_aware_paste,
            "markdown_save_enabled": self.markdown_save_enabled,
            "markdown_folder": self.markdown_folder,
            "webhook_enabled": self.webhook_enabled,
            "webhook_url": self.webhook_url,
            "webhook_template": self.webhook_template,
            "save_history": self.save_history,
            "history_retention_days": self.history_retention_days,
            "theme": self.theme,
            "duck_enabled": self.duck_enabled,
            "duck_level_pct": self.duck_level_pct,
            "passthrough_enabled": self.passthrough_enabled,
            "passthrough_level_pct": self.passthrough_level_pct,
            "passthrough_output_name": self.passthrough_output_name,
            "passthrough_mute_while_dictating": self.passthrough_mute_while_dictating,
            "keep_recordings": self.keep_recordings,
            "hotkey_quit": self.hotkey_quit,
            "hotkey_repaste": self.hotkey_repaste,
            "hotkey_llm": self.hotkey_llm,
            "hotkey_note": self.hotkey_note,
            "start_on_login": self.start_on_login,
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except Exception:
            log_app.exception("failed to save settings")


def _format_size_mb(mb: int) -> str:
    if mb >= 1024:
        return f"{mb / 1024:.1f} GB"
    return f"{mb} MB"


# ---- model cache lookup ---------------------------------------------------
# Maps Tellur's display name → HuggingFace repo id. Pulled from
# faster_whisper.utils._MODELS so it matches whatever faster-whisper would
# actually fetch (notably large-v3-turbo is on mobiuslabsgmbh, not Systran).
def _build_model_repo_map() -> dict[str, str]:
    try:
        from faster_whisper.utils import _MODELS  # type: ignore[attr-defined]
        return {n: _MODELS[n] for n in (m["name"] for m in KNOWN_MODELS) if n in _MODELS}
    except Exception:
        # Hard-coded fallback (correct as of faster-whisper 1.2.x).
        return {
            "tiny":              "Systran/faster-whisper-tiny",
            "small":             "Systran/faster-whisper-small",
            "distil-large-v3":   "Systran/faster-distil-whisper-large-v3",
            "distil-large-v3.5": "distil-whisper/distil-large-v3.5-ct2",
            "large-v3-turbo":    "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
            "large-v3":          "Systran/faster-whisper-large-v3",
        }


_MODEL_REPO: dict[str, str] = _build_model_repo_map()


def _hf_cache_root() -> Path:
    # HF_HUB_CACHE wins (most specific); else HF_HOME with /hub; else default.
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return Path(hub)
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _model_cache_dir(name: str) -> Path:
    repo_id = _MODEL_REPO.get(name, f"Systran/faster-whisper-{name}")
    return _hf_cache_root() / ("models--" + repo_id.replace("/", "--"))


def _dir_size_bytes(p: Path) -> int:
    total = 0
    if not p.exists():
        return 0
    try:
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                pass
    except OSError:
        pass
    return total


def is_model_cached(name: str) -> bool:
    """True iff the HF Hub cache contains both model.bin and config.json for
    this model — i.e. a previous WhisperModel(...) call has fully downloaded
    it and a future call will be near-instant."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        # Fallback heuristic: directory size at least 90% of expected.
        cache_dir = _model_cache_dir(name)
        size = _dir_size_bytes(cache_dir)
        expected = next(
            (m["size_mb"] for m in KNOWN_MODELS if m["name"] == name), 100,
        ) * 1024 * 1024
        return size > expected * 0.9
    repo_id = _MODEL_REPO.get(name, f"Systran/faster-whisper-{name}")
    for fname in ("model.bin", "config.json"):
        result = try_to_load_from_cache(repo_id=repo_id, filename=fname)
        if not isinstance(result, str):
            return False
        try:
            if not Path(result).exists():
                return False
        except OSError:
            return False
    return True


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
def list_input_devices() -> list[dict]:
    """Return a list of input devices as [{'index': int, 'name': str,
    'default': bool, 'channels': int}]. Falls back to an empty list on
    PortAudio errors (e.g. no audio service)."""
    try:
        devices = sd.query_devices()
        try:
            default_in_idx = sd.default.device[0]
        except Exception:
            default_in_idx = None
        out = []
        for i, d in enumerate(devices):
            if d.get("max_input_channels", 0) <= 0:
                continue
            out.append({
                "index": i,
                "name": str(d.get("name", f"Device {i}")),
                "default": (i == default_in_idx),
                "channels": int(d.get("max_input_channels", 0)),
            })
        return out
    except Exception:
        log_audio.exception("failed to enumerate input devices")
        return []


def list_output_devices() -> list[dict]:
    """Playback devices, in the same shape as list_input_devices() plus the host
    API name. Used by the passthrough monitor to pick where the mic is echoed to.

    The same speakers appear once per host API (MME, DirectSound, WASAPI,
    WDM-KS) with near-identical names, so the API is carried through and shown
    in the picker. It is not cosmetic: MME buffers ~80 ms while WASAPI is closer
    to ~15 ms, and for a monitor you hear yourself through that is the
    difference between "present" and "echo"."""
    try:
        devices = sd.query_devices()
        try:
            default_out_idx = sd.default.device[1]
        except Exception:
            default_out_idx = None
        out = []
        for i, d in enumerate(devices):
            if d.get("max_output_channels", 0) <= 0:
                continue
            try:
                api = str(sd.query_hostapis(d["hostapi"])["name"])
            except Exception:
                api = ""
            out.append({
                "index": i,
                "name": str(d.get("name", f"Device {i}")),
                "api": api,
                "default": (i == default_out_idx),
                "channels": int(d.get("max_output_channels", 0)),
            })
        return out
    except Exception:
        log_audio.exception("failed to enumerate output devices")
        return []


def output_device_keys(devices: list[dict]) -> list[str]:
    """Stable identities for a device list: "<API>::<name>", plus an occurrence
    ordinal on any duplicates.

    Every host API enumerates the same hardware under (nearly) the same name, so
    a bare name is ambiguous — matching one would always pick whichever host API
    came first. The API is part of the choice anyway, since it decides the
    monitor's latency. A few devices (the NVIDIA HDMI outputs under WDM-KS) then
    still collide on name *and* API, so the second and later ones get "::2",
    "::3" and so on. Indexes are deliberately not used: they reshuffle between
    sessions, which is why these keys exist at all."""
    seen: dict[str, int] = {}
    keys: list[str] = []
    for d in devices:
        api = (d.get("api") or "").strip()
        base = f"{api}::{d['name']}" if api else d["name"]
        n = seen.get(base, 0) + 1
        seen[base] = n
        keys.append(base if n == 1 else f"{base}::{n}")
    return keys


def find_output_device(key: str | None) -> int | None:
    """Resolve a stored output key back to a PortAudio index, or None for the
    system default. Falls back to a name-only match so keys written before the
    API qualifier existed still resolve."""
    if not key:
        return None
    devices = list_output_devices()
    for d, k in zip(devices, output_device_keys(devices)):
        if k == key:
            return d["index"]
    bare = key.split("::")[1] if "::" in key else key
    for d in devices:
        if d["name"] == bare:
            return d["index"]
    return None


class AudioRecorder:
    """Captures mono float32 audio at SAMPLE_RATE while held. Audio callback is
    kept lean — no logging or heavy computation — so it doesn't stall.

    Supports a selectable input device (index or None for system default) and
    a software gain multiplier applied per-sample. The level reported back to
    the overlay is the post-gain RMS so the meter reflects what's actually
    being fed into Whisper."""

    def __init__(self, sample_rate: int = SAMPLE_RATE):
        self.sample_rate = sample_rate
        self._frames: list[np.ndarray] = []
        self._sample_count = 0
        self._max_samples = int(MAX_RECORDING_SECONDS * sample_rate)
        self._stream: sd.InputStream | None = None
        self._level = 0.0
        # device: PortAudio device index, or None to follow system default.
        # gain: linear multiplier applied per-sample (1.0 == off).
        self._device: int | None = None
        self._gain: float = 1.0
        # Separate stream used by the Settings tab for the live meter; runs
        # only while Settings is visible and never feeds the recorder.
        self._monitor_stream: sd.InputStream | None = None
        self._monitor_level: float = 0.0
        self._monitor_wanted: bool = False
        # PortAudio status flags seen during the current recording (input
        # overflow = the driver dropped samples because our callback was
        # late). Counted in the callback, reported once at stop().
        self._overflows = 0

    def set_device(self, idx: int | None) -> None:
        if idx == self._device:
            return
        self._device = idx
        log_audio.info("input device -> %s", idx if idx is not None else "system default")
        # If a live monitor is running, restart it on the new device.
        if self._monitor_stream is not None:
            self.stop_monitor()
            self.start_monitor()

    def set_gain(self, gain: float) -> None:
        g = max(0.1, min(10.0, float(gain)))
        if abs(g - self._gain) < 1e-6:
            return
        self._gain = g
        log_audio.info("input gain -> %.2fx", g)

    def _callback(self, indata, _frames, _time_info, status) -> None:
        if status:
            self._overflows += 1
        chunk = indata.copy().reshape(-1).astype(np.float32)
        if self._gain != 1.0:
            np.multiply(chunk, self._gain, out=chunk)
            np.clip(chunk, -1.0, 1.0, out=chunk)
        # Hard cap to prevent unbounded RAM growth on a stuck-down hotkey.
        if self._sample_count + chunk.size > self._max_samples:
            return  # silently drop further audio
        self._frames.append(chunk)
        self._sample_count += chunk.size
        self._level = float(np.sqrt(np.mean(chunk * chunk)))

    def start(self) -> None:
        self._frames = []
        self._sample_count = 0
        self._level = 0.0
        self._overflows = 0
        # Tear down the monitor stream so the device is exclusively available
        # for the recording — important for drivers that don't support shared
        # mode. `_monitor_wanted` (set by the panel's start/stop_monitor)
        # tells stop() whether to bring it back.
        if self._monitor_stream is not None:
            self.stop_monitor(_internal=True)
        stream = None
        try:
            stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._callback,
                blocksize=0,
                device=self._device,
            )
            stream.start()
        except Exception:
            # Never leave a half-opened PortAudio stream behind, and give the
            # Settings meter its stream back — the old code left the meter
            # dead after one failed start (unplugged mic) until the panel was
            # hidden and re-shown.
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    pass
            self._stream = None
            if self._monitor_wanted:
                self.start_monitor(_internal=True)
            raise
        self._stream = stream
        log_audio.debug("recording started (device=%s gain=%.2f)",
                        self._device, self._gain)

    def stop(self) -> np.ndarray:
        if self._stream is None:
            # Nothing recording — but the panel may have asked for its meter
            # while we were busy; honour that now.
            if self._monitor_wanted and self._monitor_stream is None:
                self.start_monitor(_internal=True)
            return np.zeros(0, dtype=np.float32)
        try:
            self._stream.stop()
            self._stream.close()
        finally:
            self._stream = None
        # Bring the monitor back if the panel still wants it.
        if self._monitor_wanted:
            self.start_monitor(_internal=True)
        if not self._frames:
            log_audio.debug("recording stopped — no frames captured")
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(self._frames).astype(np.float32)
        rms  = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        log_audio.info("recording stopped — %d samples (%.2fs) RMS=%.5f peak=%.5f",
                       audio.size, audio.size / self.sample_rate, rms, peak)
        if self._overflows:
            log_audio.warning(
                "audio driver reported %d dropped/late buffer(s) during this "
                "recording — words may be missing", self._overflows,
            )
        return audio

    @property
    def level(self) -> float:
        return self._level

    # --- monitor stream (used by Settings tab live meter) ----------------

    def _monitor_callback(self, indata, _frames, _time_info, _status) -> None:
        chunk = indata.copy().reshape(-1).astype(np.float32)
        if self._gain != 1.0:
            np.multiply(chunk, self._gain, out=chunk)
            np.clip(chunk, -1.0, 1.0, out=chunk)
        self._monitor_level = float(np.sqrt(np.mean(chunk * chunk)))

    def start_monitor(self, *, _internal: bool = False) -> bool:
        """Start a passive monitor stream for the Settings meter. Returns
        True on success, False if the stream couldn't be opened (e.g. the
        recorder is currently active or the device is unavailable). A call
        made while recording is remembered: the stream comes up when the
        recording stops. `_internal` marks the recorder's own start/stop
        juggling, which must not change what the panel asked for."""
        if not _internal:
            self._monitor_wanted = True
        if self._stream is not None:
            return False
        if self._monitor_stream is not None:
            return True
        try:
            self._monitor_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                callback=self._monitor_callback,
                blocksize=0,
                device=self._device,
            )
            self._monitor_stream.start()
            return True
        except Exception:
            log_audio.exception("monitor stream failed to start")
            self._monitor_stream = None
            return False

    def stop_monitor(self, *, _internal: bool = False) -> None:
        if not _internal:
            self._monitor_wanted = False
        if self._monitor_stream is None:
            return
        try:
            self._monitor_stream.stop()
            self._monitor_stream.close()
        except Exception:
            log_audio.debug("monitor stream stop failed", exc_info=True)
        finally:
            self._monitor_stream = None
            self._monitor_level = 0.0

    @property
    def monitor_level(self) -> float:
        return self._monitor_level


# ===========================================================================
# mic passthrough — echo the mic into the speakers so the room stays audible
# ===========================================================================
class PassthroughMonitor:
    """Live mic -> speakers monitor, with a volume independent of the Windows
    mic level.

    Windows can already do this ("Listen to this device") with no process at
    all, but its only volume control is the mic's capture level — turning the
    passthrough down would also make you quieter to everyone in Discord. Mixing
    it here costs a running process (Tellur is already one) and buys a level
    that is genuinely separate.

    Input and output are two independent streams joined by a bounded deque
    rather than one duplex stream: they are usually different devices with
    free-running clocks, which PortAudio will not open as a duplex pair. The
    deque's maxlen is the latency ceiling — if the output side ever falls
    behind, the oldest blocks are dropped instead of letting the delay grow.

    Gain and mute are read live inside the callbacks, so changing the level or
    ducking for a dictation never restarts a stream.
    """

    def __init__(self) -> None:
        self._input_device: int | None = None
        self._output_device: int | None = None
        self._gain = 0.0
        self._muted = False
        self._level = 0.0
        self._rate = PASSTHROUGH_SAMPLE_RATE
        self._queue: deque = deque(maxlen=PASSTHROUGH_QUEUE_BLOCKS)
        self._in = None
        self._out = None
        self._lock = threading.RLock()

    # --- configuration ---------------------------------------------------

    @staticmethod
    def level_to_gain(pct: int) -> float:
        """0 % is silent, 100 % is the mic at full level. Squared so the useful
        quiet end of the range gets most of the slider travel."""
        pct = max(0, min(100, int(pct)))
        return 0.0 if pct == 0 else (pct / 100.0) ** 2

    def set_level_pct(self, pct: int) -> None:
        self._gain = self.level_to_gain(pct)

    def set_muted(self, muted: bool) -> None:
        """Silence the monitor without tearing the streams down, so unmuting is
        instant and never fights the recorder for the device."""
        self._muted = bool(muted)

    def set_devices(self, input_device, output_device) -> None:
        """Repoint the monitor. Restarts the streams only if something actually
        changed and it is currently running."""
        if input_device == self._input_device and output_device == self._output_device:
            return
        self._input_device = input_device
        self._output_device = output_device
        log_audio.info("passthrough devices -> in=%s out=%s",
                       input_device if input_device is not None else "default",
                       output_device if output_device is not None else "default")
        with self._lock:
            if self.running:
                self.stop()
                self.start()

    @property
    def running(self) -> bool:
        return self._in is not None and self._out is not None

    @property
    def level(self) -> float:
        return self._level

    # --- audio callbacks -------------------------------------------------

    def _in_cb(self, indata, _frames, _time_info, _status) -> None:
        gain = 0.0 if self._muted else self._gain
        if gain <= 0.0:
            self._level = 0.0
            return  # queue starves, output plays silence — a free mute
        chunk = indata[:, :1].astype(np.float32) * gain
        np.clip(chunk, -1.0, 1.0, out=chunk)
        self._queue.append(chunk)
        self._level = float(np.sqrt(np.mean(chunk * chunk)))

    def _out_cb(self, outdata, frames, _time_info, _status) -> None:
        try:
            block = self._queue.popleft()
        except IndexError:
            outdata.fill(0.0)
            return
        n = min(len(block), frames)
        outdata[:n] = block[:n]      # mono column broadcasts across channels
        if n < frames:
            outdata[n:] = 0.0

    # --- lifecycle -------------------------------------------------------

    def _output_channels(self) -> int:
        try:
            info = sd.query_devices(self._output_device, "output")
            return max(1, min(2, int(info.get("max_output_channels", 2))))
        except Exception:
            return 2

    def start(self) -> bool:
        """Open both streams. Returns False (and logs) if the devices can't be
        opened, so the caller can turn the setting back off."""
        with self._lock:
            if self.running:
                return True
            channels = self._output_channels()
            for rate in PASSTHROUGH_RATE_FALLBACKS:
                try:
                    self._out = sd.OutputStream(
                        samplerate=rate, channels=channels, dtype="float32",
                        callback=self._out_cb, blocksize=PASSTHROUGH_BLOCK,
                        device=self._output_device, latency="low",
                    )
                    self._in = sd.InputStream(
                        samplerate=rate, channels=1, dtype="float32",
                        callback=self._in_cb, blocksize=PASSTHROUGH_BLOCK,
                        device=self._input_device, latency="low",
                    )
                    self._out.start()
                    self._in.start()
                    self._rate = rate
                    log_audio.info(
                        "passthrough started (%d Hz, %d ch, %.1f ms blocks)",
                        rate, channels, PASSTHROUGH_BLOCK * 1000.0 / rate,
                    )
                    return True
                except Exception:
                    log_audio.debug("passthrough failed at %d Hz", rate, exc_info=True)
                    self._teardown()
            log_audio.warning("passthrough could not open the selected devices")
            return False

    def stop(self) -> None:
        with self._lock:
            if self.running:
                log_audio.info("passthrough stopped")
            self._teardown()

    def _teardown(self) -> None:
        for stream in (self._in, self._out):
            if stream is None:
                continue
            try:
                stream.stop()
                stream.close()
            except Exception:
                log_audio.debug("passthrough stream teardown failed", exc_info=True)
        self._in = None
        self._out = None
        self._queue.clear()
        self._level = 0.0


# ===========================================================================
# audio ducking — lower per-app volume while recording so background
# audio (videos, music, game sounds) doesn't distract the speaker
# ===========================================================================
class AudioDucker:
    """Snapshot every per-app audio session's master volume, set them to a
    low level while recording, then restore on release.

    Uses pycaw (Core Audio APIs via comtypes) — Windows-only. Imports are
    deferred so a missing pycaw doesn't break startup; `available()` reports
    whether the feature is usable.

    Skips our own process (so a future Tellur sound — e.g. confirm chime —
    wouldn't be ducked under itself) and is idempotent: a double-duck is a
    no-op, a restore-without-duck is a no-op.
    """

    def __init__(self, ledger_path: Path | None = None):
        self._our_pid = os.getpid()
        self._lock = threading.Lock()
        self._active = False
        # In-memory fast path for the normal duck -> restore cycle.
        self._saved: list[tuple[object, float]] = []
        # Crash-safety ledger: pre-duck levels are persisted here so a kill /
        # hang / crash mid-duck can be undone by the next launch (recover()).
        self._ledger_path = ledger_path or (DATA_DIR / "duck_state.json")
        self._watchdog: threading.Timer | None = None
        # Backstop: auto-restore if a release somehow never arrives in one run.
        self._max_active_sec = float(MAX_RECORDING_SECONDS) + 10.0

    @staticmethod
    def available() -> bool:
        try:
            from pycaw.pycaw import AudioUtilities  # noqa: F401
            return True
        except Exception:
            return False

    def duck(self, level: float = 0.05) -> None:
        """Lower every other app's master volume to `level` (0.0 to 1.0),
        recording prior levels both in memory and in the on-disk ledger for
        crash-safe restore. Safe to call from the hotkey thread."""
        level = max(0.0, min(1.0, float(level)))
        with self._lock:
            if self._active:
                return
            try:
                from pycaw.pycaw import AudioUtilities
                sessions = AudioUtilities.GetAllSessions()
            except Exception:
                log_audio.exception("ducker: failed to enumerate audio sessions")
                return

            # Pass 1: read & capture originals WITHOUT changing anything yet, so
            # a crash before the ledger is persisted can't strand a ducked
            # session that has no recovery record.
            captured: list[tuple] = []  # (vol_com, prev, iid, pid, name)
            for s in sessions:
                try:
                    if s.Process and s.Process.pid == self._our_pid:
                        continue
                    vol = s.SimpleAudioVolume
                    prev = vol.GetMasterVolume()
                    # Skip sessions already at/below target — nothing to undo,
                    # and we must never capture a (possibly stuck) low value as
                    # an "original" to restore to.
                    if prev <= level:
                        continue
                    pid = s.Process.pid if s.Process else None
                    try:
                        name = s.Process.name() if s.Process else None
                    except Exception:
                        name = None
                    captured.append((vol, prev, self._session_iid(s), pid, name))
                except Exception:
                    log_audio.debug("ducker: per-session read failed", exc_info=True)

            # Pass 2: persist the ledger BEFORE touching any volume (crash safety).
            self._write_ledger(level, captured)

            # Pass 3: apply the duck.
            saved: list[tuple[object, float]] = []
            for vol, prev, _iid, _pid, _name in captured:
                try:
                    vol.SetMasterVolume(level, None)
                    saved.append((vol, prev))
                except Exception:
                    log_audio.debug("ducker: per-session duck failed", exc_info=True)

            self._saved = saved
            self._active = True
            self._arm_watchdog()
            log_audio.info(
                "ducked %d audio sessions to %.0f%%", len(saved), level * 100
            )

    def restore(self) -> None:
        """Restore every ducked session to its pre-duck level, then clear the
        ledger. Idempotent."""
        with self._lock:
            self._cancel_watchdog()
            if not self._active:
                self._delete_ledger()   # clear any stale ledger, just in case
                return
            for vol, prev in self._saved:
                try:
                    vol.SetMasterVolume(prev, None)
                except Exception:
                    log_audio.debug("ducker: per-session restore failed", exc_info=True)
            count = len(self._saved)
            self._saved = []
            self._active = False
            self._delete_ledger()
            log_audio.info("restored volume on %d sessions", count)

    @staticmethod
    def _session_iid(s) -> str | None:
        # Stable per-session id; unchanged across a Tellur restart while the
        # other app's session lives. The key we match on during recovery.
        try:
            return s._ctl.GetSessionInstanceIdentifier()
        except Exception:
            return None

    def recover(self) -> int:
        """Run once at startup. If a ledger exists, a previous run was killed /
        hung / crashed mid-duck: restore each recorded session to its original
        level — but only if it STILL looks ducked (current <= recorded level +
        eps), so an app the user already turned back up is left untouched.
        Returns the number of sessions restored."""
        path = self._ledger_path
        try:
            if not path.exists():
                return 0
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            log_audio.debug("ducker: unreadable ledger — discarding", exc_info=True)
            self._delete_ledger()
            return 0

        level = max(0.0, min(1.0, float(data.get("level", 0.05))))
        eps = 0.02
        by_iid: dict = {}
        by_pid: dict = {}
        for e in data.get("sessions", []):
            orig = e.get("original")
            if orig is None:
                continue
            orig = max(0.0, min(1.0, float(orig)))
            if e.get("iid"):
                by_iid[e["iid"]] = orig
            if e.get("pid") is not None:
                by_pid.setdefault(e["pid"], orig)

        try:
            from pycaw.pycaw import AudioUtilities
            sessions = AudioUtilities.GetAllSessions()
        except Exception:
            log_audio.exception("ducker: recovery could not enumerate sessions")
            return 0  # keep the ledger; a later launch can retry

        restored = 0
        for s in sessions:
            try:
                iid = self._session_iid(s)
                pid = s.Process.pid if s.Process else None
                target = by_iid.get(iid) if iid is not None else None
                if target is None and pid is not None:
                    target = by_pid.get(pid)
                if target is None:
                    continue
                vol = s.SimpleAudioVolume
                if vol.GetMasterVolume() <= level + eps:
                    vol.SetMasterVolume(target, None)
                    restored += 1
            except Exception:
                log_audio.debug("ducker: per-session recovery failed", exc_info=True)

        self._delete_ledger()
        if restored:
            log_audio.info(
                "ducker: recovered %d session(s) left quiet by a previous unclean exit",
                restored,
            )
        return restored

    # --- watchdog: auto-restore if a release never arrives within one run ---
    def _arm_watchdog(self) -> None:
        self._cancel_watchdog()
        try:
            t = threading.Timer(self._max_active_sec, self._on_watchdog)
            t.daemon = True
            t.start()
            self._watchdog = t
        except Exception:
            self._watchdog = None

    def _cancel_watchdog(self) -> None:
        t, self._watchdog = self._watchdog, None
        if t is not None:
            try:
                t.cancel()
            except Exception:
                pass

    def _on_watchdog(self) -> None:
        log_audio.warning("ducker: watchdog fired — auto-restoring (no release seen)")
        self.restore()

    # --- ledger persistence -------------------------------------------------
    def _write_ledger(self, level: float, captured: list) -> None:
        try:
            payload = {
                "level": level,
                "ts": time.time(),
                "pid": self._our_pid,
                "sessions": [
                    {"iid": iid, "pid": pid, "name": name, "original": prev}
                    for (_vol, prev, iid, pid, name) in captured
                ],
            }
            self._ledger_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._ledger_path.with_suffix(self._ledger_path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            tmp.replace(self._ledger_path)
        except Exception:
            log_audio.debug("ducker: failed to write ledger", exc_info=True)

    def _delete_ledger(self) -> None:
        try:
            self._ledger_path.unlink(missing_ok=True)
        except Exception:
            log_audio.debug("ducker: failed to delete ledger", exc_info=True)


# ===========================================================================
# whisper engine
# ===========================================================================
# ===========================================================================
# Cloud transcription backend (optional) — ElevenLabs Scribe Speech-to-Text.
# Opt-in alternative to the on-device Whisper engine. Uses only stdlib urllib,
# matching the project's no-extra-deps style. The recorded mono 16 kHz clip is
# packed into a WAV and POSTed to /v1/speech-to-text; we return the text.
# NOTE: this sends audio to ElevenLabs' servers — it is NOT offline. Tellur
# stays fully local by default; this path runs only when the user opts in.
# ===========================================================================
ELEVEN_STT_URL = "https://api.elevenlabs.io/v1/speech-to-text"
# Scribe v3 (released 2026-11-10) replaced v2 as the batch model: same
# endpoint, same request, same credit cost and — measured on this app's own
# recordings — the same latency, with ~27% fewer word errors per ElevenLabs.
ELEVEN_DEFAULT_MODEL = "scribe_v3"
# Model ids that were once this app's default; a saved one is upgraded once.
ELEVEN_LEGACY_MODELS = frozenset({"scribe_v1", "scribe_v2"})
# How long an exhausted key stays benched before the pool probes it again.
# Quota errors clear on ElevenLabs' monthly reset, so without a retry an
# exhausted key stayed dead forever unless the user re-typed it in Settings.
ELEVEN_EXHAUSTED_RETRY_SEC = 6 * 3600
# A 429 is back-pressure, not a spent quota — bench the key for a minute, not
# six hours. (v2.3.4 treated both the same, so one burst of dictation could
# take a perfectly good key offline for the rest of the evening.)
ELEVEN_RATELIMIT_RETRY_SEC = 60.0
# How long a reported credit balance is trusted before we go back to just
# trying the key. Balances are invalidated by the monthly quota reset.
ELEVEN_CREDITS_TTL_SEC = 30 * 60
# Observed Scribe billing: 133 credits for a 120 s clip => ~1.108 credits/sec.
# Used only to skip a key we KNOW cannot cover the clip; when the estimate is
# wrong the API simply rejects the request, which costs no credits.
ELEVEN_CREDITS_PER_SEC = 1.11
# Time budget for one cloud transcription, scaled to the clip. Scribe answers
# in ~0.6 s + 0.022 s per second of audio (1,127 clips, Sep-Nov 2026; the
# slowest healthy one was 9.5 s for a 5-minute clip) — but now and then a
# request simply stalls: twice on 2026-11-12 a short clip sat for 52 s and
# 61 s behind the old flat 60 s timeout, with the next dictation queued up
# behind it. So:
#   * past the HEDGE mark (~1% of healthy requests get this far) a second
#     request is raced against the first on a fresh connection;
#   * past the DEADLINE the cloud is abandoned for this clip and local
#     Whisper decodes it instead.
# A 10 s clip hedges at 2.9 s and gives up at 6.8 s; a 5-minute one at 14.5 s
# and 30 s.
ELEVEN_HEDGE_BASE_SEC = 2.5
ELEVEN_HEDGE_PER_AUDIO_SEC = 0.04
ELEVEN_DEADLINE_BASE_SEC = 6.0
ELEVEN_DEADLINE_PER_AUDIO_SEC = 0.08


def eleven_time_budget(clip_sec: float) -> tuple[float, float]:
    """(hedge_after, deadline) in seconds for a clip of `clip_sec`."""
    clip_sec = max(0.0, float(clip_sec))
    return (
        ELEVEN_HEDGE_BASE_SEC + ELEVEN_HEDGE_PER_AUDIO_SEC * clip_sec,
        ELEVEN_DEADLINE_BASE_SEC + ELEVEN_DEADLINE_PER_AUDIO_SEC * clip_sec,
    )


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Pack a mono float32 [-1, 1] array into 16-bit PCM WAV bytes (stdlib)."""
    import wave
    pcm = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm16.tobytes())
    return buf.getvalue()


def save_debug_recording(audio: np.ndarray, keep: int,
                         sample_rate: int = SAMPLE_RATE) -> None:
    """Write `audio` to DEBUG_RECORDINGS_DIR and prune to the newest `keep`.

    Runs on a daemon thread so the paste path never waits on disk. File names
    carry sample count and RMS (rec_YYYYmmdd_HHMMSS_<n>smp_rms<x>.wav) so a
    clip can be matched to its "recording stopped" log line.
    """
    if keep <= 0 or audio.size < sample_rate:   # ignore sub-second blips
        return
    rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else 0.0
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"rec_{stamp}_{audio.size}smp_rms{rms:.4f}.wav"
    data = _audio_to_wav_bytes(audio, sample_rate)

    def _write() -> None:
        try:
            DEBUG_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
            (DEBUG_RECORDINGS_DIR / name).write_bytes(data)
            files = sorted(DEBUG_RECORDINGS_DIR.glob("rec_*.wav"))
            for old in files[:-keep]:
                try:
                    old.unlink()
                except OSError:
                    pass
        except Exception:
            log_audio.exception("debug recording save failed")

    threading.Thread(target=_write, daemon=True, name="debug-rec-save").start()


def _encode_multipart(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    file_content_type: str = "application/octet-stream",
) -> tuple[bytes, str]:
    """Build a multipart/form-data body with stdlib only.
    Returns (body_bytes, content_type_header)."""
    boundary = "----TellurForm" + os.urandom(16).hex()
    b = boundary.encode()
    crlf = b"\r\n"
    parts: list[bytes] = []
    for name, value in fields.items():
        parts += [
            b"--" + b,
            f'Content-Disposition: form-data; name="{name}"'.encode(),
            b"",
            str(value).encode("utf-8"),
        ]
    parts += [
        b"--" + b,
        f'Content-Disposition: form-data; name="{file_field}"; filename="{filename}"'.encode(),
        f"Content-Type: {file_content_type}".encode(),
        b"",
        file_bytes,
        b"--" + b + b"--",
        b"",
    ]
    return crlf.join(parts), f"multipart/form-data; boundary={boundary}"


class ElevenAuthError(Exception):
    """ElevenLabs returned 401/402/429 for a key — invalid key, out of credits,
    or rate-limited. The Scribe failover treats this as 'skip this key and try
    the next one', and puts the key on a short cooldown."""

    def __init__(self, code: int, status: str = "", message: str = ""):
        self.code = code
        self.status = status          # e.g. "quota_exceeded", "detected_unusual_activity"
        super().__init__(message or f"HTTP {code} {status}".strip())


class ElevenAllKeysFailed(Exception):
    """Every configured ElevenLabs key failed (exhausted / invalid / on cooldown)."""


class ElevenTimeout(ElevenAllKeysFailed):
    """The cloud did not answer inside the clip's time budget (see
    eleven_time_budget). Says nothing about the keys — the caller falls back
    to local Whisper for this one clip and tries the cloud again next time."""


def _eleven_error_detail(err: "urllib.error.HTTPError") -> tuple[str, str]:
    """Extract ElevenLabs' {"detail": {"status": ..., "message": ...}} from an
    error response body. Returns (status, message); best-effort, never raises."""
    try:
        body = err.read().decode("utf-8", errors="replace")
    except Exception:
        return "", ""
    try:
        detail = (json.loads(body) or {}).get("detail")
    except Exception:
        return "", body[:200].strip()
    if isinstance(detail, dict):
        return str(detail.get("status") or ""), str(detail.get("message") or "")
    if isinstance(detail, str):
        return "", detail
    return "", body[:200].strip()


def _eleven_credits_remaining(message: str) -> float | None:
    """Pull the balance out of a quota rejection, or None if it isn't there.

    ElevenLabs spells it out for us:
      "This request exceeds your quota of 10000. You have 20 credits
       remaining, while 133 credits are required for this request."
    A key reporting >0 remaining is only out of room for THIS clip, so it gets
    a zero-length cooldown and stays eligible for shorter ones."""
    if not message:
        return None
    m = re.search(r"have\s+([\d,]+(?:\.\d+)?)\s+credits?\s+remaining", message, re.I)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


class ElevenLabsScribeClient:
    """ElevenLabs Speech-to-Text (Scribe) over stdlib urllib.

    Batch-transcribes a recorded clip via POST /v1/speech-to-text. Auth is the
    `xi-api-key` header (NOT Bearer). Returns the transcript text. Synchronous —
    call from a worker thread so the Qt event loop never blocks."""

    def __init__(
        self,
        api_key: str,
        model_id: str = ELEVEN_DEFAULT_MODEL,
        timeout: float = 60.0,
        clean: bool = True,
    ):
        self.api_key = api_key
        self.model_id = model_id or ELEVEN_DEFAULT_MODEL
        self.timeout = timeout
        self.clean = clean

    def transcribe(self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
        return self.transcribe_wav(_audio_to_wav_bytes(audio, sample_rate))

    def _fields(self) -> dict[str, str]:
        """Form fields for the request. Scribe v3 is far more literal than v2
        about disfluencies ("I want you to l- to as-- act like…"), so
        `no_verbatim` is what keeps dictation readable; it in turn spells
        numbers out ("version two"), which `numbers_format` puts back. None of
        these change what a request costs (checked against the credit quote
        in a quota rejection)."""
        fields = {"model_id": self.model_id, "tag_audio_events": "false"}
        if self.clean:
            fields["no_verbatim"] = "true"
        if not self.model_id.startswith(("scribe_v1", "scribe_v2")):
            fields["numbers_format"] = "digits"      # v3+ only
        return fields

    def transcribe_wav(self, wav: bytes) -> str:
        """POST an already-encoded WAV. Split out so the pool can encode a
        clip once and hand the same bytes to a raced second request."""
        body, content_type = _encode_multipart(
            fields=self._fields(),
            file_field="file",
            filename="audio.wav",
            file_bytes=wav,
            file_content_type="audio/wav",
        )
        headers = {
            "xi-api-key": self.api_key,
            "Content-Type": content_type,
            "User-Agent": f"Tellur/{__version__}",
        }
        req = urllib.request.Request(
            ELEVEN_STT_URL, data=body, headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code in (401, 402, 429):
                status, message = _eleven_error_detail(e)
                raise ElevenAuthError(e.code, status, message) from None
            raise
        data = json.loads(raw)
        # Scribe responds with {"text": "...", "language_code": ..., "words": [...]}.
        return (data.get("text") or "").strip()


class ElevenLabsScribePool:
    """Persistent failover across the configured ElevenLabs keys.

    The active key is the first Settings.eleven_keys() entry that is NOT marked
    exhausted. When a key returns 401/402/429 (out of credits / invalid / rate-
    limited) it's recorded in Settings.eleven_exhausted and persisted to disk, so
    the switch survives restarts and cascades forward as each key is used up.
    Exhausted keys are re-probed after ELEVEN_EXHAUSTED_RETRY_SEC (quotas reset
    monthly); a successful probe re-enables the key, a failed one re-arms the
    cooldown — so a refilled key comes back on its own within one dictation.
    Transient errors (network / 5xx) move to the next key for THIS request only,
    without marking anything exhausted. Reads keys + exhausted state live from
    Settings on every call, so it never goes stale when the user edits keys.
    Raises ElevenAllKeysFailed when no usable key remains.

    Every clip also runs against a time budget (eleven_time_budget): a request
    that is slow to answer gets a second one raced against it, and past the
    deadline ElevenTimeout is raised so the caller can decode locally instead
    of leaving the user staring at a "transcribing" pill for a minute."""

    def __init__(self, settings: "Settings", timeout: float | None = None):
        self._settings = settings
        # None → the deadline scales with the clip. A number pins it (tests).
        self.timeout = timeout

    def transcribe(
        self, audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
    ) -> tuple[str, str, int]:
        """Returns (text, key_used, index_in_eleven_keys)."""
        if not self._settings.eleven_keys():
            raise ElevenAllKeysFailed("no ElevenLabs API key configured")
        clip_sec = audio.size / float(sample_rate or SAMPLE_RATE)
        hedge_after, deadline = eleven_time_budget(clip_sec)
        if self.timeout is not None:
            deadline = float(self.timeout)
            hedge_after = min(hedge_after, deadline / 2.0)
        wav = _audio_to_wav_bytes(audio, sample_rate)
        results: "queue.Queue[tuple[str, bool, object]]" = queue.Queue()
        # Set once this call has returned (answer in hand, or given up): a
        # request still in flight then discards whatever it eventually gets.
        abandoned = threading.Event()

        def run(tag: str) -> None:
            try:
                results.put((tag, True, self._run_chain(
                    wav, clip_sec, deadline, abandoned)))
            except BaseException as e:   # noqa: BLE001 — relayed to the waiter
                results.put((tag, False, e))

        def launch(tag: str) -> None:
            threading.Thread(
                target=run, args=(tag,), daemon=True, name=f"scribe-{tag}",
            ).start()

        t0 = time.monotonic()
        launch("first")
        in_flight, hedged = 1, False
        last_err: BaseException | None = None
        try:
            while in_flight:
                limit = deadline if hedged else hedge_after
                try:
                    tag, ok, value = results.get(
                        timeout=max(0.0, limit - (time.monotonic() - t0)))
                except queue.Empty:
                    if not hedged:
                        hedged = True
                        log_engine.warning(
                            "ElevenLabs slow: no answer %.1fs into a %.1fs clip "
                            "— racing a second request (gives up at %.1fs)",
                            time.monotonic() - t0, clip_sec, deadline,
                        )
                        launch("hedge")
                        in_flight += 1
                        continue
                    raise ElevenTimeout(
                        "no answer within %.1fs for a %.1fs clip"
                        % (deadline, clip_sec)
                    ) from None
                in_flight -= 1
                if ok:
                    if hedged:
                        log_engine.info(
                            "ElevenLabs %s request answered after %.1fs",
                            "second (hedge)" if tag == "hedge" else "original",
                            time.monotonic() - t0,
                        )
                    return value   # type: ignore[return-value]
                # That request failed outright (keys spent, network down). If
                # its twin is still out, its answer can still save the clip.
                last_err = value   # type: ignore[assignment]
            assert last_err is not None
            raise last_err
        finally:
            abandoned.set()

    def _run_chain(
        self, wav: bytes, clip_sec: float, sock_timeout: float,
        abandoned: threading.Event,
    ) -> tuple[str, str, int]:
        """One pass over the keys: first usable key that answers wins.
        Returns (text, key_used, index_in_eleven_keys) or raises
        ElevenAllKeysFailed. Runs on a worker thread; `abandoned` means the
        caller has stopped waiting, so stop spending requests."""
        s = self._settings
        keys = s.eleven_keys()
        if not keys:
            raise ElevenAllKeysFailed("no ElevenLabs API key configured")
        model_id = s.eleven_model_id or ELEVEN_DEFAULT_MODEL
        # Only keys still inside the retry cooldown are skipped; ones past it
        # get probed again below and re-enabled if the quota came back.
        exhausted = s.eleven_exhausted_active()
        marked = set(s.eleven_exhausted)
        last_err: Exception | None = None
        skipped = 0
        need_credits = max(1.0, clip_sec * ELEVEN_CREDITS_PER_SEC)
        for i, key in enumerate(keys):
            if abandoned.is_set():
                break
            if key in exhausted:
                skipped += 1
                continue
            # A key we know is nearly empty can still take a short clip. Skip it
            # only when the balance we were told about cannot possibly cover
            # this one — that saves a guaranteed-failed round trip without ever
            # writing off capacity we could have used.
            known = s.eleven_credits_known(key)
            if known is not None and known < need_credits:
                skipped += 1
                log_engine.debug(
                    "ElevenLabs key #%d (…%s) skipped: ~%.0f credits needed for "
                    "%.1fs, only %.0f known remaining.",
                    i + 1, key[-4:], need_credits, clip_sec, known,
                )
                continue
            try:
                # The socket timeout only reaps a request the waiter has long
                # since given up on; the waiter enforces the real deadline.
                text = ElevenLabsScribeClient(
                    key, model_id=model_id, timeout=sock_timeout,
                    clean=bool(getattr(s, "eleven_clean_transcript", True)),
                ).transcribe_wav(wav)
                if abandoned.is_set():
                    log_engine.debug(
                        "ElevenLabs key #%d (…%s) answered after the clip "
                        "was already served; discarded.", i + 1, key[-4:],
                    )
                    return text, key, i
                if key in marked:
                    s.clear_eleven_exhausted(key)
                    log_engine.info(
                        "ElevenLabs key #%d (…%s) works again after cooldown; "
                        "re-enabled.", i + 1, key[-4:],
                    )
                return text, key, i
            except ElevenAuthError as e:
                last_err = e
                left = _eleven_credits_remaining(str(e))
                if e.code == 429:
                    # Back-pressure, not exhaustion: brief cooldown only.
                    cooldown: float | None = ELEVEN_RATELIMIT_RETRY_SEC
                    why = "rate-limited"
                elif left is not None and left > 0:
                    # Has credits, just not enough for this clip. Stay eligible
                    # for shorter ones; the recorded balance does the gating.
                    cooldown = 0.0
                    why = "%.0f credits left, ~%.0f needed" % (left, need_credits)
                else:
                    cooldown = None                  # default long cooldown
                    why = "out of credits/invalid"
                s.mark_eleven_exhausted(              # persisted → survives restart
                    key, retry_after_sec=cooldown, credits_left=left,
                )
                if cooldown != 0.0:
                    exhausted.add(key)
                log_engine.warning(
                    "ElevenLabs key #%d (…%s) %s (HTTP %s %s); "
                    "cooldown %s, switching to the next key.",
                    i + 1, key[-4:], why, e.code, e.status or "",
                    "default" if cooldown is None else ("%.0fs" % cooldown),
                )
            except urllib.error.HTTPError as e:        # non-quota HTTP status
                last_err = e
                if abandoned.is_set():
                    break
                if 400 <= e.code < 500:
                    # 400/413/422…: the request itself is bad for THIS clip
                    # (too long, malformed). Every key would reject it the
                    # same way — don't re-upload it N times.
                    log_engine.warning(
                        "ElevenLabs key #%d (…%s) rejected the request (HTTP %s); "
                        "not retrying other keys.", i + 1, key[-4:], e.code,
                    )
                    break
                log_engine.warning(
                    "ElevenLabs key #%d (…%s) server error (HTTP %s); trying next key.",
                    i + 1, key[-4:], e.code,
                )
            except urllib.error.URLError as e:          # DNS / connect / TLS / timeout
                last_err = e
                if abandoned.is_set():
                    break
                # Connection-level failures are about the network, not the
                # key: the next key would hit the same wall. One try, then
                # fall back to local for this clip.
                log_engine.warning(
                    "ElevenLabs key #%d (…%s) connection error (%s); "
                    "not retrying other keys.", i + 1, key[-4:], e.reason,
                )
                break
            except Exception as e:                     # parse / unexpected — transient
                last_err = e
                if abandoned.is_set():
                    break
                log_engine.warning(
                    "ElevenLabs key #%d (…%s) transient error (%s); trying next key.",
                    i + 1, key[-4:], e,
                )
        raise ElevenAllKeysFailed(
            f"all {len(keys)} ElevenLabs key(s) unavailable"
            + (f" ({skipped} already exhausted)" if skipped else "")
        ) from last_err


class GpuLoadMonitor:
    """Answers one question for the auto backend: is the GPU too busy for a
    local decode right now?

    NVML (the nvidia-smi library) is queried lazily and the verdict cached for
    AUTO_GPU_CACHE_SEC so the record-start badge and the decode a few seconds
    later act on the same sample. NVML calls are microseconds and thread-safe;
    the lock here only guards our own init/cache state. If NVML is missing or
    the GPU isn't NVIDIA, busy_reason() reports busy with a reason — on such a
    machine local Whisper runs on CPU, which is exactly the slow path "auto"
    exists to avoid."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._handle = None
        self._init_failed: str | None = None
        self._cached_at = 0.0
        self._cached: tuple[str | None, str | None] = (None, None)
        self._free_mb: float | None = None

    def free_vram_mb(self) -> float | None:
        """Free VRAM in MB as of the current (cached) sample, or None when
        NVML can't say."""
        self.busy_reason()          # refreshes the sample if it has gone stale
        with self._lock:
            return self._free_mb

    def _ensure_handle(self):
        if self._handle is not None or self._init_failed is not None:
            return
        try:
            import pynvml
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._pynvml = pynvml
        except Exception as e:
            self._init_failed = f"NVML unavailable ({e.__class__.__name__})"
            log_engine.warning(
                "auto backend: %s — treating the GPU as busy, so cloud STT "
                "is preferred on this machine", self._init_failed,
            )

    def busy_reason(self, need_vram: bool = True) -> str | None:
        """None if the GPU is free enough for a local decode, else a short
        human-readable reason ('GPU 87% busy', 'only 1.2 GB VRAM free').

        need_vram=False skips the free-VRAM floor: an already-loaded model
        needs no new allocation, so only raw utilization matters. The floor
        only guards fresh loads (warm-up, first decode, post-release)."""
        with self._lock:
            now = time.monotonic()
            if now - self._cached_at >= AUTO_GPU_CACHE_SEC:
                self._cached = self._sample()
                self._cached_at = now
            util_reason, vram_reason = self._cached
            return util_reason or (vram_reason if need_vram else None)

    def _sample(self) -> tuple[str | None, str | None]:
        """(utilization-verdict, vram-verdict) — cached as a pair so callers
        with different need_vram get answers from the same sample."""
        self._ensure_handle()
        self._free_mb = None
        if self._init_failed is not None:
            return self._init_failed, None
        try:
            util = self._pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
            free_mb = self._pynvml.nvmlDeviceGetMemoryInfo(
                self._handle).free / (1024 * 1024)
            self._free_mb = float(free_mb)
        except Exception as e:
            # Driver hiccup mid-session (sleep/resume, driver update). Fail
            # toward cloud for this sample and drop the handle so the next
            # sample re-runs nvmlInit instead of re-using a stale handle
            # forever (which would report "busy" for the rest of the session).
            self._handle = None
            self._init_failed = None
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass
            return f"GPU query failed ({e.__class__.__name__})", None
        util_reason = (f"GPU {util:.0f}% busy"
                       if util >= AUTO_GPU_BUSY_UTIL_PCT else None)
        vram_reason = (f"only {free_mb / 1024:.1f} GB VRAM free"
                       if free_mb < AUTO_GPU_MIN_FREE_VRAM_MB else None)
        return util_reason, vram_reason


GPU_MONITOR = GpuLoadMonitor()


class WhisperEngine:
    """Wraps faster-whisper. Single-threaded transcribe; callers serialize."""

    def __init__(self, name: str = DEFAULT_MODEL, settings: "Settings | None" = None):
        self.name = name
        # Optional Settings ref so transcribe() can route to a cloud backend
        # (ElevenLabs Scribe) when the user opts in. None → always local.
        self.settings = settings
        self.device = PREFERRED_DEVICE
        self.compute_type = CUDA_COMPUTE_TYPE
        self._model: WhisperModel | None = None
        # Two locks with distinct responsibilities:
        #   _state_lock — short critical section for reading/swapping the
        #                 self._model reference and adjacent flags. Also held
        #                 during transcribe() so a concurrent switch can't
        #                 tear down the model mid-decode.
        #   _build_lock — serializes the (potentially multi-minute) builds.
        #                 Held during _build_model, NEVER held during the swap
        #                 or during transcribe.
        self._state_lock = threading.Lock()
        self._build_lock = threading.Lock()
        self._supports_hotwords = False
        # ElevenLabs failover pool (created lazily; reads keys + exhausted state
        # live from Settings on each call, so it never goes stale).
        self._eleven_pool: "ElevenLabsScribePool | None" = None
        # Backend that served the most recent transcribe(): "local" |
        # "elevenlabs". Written by the worker thread, read by the UI thread to
        # badge the overlay; a lone str assignment so no lock is needed.
        self.last_backend: str = "local"
        # Consecutive auto-mode clips routed to cloud because the GPU was
        # busy. At AUTO_RELEASE_AFTER_BUSY the local model's VRAM is released.
        self._auto_busy_streak = 0
        # Graveyard for replaced/released models. NEVER let a CUDA-resident
        # WhisperModel be garbage-collected mid-session: CTranslate2's C++
        # destructor frees device memory in a noexcept context, so if the CUDA
        # context has died (e.g. a WDDM reset while a game had the GPU pinned)
        # the error escalates to std::terminate -> uncatchable 0xc0000409
        # process abort. That killed Tellur on 2026-08-16 the first time
        # release_local_model() ever fired. Instead we unload_model() (a normal
        # bound method — CUDA errors surface as catchable Python exceptions)
        # and park the husk here; the handful of retained host-side objects is
        # a few MB per model per session.
        self._released_husks: list = []

    def _build_model(self, name: str) -> tuple["WhisperModel", str, str]:
        """Build a WhisperModel, trying CUDA first and falling back to CPU.
        Does NOT mutate engine state — caller is responsible for installing it.
        May download the model on first use (via faster-whisper / HF Hub)."""
        # A model already on disk is opened straight from the cache. Without
        # this, every load first asked huggingface.co whether the repo had
        # changed: 50-250 ms when the site answers, its full network timeout
        # when it doesn't — on an app whose pitch is that it works offline.
        cached = False
        try:
            cached = is_model_cached(name)
        except Exception:
            log_engine.debug("model cache probe failed", exc_info=True)

        def make(device: str, compute: str) -> "WhisperModel":
            if cached:
                try:
                    return WhisperModel(name, device=device, compute_type=compute,
                                        local_files_only=True)
                except Exception:
                    # Incomplete cache entry (or a real device failure): let
                    # the normal path sort it out, downloading if needed.
                    log_engine.debug(
                        "cache-only load of %s on %s failed; retrying online",
                        name, device, exc_info=True)
            return WhisperModel(name, device=device, compute_type=compute)

        try:
            return make("cuda", CUDA_COMPUTE_TYPE), "cuda", CUDA_COMPUTE_TYPE
        except Exception:
            log_engine.exception("CUDA load failed for %s — falling back to CPU", name)
            return make("cpu", CPU_COMPUTE_TYPE), "cpu", CPU_COMPUTE_TYPE

    def _detect_hotwords_support(self, model: "WhisperModel") -> bool:
        try:
            sig = inspect.signature(model.transcribe)
            return "hotwords" in sig.parameters
        except (TypeError, ValueError):
            return False

    def load(self) -> WhisperModel:
        # Fast path: already loaded.
        with self._state_lock:
            if self._model is not None:
                return self._model
        # Serialize with any concurrent build (warm-up vs. user switch race).
        with self._build_lock:
            with self._state_lock:
                if self._model is not None:
                    return self._model  # another builder beat us to it
            t0 = time.monotonic()
            new_model, dev, ct = self._build_model(self.name)
            supports = self._detect_hotwords_support(new_model)
            with self._state_lock:
                self._model = new_model
                self.device = dev
                self.compute_type = ct
                self._supports_hotwords = supports
            log_engine.info(
                "model loaded name=%s device=%s compute=%s hotwords=%s in %.2fs",
                self.name, dev, ct, supports, time.monotonic() - t0,
            )
            return new_model

    def switch_to(self, name: str) -> None:
        """Swap to a different model. Blocks until loaded (and downloaded, if
        not cached). The build runs under _build_lock (so a concurrent load()
        or switch_to() waits) but NOT under _state_lock (so in-flight
        transcribe()s aren't blocked during a multi-minute download). Only the
        brief reference swap takes _state_lock."""
        with self._state_lock:
            if name == self.name and self._model is not None:
                return
        log_engine.info("switching model: %s -> %s", self.name, name)
        t0 = time.monotonic()
        with self._build_lock:
            # Re-check under build_lock: another switch may have already
            # landed the same model while we waited.
            with self._state_lock:
                if name == self.name and self._model is not None:
                    return
            new_model, dev, ct = self._build_model(name)
            supports = self._detect_hotwords_support(new_model)
            with self._state_lock:
                old_model = self._model
                self._model = new_model
                self.name = name
                self.device = dev
                self.compute_type = ct
                self._supports_hotwords = supports
            # Same destructor hazard as release_local_model: retire the old
            # model explicitly instead of letting the swap GC it.
            if old_model is not None:
                self._retire_model(old_model, "model switch")
        log_engine.info("model switched to %s (%s/%s) in %.2fs",
                        name, dev, ct, time.monotonic() - t0)

    def _retire_model(self, husk: "WhisperModel", why: str) -> None:
        """Free a model's device memory WITHOUT ever running its destructor.

        unload_model() raises a normal Python exception if the CUDA context is
        broken (in which case the driver already reclaimed the memory during
        the device reset, so there is nothing left to free anyway). The husk is
        parked so GC can't trigger the noexcept C++ destructor mid-session."""
        try:
            husk.model.unload_model()
            log_engine.info("%s — released the local model's VRAM", why)
        except Exception as exc:
            log_engine.warning(
                "%s — unload_model failed (%s); CUDA context likely died "
                "(device reset under load), so its VRAM is already gone. "
                "Parking the dead model object.", why, exc,
            )
        self._released_husks.append(husk)

    @property
    def model_resident(self) -> bool:
        """True while a local model is loaded (a lone reference read)."""
        return self._model is not None

    def release_local_model(self) -> None:
        """Unload the model to give its VRAM back (auto mode, sustained GPU
        load). load() rebuilds it transparently on the next local decode.
        Takes _state_lock, so it cannot tear the model down mid-transcribe."""
        with self._state_lock:
            if self._model is None:
                return
            husk = self._model
            self._model = None
        self._retire_model(
            husk,
            "auto backend: GPU busy for %d clips in a row"
            % AUTO_RELEASE_AFTER_BUSY,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        initial_prompt: str | None = None,
        hotwords: str | None = None,
    ) -> str:
        if audio.size < int(SAMPLE_RATE * MIN_AUDIO_SECONDS):
            log_engine.debug("audio too short (%d samples); skipping", audio.size)
            return ""
        # Optional cloud backend: route to ElevenLabs Scribe when selected. Keys
        # fail over automatically (primary → backup → …); an exhausted key is
        # persisted so the switch sticks across restarts. Only if EVERY key is
        # exhausted do we fall back to local Whisper, so dictation never fully
        # breaks even though the user chose the cloud engine.
        #
        # "auto" routes per-clip on GPU load: a free GPU decodes locally
        # (faster than the network round-trip and costs no credits); a busy
        # GPU (game, render, LLM decode) sends the clip to the cloud instead
        # of contending for it. Sustained busy also releases the local model's
        # VRAM back to whatever is using the GPU.
        s = self.settings
        backend = getattr(s, "stt_backend", "local") if s is not None else "local"
        if backend == "auto":
            busy = GPU_MONITOR.busy_reason(need_vram=self._model is None)
            if busy is None:
                self._auto_busy_streak = 0
                backend = "local"
                log_engine.info("auto backend: GPU free — decoding locally")
            else:
                self._auto_busy_streak += 1
                backend = "elevenlabs"
                log_engine.info(
                    "auto backend: %s — routing to ElevenLabs (streak %d)",
                    busy, self._auto_busy_streak,
                )
        auto_routed = (backend == "elevenlabs"
                       and getattr(s, "stt_backend", "local") == "auto")
        if backend == "elevenlabs":
            if s.eleven_keys():
                if self._eleven_pool is None:
                    self._eleven_pool = ElevenLabsScribePool(s)
                t0 = time.monotonic()
                try:
                    text, used_key, idx = self._eleven_pool.transcribe(audio, SAMPLE_RATE)
                    self.last_backend = "elevenlabs"
                    log_engine.info(
                        "scribe transcribed via key #%d (…%s) audio=%.2fs in %.0fms text=%r",
                        idx + 1, used_key[-4:], audio.size / SAMPLE_RATE,
                        (time.monotonic() - t0) * 1000, text,
                    )
                    # Release the local model's VRAM only now — once the
                    # cloud has actually served the Nth busy clip in a row.
                    # Releasing on routing alone (the old behaviour) meant a
                    # clip whose cloud call then failed fell through to a
                    # full model reload on the busy GPU it was avoiding.
                    if (auto_routed
                            and self._auto_busy_streak >= AUTO_RELEASE_AFTER_BUSY
                            and self._model is not None):
                        # ...and only if the card is actually short on
                        # memory; see AUTO_RELEASE_BELOW_FREE_VRAM_MB.
                        free_mb = GPU_MONITOR.free_vram_mb()
                        if (free_mb is None
                                or free_mb < AUTO_RELEASE_BELOW_FREE_VRAM_MB):
                            self.release_local_model()
                    return text
                except ElevenTimeout as e:
                    log_engine.warning(
                        "ElevenLabs gave no answer in time (%s); decoding "
                        "this clip with local Whisper instead.", e,
                    )
                    self._auto_busy_streak = 0
                    # fall through to the local decode below
                except ElevenAllKeysFailed as e:
                    log_engine.error(
                        "all ElevenLabs keys exhausted/failed (%s); "
                        "falling back to local Whisper.", e,
                    )
                    # The cloud didn't serve; the streak measures "cloud is
                    # carrying the load", so it restarts from here.
                    self._auto_busy_streak = 0
                    # fall through to the local decode below
            else:
                log_engine.warning(
                    "elevenlabs backend selected but no API key set; using local Whisper",
                )
                self._auto_busy_streak = 0
        # Everything below this point is the local decode.
        self.last_backend = "local"
        # Rescue quiet recordings before the local decode. Whisper's VAD and
        # decoder give up on speech recorded well below full scale (headset
        # mic drifted from the mouth), returning '' even though the words are
        # audible. Peak-normalizing is a plain linear gain and fixes that
        # class of empties. Scribe never needed this — only the local path.
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak < QUIET_PEAK_SILENCE:
            log_engine.warning(
                "recording is near-silent (peak=%.5f) — mic muted or wrong "
                "input device? Transcription will likely be empty.", peak,
            )
        elif peak < QUIET_PEAK_NORMALIZE:
            audio = audio * (QUIET_PEAK_TARGET / peak)
            log_engine.debug(
                "quiet recording normalized (peak %.4f → %.2f)",
                peak, QUIET_PEAK_TARGET,
            )
        # Ensure a model is loaded (may take minutes on first run).
        self.load()
        # Hold _state_lock for the entire decode so a switch_to() can't tear
        # the model down mid-call. The lock is microseconds compared to the
        # decode itself, but it correctly serializes against the swap.
        with self._state_lock:
            model = self._model
            supports_hotwords = self._supports_hotwords
            if model is None:
                return ""
            fragile = self.prompt_fragile()
            # Plain decode: audio only, faster-whisper defaults for the
            # decoder. This is the whole config for prompt-fragile (distil)
            # models and the rescue config for everything else.
            plain = dict(
                language=LANGUAGE,
                beam_size=BEAM_SIZE,
                vad_filter=True,
                # threshold 0.35 (Silero default 0.5): quiet-but-real speech
                # was getting classified as silence and dropped wholesale.
                vad_parameters={"min_silence_duration_ms": 250,
                                "threshold": 0.35},
                condition_on_previous_text=False,
            )
            if fragile:
                kwargs = dict(plain)
            else:
                kwargs = dict(
                    plain,
                    no_repeat_ngram_size=NO_REPEAT_NGRAM_SIZE,
                    repetition_penalty=REPETITION_PENALTY,
                    initial_prompt=initial_prompt,
                )
                if supports_hotwords and hotwords:
                    kwargs["hotwords"] = hotwords
            t0 = time.monotonic()
            segments, _info = model.transcribe(audio, **kwargs)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            dur = audio.size / SAMPLE_RATE
            log_engine.info(
                "transcribed audio=%.2fs in %.0fms%s text=%r",
                dur, (time.monotonic() - t0) * 1000,
                " (plain)" if fragile else "", text,
            )
            if self._looks_truncated(text, dur, peak):
                # Early end-of-text or a VAD that swallowed the speech. One
                # retry with the plain config and no VAD; keep whichever
                # decode found more words.
                t1 = time.monotonic()
                rescue = dict(plain, vad_filter=False)
                rescue.pop("vad_parameters", None)
                segments, _info = model.transcribe(audio, **rescue)
                text2 = " ".join(seg.text.strip() for seg in segments).strip()
                n1, n2 = len(text.split()), len(text2.split())
                log_engine.warning(
                    "local decode looks truncated (%d words in %.1fs) — "
                    "re-decoded plain/no-VAD in %.0fms: %d words%s",
                    n1, dur, (time.monotonic() - t1) * 1000, n2,
                    " (kept retry)" if n2 > n1 else " (kept original)",
                )
                if n2 > n1:
                    log_engine.info("rescued text=%r", text2)
                    text = text2
            return text

    def prompt_fragile(self) -> bool:
        """True when the loaded model must be decoded without any prompt
        context (see PROMPT_FRAGILE_MODEL_PREFIXES)."""
        base = (self.name or "").rsplit("/", 1)[-1].lower()
        return base.startswith(PROMPT_FRAGILE_MODEL_PREFIXES)

    @staticmethod
    def _looks_truncated(text: str, dur: float, peak: float) -> bool:
        if peak < QUIET_PEAK_SILENCE or dur < LOCAL_GUARD_MIN_SECONDS:
            return False
        words = len(text.split())
        return words / dur < LOCAL_GUARD_MIN_WORDS_PER_SEC


# ===========================================================================
# LLM post-processing (v1.7) — talks to OpenAI-compatible local endpoints
# (LM Studio, Ollama, llama.cpp server, vLLM, …). Cloud-providers also work
# but the project's default story is local-first / offline-first.
# ===========================================================================

# Built-in prompt presets. Each preset has an id, label, and system prompt
# string. The user prompt is always just the transcript text. We default to
# the cleanup preset because it's the lowest-impact "always wins" option.
BUILTIN_LLM_PROMPTS: list[dict] = [
    {
        "id": "cleanup",
        "label": "Clean up (remove filler words)",
        "system": (
            "You receive a raw speech-to-text transcription. Lightly clean it up: "
            "remove filler words (um, uh, like, you know, sort of, kind of, I mean), "
            "fix obvious dictation errors, and produce natural punctuation and "
            "capitalization. PRESERVE the speaker's voice and meaning exactly. "
            "Do not add information that wasn't said. Do not rephrase unless the "
            "transcript is grammatically broken. Output ONLY the cleaned text — "
            "no preamble, no commentary, no quotes around the result."
        ),
    },
    {
        "id": "paragraph",
        "label": "Make it a clean paragraph",
        "system": (
            "You receive a raw speech-to-text transcription. Rewrite it as ONE clean, "
            "well-punctuated paragraph in the speaker's voice. Remove filler words, "
            "fix run-ons, and combine fragments. Preserve meaning and tone exactly. "
            "Output ONLY the paragraph — no preamble, no quotes."
        ),
    },
    {
        "id": "formal",
        "label": "Make it formal",
        "system": (
            "You receive a raw speech-to-text transcription. Rewrite it in formal, "
            "professional written English suitable for a business email or report. "
            "Preserve meaning exactly; only adjust tone and structure. "
            "Output ONLY the rewritten text — no preamble, no quotes."
        ),
    },
    {
        "id": "bullets",
        "label": "Convert to bullet points",
        "system": (
            "You receive a raw speech-to-text transcription. Restructure its content "
            "as a concise bulleted list using '- ' markdown bullets. Group related "
            "points; drop filler. Preserve meaning. Output ONLY the bullets — no "
            "preamble, no headings unless the speaker explicitly stated them."
        ),
    },
    {
        "id": "email",
        "label": "Convert to email",
        "system": (
            "You receive a raw speech-to-text transcription that the speaker wants "
            "as an email. Format it as a clean email body (no Subject line unless the "
            "speaker stated one). Use a polite professional tone in the speaker's "
            "voice. Output ONLY the email body."
        ),
    },
    {
        "id": "slack",
        "label": "Convert to Slack message",
        "system": (
            "You receive a raw speech-to-text transcription that the speaker wants "
            "to send as a Slack message. Make it conversational and concise (Slack "
            "norms: shorter sentences, friendly, optional emoji only if speaker "
            "implied tone). Output ONLY the message text."
        ),
    },
    {
        "id": "summarize",
        "label": "Summarize",
        "system": (
            "You receive a raw speech-to-text transcription. Produce a tight summary "
            "(2–4 sentences) that captures the speaker's key points. Output ONLY the "
            "summary — no preamble, no headings."
        ),
    },
]


def llm_prompt_by_id(prompt_id: str) -> dict | None:
    for p in BUILTIN_LLM_PROMPTS:
        if p["id"] == prompt_id:
            return p
    return None


class LLMClient:
    """OpenAI-compatible chat completions client using only urllib (stdlib).

    Designed for local-first endpoints: LM Studio, Ollama (`/v1/`), llama.cpp
    server, vLLM, etc. The same code also works against cloud endpoints if
    the user wants — we don't gate on URL.

    Synchronous. Caller is expected to invoke from a worker thread so we
    never block the Qt event loop."""

    def __init__(
        self,
        base_url: str = "http://localhost:1234/v1",
        model: str = "",
        api_key: str = "",
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, system: str, user: str) -> str:
        """Single-turn chat. Returns the assistant's reply as a string.
        Raises an exception on transport or server errors."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "stream": False,
        }
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        # OpenAI-compatible shape: choices[0].message.content
        try:
            return (data["choices"][0]["message"]["content"] or "").strip()
        except (KeyError, IndexError, TypeError) as e:
            raise RuntimeError(f"unexpected LLM response shape: {e!r} — body={raw[:300]}")

    def ping(self) -> bool:
        """Cheap reachability check — list models. Returns True on 2xx."""
        url = f"{self.base_url}/models"
        headers: dict = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                return 200 <= resp.status < 300
        except Exception:
            return False


# ===========================================================================
# integration sinks (v1.8) — write each transcript to durable / external
# destinations beyond the in-app history & clipboard. Each sink is one-shot
# and best-effort: a failure here logs and moves on, it never blocks the
# main transcript pipeline.
# ===========================================================================
class MarkdownDailySink:
    """Appends every accepted transcript to `YYYY-MM-DD.md` under a configured
    folder. One file per local-time day. Header is created on first write of
    a day so existing days don't get re-headed."""

    def __init__(self, folder: Path):
        self.folder = folder

    def write(self, text: str, *, ts: float, raw: str = "") -> None:
        if not text.strip():
            return
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            day = time.strftime("%Y-%m-%d", time.localtime(ts))
            path = self.folder / f"{day}.md"
            is_new = not path.exists()
            stamp = time.strftime("%H:%M:%S", time.localtime(ts))
            with path.open("a", encoding="utf-8") as f:
                if is_new:
                    f.write(f"# Tellur transcripts — {day}\n\n")
                f.write(f"## {stamp}\n\n{text.strip()}\n\n")
            log_app.debug("markdown sink wrote %d chars to %s", len(text), path.name)
        except Exception:
            log_app.exception("markdown sink failed")


class WebhookSink:
    """POSTs each transcript to a user-configured URL. The template string
    supports `{text}`, `{raw}`, and `{ts}` placeholders. If the template
    parses as JSON, Content-Type is set to application/json; otherwise the
    template is sent as text/plain. Failures are logged and swallowed."""

    def __init__(self, url: str, template: str, timeout: float = 10.0):
        self.url = url
        self.template = template
        self.timeout = timeout

    @staticmethod
    def _json_escape(s: str) -> str:
        # json.dumps(s)[1:-1] gives us the body of a JSON string without quotes,
        # which is the right thing to substitute into a JSON template.
        return json.dumps(s)[1:-1]

    def post(self, text: str, *, ts: float, raw: str = "") -> None:
        if not self.url:
            return
        body = (
            self.template
            .replace("{text}", self._json_escape(text))
            .replace("{raw}", self._json_escape(raw))
            .replace("{ts}", str(int(ts)))
        )
        # Decide content type based on whether the rendered body parses as JSON.
        content_type = "text/plain; charset=utf-8"
        try:
            json.loads(body)
            content_type = "application/json"
        except Exception:
            pass
        try:
            req = urllib.request.Request(
                self.url,
                data=body.encode("utf-8"),
                headers={"Content-Type": content_type, "User-Agent": f"Tellur/{__version__}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                if not (200 <= resp.status < 300):
                    log_app.warning("webhook returned %s", resp.status)
                else:
                    log_app.debug("webhook POST OK (%d chars body)", len(body))
        except Exception:
            log_app.exception("webhook POST failed")


# ===========================================================================
# text injection (terminal-aware: clipboard + paste chord, or typed unicode)
# ===========================================================================
def pick_paste_strategy() -> str:
    """Decide how to inject text into the focused window, from the foreground
    window's class and process (see the terminal-aware paste constants for
    the why). Returns "type" for character-by-character unicode typing, or a
    chord for `keyboard.send` ("ctrl+v", "shift+insert", "ctrl+shift+v")."""
    cls_name, proc = get_foreground_window_info()
    el_class: str | None = None
    if cls_name == CONSOLE_WINDOW_CLASS:
        # Bare cmd/PowerShell window — chords disabled in raw mode; type instead.
        strategy = "type"
    elif proc in ELECTRON_EDITOR_PROCS:
        # Editor, chat panel, and integrated terminal all share this window
        # class; only the focused element distinguishes them. An xterm.js
        # terminal (incl. a raw-mode TUI like Claude Code) can't be relied on to
        # accept a synthetic Ctrl+V, so type the characters — the one path that
        # always lands. Everything else in the editor (chat box included) keeps
        # Ctrl+V, which the v2.3.1 revert established as the working default.
        el_class = get_focused_element_class()
        if el_class and XTERM_FOCUS_CLASS_MARKER in el_class.lower():
            strategy = "type"
        else:
            strategy = "ctrl+v"
    else:
        strategy = PASTE_CHORD_BY_PROCESS.get(proc or "", "ctrl+v")
    log_paste.debug(
        "paste target class=%s proc=%s el=%s → strategy=%s",
        cls_name, proc, el_class, strategy,
    )
    return strategy


class TextInjector:
    """Injects text into the focused window. Default path is clipboard +
    paste chord (Ctrl+V for most apps; per-app overrides for IDE terminals
    that only paste on Shift+Insert). Legacy console windows get the text
    *typed* as unicode key events instead — no paste chord works there once
    a TUI app holds the console in raw mode.

    For chord pastes, the previous clipboard contents are captured on the
    FIRST paste in a rapid-fire session and only restored after
    CLIPBOARD_RESTORE_DELAY_SEC of quiet, so consecutive dictations never
    lose the user's original clipboard. The typing path never touches the
    clipboard at all."""

    def __init__(self, settings: Settings | None = None):
        self._lock = threading.Lock()
        self._paste_serial = threading.Lock()
        self._saved: str | None = None
        self._gen = 0
        self._settings = settings

    def _type_into_console(self, text: str, *, press_enter: bool) -> None:
        """Type `text` as synthetic unicode key events (the keyboard lib's
        KEYEVENTF_UNICODE path — layout-independent, accepted by conhost in
        both cooked and raw input modes). Newlines/tabs are flattened to
        spaces so we can never press Enter into a shell (which would execute
        a partial command line) or submit a TUI prompt mid-transcript."""
        cleaned = re.sub(r"[\r\n\t]+", " ", text).strip()
        if not cleaned:
            return
        try:
            keyboard.write(cleaned, delay=TERMINAL_TYPE_KEY_DELAY_SEC)
        except Exception:
            log_paste.exception("console typing failed")
            return
        log_paste.debug("typed %d chars into console window", len(cleaned))
        if press_enter:
            try:
                time.sleep(PASTE_KEYSTROKE_DELAY_SEC)
                keyboard.send("enter")
                log_paste.debug("send-and-enter: pressed Enter")
            except Exception:
                log_paste.exception("send-and-enter keystroke failed")

    def _wait_for_keys_released(self) -> None:
        """Block (bounded) while Ctrl+Win is physically held. See
        PASTE_WAIT_FOR_KEYS_SEC. Called off the GUI thread."""
        try:
            if not _ctrl_win_held():
                return
        except Exception:
            return
        t0 = time.monotonic()
        while time.monotonic() - t0 < PASTE_WAIT_FOR_KEYS_SEC:
            time.sleep(0.02)
            try:
                if not _ctrl_win_held():
                    break
            except Exception:
                break
        log_paste.debug("paste waited %.2fs for Ctrl+Win release",
                        time.monotonic() - t0)

    def paste(self, text: str, *, press_enter: bool = False) -> None:
        if not text:
            return
        # Serialize whole pastes: two transcripts can finish while the user
        # holds Ctrl+Win for a third; they must go out in the order they
        # arrived here, and never interleave clipboard/keystroke sequences.
        with self._paste_serial:
            self._wait_for_keys_released()
            self._paste_now(text, press_enter=press_enter)

    def _paste_now(self, text: str, *, press_enter: bool = False) -> None:
        terminal_aware = (
            self._settings is None or self._settings.terminal_aware_paste
        )
        strategy = pick_paste_strategy() if terminal_aware else "ctrl+v"
        if strategy == "type":
            self._type_into_console(text, press_enter=press_enter)
            return
        # Track whether THIS call was the one that populated _saved, so that if
        # we abort early we can roll it back (otherwise the user's clipboard
        # gets permanently stuck under our _saved).
        we_set_saved = False
        with self._lock:
            if self._saved is None:
                try:
                    self._saved = pyperclip.paste()
                    we_set_saved = True
                except Exception:
                    log_paste.debug("clipboard read failed", exc_info=True)
            self._gen += 1
            my_gen = self._gen

        def _abort_clear(restore: bool) -> None:
            if not we_set_saved:
                return
            with self._lock:
                # Only clear if no newer paste has populated it.
                if self._gen != my_gen:
                    return
                saved, self._saved = self._saved, None
            if restore and saved is not None:
                # The clipboard already holds our text; put the user's
                # content back rather than leaving it replaced.
                try:
                    pyperclip.copy(saved)
                except Exception:
                    log_paste.debug("clipboard restore after abort failed",
                                    exc_info=True)

        try:
            pyperclip.copy(text)
        except Exception:
            log_paste.exception("clipboard copy failed")
            _abort_clear(restore=False)
            return
        time.sleep(PASTE_KEYSTROKE_DELAY_SEC)
        try:
            keyboard.send(strategy)
        except Exception:
            log_paste.exception("paste keystroke failed")
            _abort_clear(restore=True)
            return
        log_paste.debug("pasted %d chars (%s)", len(text), strategy)

        # Optional "send & enter" mode (v1.8): some chat apps want a press of
        # Enter to submit the freshly-pasted message. We add a tiny gap so the
        # paste finishes before the Enter so single-line apps don't race.
        if press_enter:
            try:
                time.sleep(PASTE_KEYSTROKE_DELAY_SEC)
                keyboard.send("enter")
                log_paste.debug("send-and-enter: pressed Enter")
            except Exception:
                log_paste.exception("send-and-enter keystroke failed")

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
# input resilience — hook-free key polling, hook watchdog, input priority
#
# The push-to-talk poll used to read keyboard.is_pressed(), which consults a
# table maintained by the keyboard lib's WH_KEYBOARD_LL hook. Windows
# silently REMOVES such a hook when its callback misses the
# LowLevelHooksTimeout deadline — which reliably happens while a heavy game
# (GTA V Enhanced, 2026-07-23) launches and starves this process. No error is
# raised anywhere; dictation is simply dead until restart. So:
#   * the primary Ctrl+Win poll now asks the input system directly via
#     GetAsyncKeyState, which needs no hook and cannot be revoked;
#   * the chord hotkeys (quit/cancel/repaste/llm/note) still ride the hook,
#     so HotkeyWatcher probes for hook death and reinstalls a fresh hook on a
#     new pump thread (registered handlers hang off keyboard._listener and
#     survive — only the OS-level hook is replaced);
#   * raise_input_priority() makes the whole failure less likely by keeping
#     the hook callback schedulable under full-core load.
# ===========================================================================
_WIN32 = sys.platform == "win32"
if _WIN32:
    import ctypes as _ctypes
    # Private WinDLL instances (not ctypes.windll.*) so our argtypes can't
    # leak onto the shared cached function objects other libs use.
    _user32 = _ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = _ctypes.WinDLL("kernel32", use_last_error=True)
    _user32.GetAsyncKeyState.argtypes = (_ctypes.c_int,)
    _user32.GetAsyncKeyState.restype = _ctypes.c_short
    _user32.VkKeyScanW.argtypes = (_ctypes.c_wchar,)
    _user32.VkKeyScanW.restype = _ctypes.c_short
    _user32.SetWindowPos.argtypes = (
        _ctypes.c_void_p, _ctypes.c_void_p, _ctypes.c_int, _ctypes.c_int,
        _ctypes.c_int, _ctypes.c_int, _ctypes.c_uint)
    _kernel32.SetPriorityClass.argtypes = (_ctypes.c_void_p, _ctypes.c_uint)
    _kernel32.SetThreadPriority.argtypes = (_ctypes.c_void_p, _ctypes.c_int)
    _kernel32.OpenThread.argtypes = (
        _ctypes.c_uint, _ctypes.c_int, _ctypes.c_uint)
    _kernel32.OpenThread.restype = _ctypes.c_void_p
    _kernel32.CloseHandle.argtypes = (_ctypes.c_void_p,)
    # Pseudo-handles are pointer-sized -1/-2. Passing a bare Python -1
    # through untyped ctypes truncates to 32 bits and the call fails with an
    # invalid handle (bit us in 2.3.4 testing) — hence c_void_p here.
    _CURRENT_PROCESS = _ctypes.c_void_p(-1)
    _CURRENT_THREAD = _ctypes.c_void_p(-2)

_VK_CONTROL, _VK_LWIN, _VK_RWIN = 0x11, 0x5B, 0x5C
_VK_KEYBOARD_FIRST = 0x08   # skip mouse-button VKs (0x01-0x06): keyboard hook
                            # health must not be judged by mouse activity


# Chord hotkeys (quit / repaste / llm / note / Esc-cancel) are polled with
# GetAsyncKeyState exactly like the Ctrl+Win push-to-talk, so they no longer
# depend on the `keyboard` lib's WH_KEYBOARD_LL hook at all. Two real bugs
# went away with the hook: (1) the lib re-fired a chord on every keyboard
# auto-repeat, so holding Ctrl+Win+Z for a second toggled note mode ~12
# times (69 toggles in one minute on 2026-08-08) and holding Ctrl+Win+B
# re-pasted repeatedly; (2) Windows silently revokes the LL hook under load,
# which the watchdog below papered over with a fresh pump thread ~hourly
# (57 replacements in one 8-day session, each leaking a message-pump thread).
# Edge detection on physical key state fires once per press, period.
_VK_BY_NAME = {
    "esc": 0x1B, "escape": 0x1B, "space": 0x20, "spacebar": 0x20,
    "enter": 0x0D, "return": 0x0D, "tab": 0x09, "backspace": 0x08,
    "delete": 0x2E, "del": 0x2E, "insert": 0x2D, "ins": 0x2D,
    "home": 0x24, "end": 0x23, "page up": 0x21, "pageup": 0x21,
    "page down": 0x22, "pagedown": 0x22, "up": 0x26, "down": 0x28,
    "left": 0x25, "right": 0x27, "pause": 0x13, "print screen": 0x2C,
    "caps lock": 0x14, "num lock": 0x90, "scroll lock": 0x91,
    "menu": 0x5D, "apps": 0x5D,
}
for _i in range(1, 25):
    _VK_BY_NAME[f"f{_i}"] = 0x6F + _i
# Modifier tokens → the VKs that satisfy them (any one of the tuple).
_MOD_VKS = {
    "ctrl": (0x11,), "control": (0x11,), "left ctrl": (0xA2,),
    "right ctrl": (0xA3,),
    "windows": (0x5B, 0x5C), "win": (0x5B, 0x5C), "left windows": (0x5B,),
    "right windows": (0x5C,), "cmd": (0x5B, 0x5C), "super": (0x5B, 0x5C),
    "alt": (0x12,), "left alt": (0xA4,), "right alt": (0xA5,), "alt gr": (0xA5,),
    "shift": (0x10,), "left shift": (0xA0,), "right shift": (0xA1,),
}


def parse_chord(combo: str) -> tuple[tuple[tuple[int, ...], ...], int] | None:
    """Turn a keyboard-lib combo string ("ctrl+windows+z", "escape") into
    (modifier groups, final VK) for GetAsyncKeyState polling. Each modifier
    group is a tuple of VKs of which any one must be down; the final key is a
    single VK. Returns None if the string has no single non-modifier key or
    names something this table can't map (caller falls back to the hook)."""
    if not combo:
        return None
    tokens = [t.strip().lower() for t in combo.split("+")]
    tokens = [t for t in tokens if t]
    if not tokens:
        return None
    mods: list[tuple[int, ...]] = []
    final: int | None = None
    for t in tokens:
        if t in _MOD_VKS:
            mods.append(_MOD_VKS[t])
            continue
        if final is not None:
            return None          # two non-modifier keys — not a chord we poll
        if t in _VK_BY_NAME:
            final = _VK_BY_NAME[t]
        elif len(t) == 1 and (t.isalnum() and t.isascii()):
            final = ord(t.upper())
        elif len(t) == 1 and _WIN32:
            # Punctuation: VkKeyScanW gives VK in the low byte, shift state
            # in the high byte (ignored — the chord's own modifiers rule).
            r = _user32.VkKeyScanW(t)
            if r == -1:
                return None
            final = r & 0xFF
        else:
            return None
    if final is None:
        return None
    return tuple(mods), final


def _chord_held(chord: tuple[tuple[tuple[int, ...], ...], int]) -> bool:
    """True while every modifier group has a key down AND the final key is
    down, per GetAsyncKeyState."""
    down = _user32.GetAsyncKeyState
    mods, final = chord
    if not down(final) & 0x8000:
        return False
    for group in mods:
        if not any(down(vk) & 0x8000 for vk in group):
            return False
    return True


def _ctrl_win_held() -> bool:
    """True while Ctrl and a Win key are physically held, per GetAsyncKeyState
    (hook-free). Non-Windows falls back to the keyboard lib."""
    if not _WIN32:
        return keyboard.is_pressed("ctrl") and (
            keyboard.is_pressed("left windows")
            or keyboard.is_pressed("right windows"))
    down = _user32.GetAsyncKeyState
    return bool(down(_VK_CONTROL) & 0x8000) and (
        bool(down(_VK_LWIN) & 0x8000) or bool(down(_VK_RWIN) & 0x8000))


def _keyboard_down_bitmap() -> int:
    """Bitmap of currently-down keyboard VKs via GetAsyncKeyState. A key held
    across two probe ticks that the LL hook never reported is proof the hook
    is gone. Tap-and-release between ticks can slip by — fine, the point is
    catching sustained activity (WASD in a game) with a silent hook."""
    if not _WIN32:
        return 0
    down = _user32.GetAsyncKeyState
    bits = 0
    for vk in range(_VK_KEYBOARD_FIRST, 256):
        if down(vk) & 0x8000:
            bits |= 1 << vk
    return bits


def _bump_current_thread_priority() -> None:
    """THREAD_PRIORITY_HIGHEST for the calling thread (hook pump threads)."""
    if _WIN32:
        try:
            _kernel32.SetThreadPriority(_CURRENT_THREAD, 2)
        except Exception:
            pass


def raise_input_priority() -> None:
    """Above-normal process priority plus highest thread priority for the
    keyboard lib's hook pump thread. The LL hook callback must answer within
    the OS deadline even while a AAA game launch saturates every core, or
    Windows revokes the hook without notice. Call once after startup (the
    keyboard listener thread must already exist)."""
    if not _WIN32:
        return
    try:
        ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000
        ok = _kernel32.SetPriorityClass(
            _CURRENT_PROCESS, ABOVE_NORMAL_PRIORITY_CLASS)
        log_app.info("process priority -> above normal (%s)",
                     "ok" if ok else "failed")
    except Exception:
        log_app.exception("failed to raise process priority")
    try:
        t = getattr(keyboard._listener, "listening_thread", None)
        tid = getattr(t, "native_id", None)
        if tid:
            THREAD_SET_INFORMATION, THREAD_QUERY_INFORMATION = 0x0020, 0x0040
            h = _kernel32.OpenThread(
                THREAD_SET_INFORMATION | THREAD_QUERY_INFORMATION, False, tid)
            if h:
                _kernel32.SetThreadPriority(h, 2)  # THREAD_PRIORITY_HIGHEST
                _kernel32.CloseHandle(h)
                log_hotkey.info("hook thread priority -> highest (tid=%s)", tid)
    except Exception:
        log_hotkey.exception("failed to raise hook thread priority")


# ===========================================================================
# hotkey watcher — polls Ctrl + Win held; emits press / release transitions
# ===========================================================================
class HotkeyWatcher(QObject):
    """Polls Ctrl+Win as the push-to-talk trigger. Supports two modes:
    - "hold"   — press emits `pressed`, release emits `released` (default)
    - "toggle" — first press starts recording (pressed); next press ends
                 it (released). Ignores release events in between.

    Both the Ctrl+Win poll AND the quit / cancel / re-paste / llm / note
    chords are hook-free (GetAsyncKeyState edge detection, 20 ms poll — see
    the input resilience section above). The `keyboard` lib's low-level hook
    is only installed as a fallback for a chord string `parse_chord` can't
    map; only then does the hook-death watchdog run.
    """

    pressed = pyqtSignal()
    released = pyqtSignal()
    quit_requested = pyqtSignal()
    cancel_requested = pyqtSignal()       # Esc during recording → abandon
    repaste_requested = pyqtSignal()      # Ctrl+Win+B → re-paste last transcript
    llm_apply_requested = pyqtSignal()    # Ctrl+Win+L → run default LLM prompt on last
    note_toggle_requested = pyqtSignal()  # Ctrl+Win+N → start/stop a note recording

    # Slots whose binding is configurable via Settings (everything but the
    # primary Ctrl+Win poll, which doesn't use keyboard.add_hotkey at all).
    _CONFIGURABLE_SLOTS = ("quit", "repaste", "llm", "note")
    _ALL_SLOTS = _CONFIGURABLE_SLOTS + ("cancel",)

    def __init__(self, poll_ms: int = HOTKEY_POLL_MS,
                 settings: "Settings | None" = None):
        super().__init__()
        self._held_keys = False
        self._recording = False
        self._mode = "hold"
        # When the App is mid note-recording, dictation-poll must stand down
        # so a still-held Ctrl+Win doesn't keep retriggering dictation under
        # the note. App flips this via set_note_in_progress().
        self._note_in_progress = False
        self._settings = settings
        # Track add_hotkey handles per slot so we can remove a stale binding
        # before installing a new one (user changes a hotkey in Settings).
        # Only populated for slots that fell back to the keyboard-lib hook.
        self._hotkey_handles: dict[str, int] = {}
        # Polled chords: slot -> parsed (modifier groups, final VK), plus the
        # per-slot "was held last tick" flag for edge detection.
        self._chords: dict[str, tuple] = {}
        self._chord_down: dict[str, bool] = {}
        self._timer = QTimer(self)
        self._timer.setInterval(poll_ms)
        self._timer.timeout.connect(self._poll)

        # --- hook-death watchdog (fallback slots only) ---------------------
        # A chord that couldn't be parsed for polling lives on the keyboard
        # lib's WH_KEYBOARD_LL hook, which Windows silently revokes under
        # load (see input resilience section). Track every event the hook
        # DOES deliver; if fresh key activity shows up via GetAsyncKeyState
        # while the hook stays silent, install a replacement hook. None of
        # this is armed while every slot polls natively.
        self._hook_last_event = time.monotonic()
        self._hook_down_bitmap = 0
        self._last_rehook = 0.0
        self._rehook_count = 0
        self._hook_probe_installed = False
        self._hook_timer = QTimer(self)
        self._hook_timer.setInterval(HOOK_HEALTH_POLL_MS)
        self._hook_timer.timeout.connect(self._check_hook_health)
        for slot in self._ALL_SLOTS:
            self._register_slot(slot)

    def _slot_emit(self, slot: str):
        return {
            "quit":    self.quit_requested.emit,
            "repaste": self.repaste_requested.emit,
            "llm":     self.llm_apply_requested.emit,
            "note":    self.note_toggle_requested.emit,
            "cancel":  self.cancel_requested.emit,
        }[slot]

    def _slot_combo(self, slot: str) -> str:
        """Resolve a slot's combo string. Cancel is always Esc; the
        configurable ones come from Settings (with a built-in default if
        Settings isn't wired up or the field is missing)."""
        if slot == "cancel":
            return "escape"
        defaults = {
            "quit":    "ctrl+windows+q",
            "repaste": "ctrl+windows+b",   # ctrl+win+v is reserved by Win11
            "llm":     "ctrl+windows+l",
            "note":    "ctrl+windows+z",
        }
        if self._settings is not None:
            v = getattr(self._settings, f"hotkey_{slot}", None)
            if isinstance(v, str) and v.strip():
                return v.strip().lower()
        return defaults[slot]

    def _register_slot(self, slot: str) -> None:
        """Bind (or rebind) one slot. Removes any prior binding first so a
        Settings edit doesn't leave the old combo wired up. Polled when the
        combo parses; the keyboard-lib hook is the fallback."""
        combo = self._slot_combo(slot)
        if slot in self._hotkey_handles:
            try:
                keyboard.remove_hotkey(self._hotkey_handles[slot])
            except (KeyError, ValueError):
                pass
            except Exception:
                log_hotkey.exception("failed to remove prior %s hotkey", slot)
            self._hotkey_handles.pop(slot, None)
        self._chords.pop(slot, None)
        self._chord_down.pop(slot, None)
        chord = parse_chord(combo) if _WIN32 else None
        if chord is not None:
            self._chords[slot] = chord
            self._chord_down[slot] = False
            log_hotkey.info("registered %s hotkey: %s (polled)", slot, combo)
            return
        try:
            kwargs = {"suppress": False} if slot == "cancel" else {}
            handle = keyboard.add_hotkey(combo, self._slot_emit(slot), **kwargs)
            self._hotkey_handles[slot] = handle
            log_hotkey.info("registered %s hotkey: %s (hook fallback)", slot, combo)
            self._ensure_hook_watchdog()
        except Exception:
            log_hotkey.exception("failed to register %s hotkey (%r)", slot, combo)

    def _ensure_hook_watchdog(self) -> None:
        """Arm the LL-hook health probe — only needed once a slot has fallen
        back to keyboard.add_hotkey."""
        if self._hook_probe_installed:
            return
        try:
            keyboard.hook(self._note_hook_event)
            self._hook_probe_installed = True
        except Exception:
            log_hotkey.exception("failed to register hook-health probe")
        if self._timer.isActive() and not self._hook_timer.isActive():
            self._hook_timer.start()

    @property
    def _hook_in_use(self) -> bool:
        return bool(self._hotkey_handles)

    def _poll_chords(self) -> None:
        """Edge-detect every polled chord; fire once per physical press."""
        for slot, chord in self._chords.items():
            try:
                held = _chord_held(chord)
            except Exception:
                log_hotkey.exception("chord poll failed for %s", slot)
                held = False
            if held and not self._chord_down.get(slot, False):
                self._chord_down[slot] = True
                try:
                    self._slot_emit(slot)()
                except Exception:
                    log_hotkey.exception("chord %s handler raised", slot)
            elif not held:
                self._chord_down[slot] = False

    def apply_settings(self) -> None:
        """Re-bind every configurable hotkey from current Settings. Call
        this when Settings.changed fires so edits take effect live."""
        for slot in self._CONFIGURABLE_SLOTS:
            self._register_slot(slot)

    def set_mode(self, mode: str) -> None:
        """Switch between 'hold' and 'toggle'. Mid-recording mode changes
        end any in-flight recording cleanly."""
        if mode not in ("hold", "toggle"):
            mode = "hold"
        if mode == self._mode:
            return
        log_hotkey.info("hotkey mode -> %s", mode)
        # If we were recording, force-release so we don't get stuck.
        if self._recording:
            self._recording = False
            self.released.emit()
        self._mode = mode

    def start(self) -> None:
        self._timer.start()
        if self._hook_in_use:
            self._hook_timer.start()
        log_hotkey.info(
            "watching ctrl+win in %s mode (async key poll; %d chord(s) polled, "
            "%d on keyboard hook%s)", self._mode, len(self._chords),
            len(self._hotkey_handles),
            " + hook watchdog" if self._hook_in_use else "")

    def _all_held(self) -> bool:
        try:
            return _ctrl_win_held()
        except Exception:
            log_hotkey.exception("key state poll failed")
            return False

    # --- hook-death watchdog ----------------------------------------------
    def _note_hook_event(self, _event) -> None:
        # Runs on the hook pump thread for every key event — keep it tiny.
        self._hook_last_event = time.monotonic()

    def _check_hook_health(self) -> None:
        """Periodic probe: a key went down (per GetAsyncKeyState) since last
        tick, yet the LL hook reported nothing for 1.5+ ticks → hook is dead.
        Held gaming keys (WASD) trip this within a tick or two."""
        bitmap = _keyboard_down_bitmap()
        fresh = bitmap & ~self._hook_down_bitmap
        self._hook_down_bitmap = bitmap
        if not fresh:
            return
        quiet = time.monotonic() - self._hook_last_event
        if quiet > (self._hook_timer.interval() / 1000.0) * 1.5:
            self._rehook(f"key activity while hook silent {quiet:.1f}s")

    def _verify_hook_saw_press(self, t_press: float) -> None:
        """Scheduled 1s after each PTT press (detected hook-free): a live
        hook must have seen those same Ctrl/Win keydowns. Dictation itself
        already started — this only resurrects the chord hotkeys sooner."""
        if self._hook_last_event < t_press - 0.05:
            self._rehook("push-to-talk press invisible to hook")

    def _rehook(self, reason: str) -> None:
        now = time.monotonic()
        if now - self._last_rehook < REHOOK_MIN_GAP_SEC:
            return
        self._last_rehook = now
        self._rehook_count += 1
        log_hotkey.warning(
            "keyboard hook dead (%s) — installing replacement #%d "
            "(chord hotkeys were offline; dictation itself is hook-free)",
            reason, self._rehook_count)
        try:
            # Fresh SetWindowsHookEx + message pump on a new thread, feeding
            # the same keyboard._listener — all registered handlers/hotkeys
            # carry over. The old pump thread keeps running harmlessly; its
            # hook is the thing that died. Private API, pinned lib version.
            def _pump():
                _bump_current_thread_priority()
                keyboard._os_keyboard.listen(keyboard._listener.direct_callback)
            threading.Thread(target=_pump, daemon=True,
                             name=f"kb-hook-{self._rehook_count}").start()
            self._hook_last_event = time.monotonic()  # grace for the new hook
        except Exception:
            log_hotkey.exception("hook reinstall failed")

    @property
    def recording(self) -> bool:
        return self._recording

    def force_release(self) -> None:
        """External cancel — used by Esc to abandon the current recording.
        Emits `released` if we were recording so the App can stop the audio
        stream, but the caller is responsible for ignoring the transcript
        (e.g. via a 'cancelled' flag)."""
        if self._recording:
            self._recording = False
            self.released.emit()

    def set_note_in_progress(self, on: bool) -> None:
        """Pause/resume Ctrl+Win dictation polling while a Ctrl+Win+N note is
        being recorded. Without this, a user still holding Ctrl+Win after
        triggering the note would keep retriggering dictation underneath."""
        self._note_in_progress = bool(on)
        if on and self._recording:
            self._recording = False

    def _poll(self) -> None:
        # Chords first: a note-toggle / Esc press must still be seen while a
        # note is in progress (the dictation poll below stands down then).
        if self._chords:
            self._poll_chords()
        if self._note_in_progress:
            # Stand down so the note recording isn't fighting a dictation
            # press/release in parallel. Treat keys as not-held so when the
            # note ends and the user has long since released Ctrl+Win, the
            # next press is a clean fresh transition.
            self._held_keys = False
            return
        held = self._all_held()
        # Detect raw key transitions
        key_press = held and not self._held_keys
        key_release = not held and self._held_keys
        self._held_keys = held

        if key_press and self._hook_in_use:
            # Cross-check the LL hook against this hook-free press detection
            # (fires once per hold — key_press is an edge, not a level).
            t_press = time.monotonic()
            QTimer.singleShot(
                1000, lambda: self._verify_hook_saw_press(t_press))

        if self._mode == "hold":
            # Recording follows key state 1:1
            if key_press:
                self._recording = True
                self.pressed.emit()
            elif key_release:
                if self._recording:
                    self._recording = False
                    self.released.emit()
        else:  # toggle
            # Each fresh KEY PRESS flips recording state
            if key_press:
                if not self._recording:
                    self._recording = True
                    self.pressed.emit()
                else:
                    self._recording = False
                    self.released.emit()
            # Key releases are ignored in toggle mode


# ===========================================================================
# overlay — tiny indicator at bottom-center of the primary monitor
# ===========================================================================
def read_communications_mic_mute() -> tuple[bool | None, str]:
    """Read the same Windows input endpoint controlled by IEM Control.

    Never infer mute from silence or a saved button press. Re-resolve the
    endpoint on each check to follow device changes and USB reconnects.
    Caller must initialize COM on its thread.
    """
    try:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities
        from pycaw.api.endpointvolume import IAudioEndpointVolume
        endpoint = AudioUtilities.GetDeviceEnumerator().GetDefaultAudioEndpoint(1, 2)
        device = AudioUtilities.CreateDevice(endpoint)
        volume = endpoint.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return bool(volume.QueryInterface(IAudioEndpointVolume).GetMute()), device.FriendlyName
    except Exception:
        return None, "Default communications microphone unavailable"


class CommsMicProbe:
    """Cheap repeated reads of the communications mic's mute state.

    read_communications_mic_mute() walks the device enumerator and opens the
    endpoint's property store on every call — ~10 ms of CPU. The overlay polls
    2.5×/s for the life of the process, which made this the bulk of Tellur's
    idle CPU (and 10 ms GIL holds next to the 20 ms hotkey poll). GetMute()
    on a held IAudioEndpointVolume is microseconds, so keep the interface and
    re-resolve the default endpoint only every RESOLVE_EVERY_SEC (to follow a
    changed default device / USB reconnect) or after any failure.
    Single-threaded: create and use on one COM-initialized thread."""

    RESOLVE_EVERY_SEC = 3.0

    def __init__(self) -> None:
        self._volume = None
        self._name = "Default communications microphone unavailable"
        self._resolved_at = 0.0

    def _resolve(self) -> None:
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities
        from pycaw.api.endpointvolume import IAudioEndpointVolume
        endpoint = AudioUtilities.GetDeviceEnumerator().GetDefaultAudioEndpoint(1, 2)
        name = AudioUtilities.CreateDevice(endpoint).FriendlyName
        volume = endpoint.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        self._volume = volume.QueryInterface(IAudioEndpointVolume)
        self._name = name

    def read(self) -> tuple[bool | None, str]:
        now = time.monotonic()
        try:
            if (self._volume is None
                    or now - self._resolved_at >= self.RESOLVE_EVERY_SEC):
                self._resolved_at = now
                self._resolve()
            return bool(self._volume.GetMute()), self._name
        except Exception:
            self._volume = None
            return None, "Default communications microphone unavailable"


class Overlay(QWidget):
    mic_status_changed = pyqtSignal(object, str)
    MIC_STATUS_HEIGHT = 16
    MIC_STATUS_WIDTH = 16

    def __init__(self, monitor_microphone: bool = True):
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
        self.resize(max(OVERLAY_WIDTH, self.MIC_STATUS_WIDTH), OVERLAY_HEIGHT + self.MIC_STATUS_HEIGHT)
        self._mic_muted: bool | None = None
        self._mic_name = "Checking default communications microphone"
        self.mic_status_changed.connect(self._set_mic_status)
        if monitor_microphone:
            # Core Audio calls stay off the GUI thread. Qt queues the result.
            stop = threading.Event()
            self.destroyed.connect(lambda: stop.set())
            signal = self.mic_status_changed
            def watch_mic():
                import comtypes
                comtypes.CoInitialize()
                try:
                    probe = CommsMicProbe()
                    last = None
                    while not stop.is_set():
                        status = probe.read()
                        # Emit on change only: the slot ignored repeats
                        # anyway, and each emit is a queued cross-thread
                        # event on the GUI loop.
                        if status != last:
                            last = status
                            try:
                                signal.emit(*status)
                            except RuntimeError:
                                break  # widget was deleted during the query
                        stop.wait(0.4)
                    probe = None    # drop the COM pointer before CoUninitialize
                finally:
                    comtypes.CoUninitialize()
            threading.Thread(target=watch_mic, name="mic-mute-status", daemon=True).start()

        self._state = "idle"
        self._pulse_t = 0.0
        self._history: deque[float] = deque([0.0] * BAR_COUNT, maxlen=BAR_COUNT)
        self._scale = float(BAR_LEVEL_SCALE)
        # Engine badge: True while dictation is being served by local Whisper.
        # The pill grows upward and paints a label under the bars, so a silent
        # fallback from the cloud engine is visible while you're still talking.
        self._badge = False
        self._badge_font = QFont()
        self._badge_font.setPointSize(OVERLAY_BADGE_FONT_PT)
        self._badge_font.setBold(True)
        self._badge_font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.6)

        self._anim = QTimer(self)
        self._anim.timeout.connect(self._tick)
        self._anim.start(33)
        self._position()

        # Re-position if the user changes their primary monitor while the
        # app is running, so the overlay doesn't end up on a screen that
        # no longer exists.
        guiapp = QGuiApplication.instance()
        if guiapp is not None:
            # Bound method of this QObject (NOT a lambda): Qt drops the
            # connection when the overlay is destroyed. A lambda survived
            # "Refresh UI" and then called _position() on the deleted widget
            # at the next monitor change.
            guiapp.primaryScreenChanged.connect(self._on_primary_screen_changed)

    def _on_primary_screen_changed(self, _screen) -> None:
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

    def _content_height(self) -> int:
        """Height available to the bars/dot — the badge row is carved off the
        bottom, so the indicator sits directly under the level bars."""
        return self.height() - self.MIC_STATUS_HEIGHT - (OVERLAY_BADGE_HEIGHT if self._badge else 0)

    def _set_mic_status(self, muted: bool | None, name: str) -> None:
        if (muted, name) == (self._mic_muted, self._mic_name):
            return
        self._mic_muted, self._mic_name = muted, name
        state = "muted" if muted is True else "unmuted" if muted is False else "status unknown"
        self.setAccessibleName(f"{name}: {state}")
        self.setToolTip(f"Windows communications microphone: {name}\n{state}")
        self._reassert_topmost()
        self.update()

    def set_engine(self, engine: str) -> None:
        """Show/hide the local-Whisper badge. Only the literal "local" shows it;
        "elevenlabs", "pending", and anything else hide it.

        The badge is **confirmatory, never predictive**. It is cleared at record
        start ("pending") and only raised once the transcribe has returned and
        reported that local Whisper is what actually ran. Guessing up front from
        key state would raise it in cases where a key turns out to work — the
        badge would then be claiming something untrue. It stays up after a local
        dictation, as a notification, until the next time you press the hotkey.

        Live *before-you-talk* engine state lives in Settings → Transcription
        engine instead, where being wrong for a moment costs nothing."""
        badge = (engine or "").lower() == "local"
        if badge == self._badge:
            return
        self._badge = badge
        # Bottom edge stays put: _position() derives y from the new height, so
        # the pill grows upward and the bars shift up by the badge row.
        self.resize(
            max(self.MIC_STATUS_WIDTH, OVERLAY_BADGE_WIDTH if badge else OVERLAY_WIDTH),
            OVERLAY_HEIGHT + self.MIC_STATUS_HEIGHT + (OVERLAY_BADGE_HEIGHT if badge else 0),
        )
        self._position()
        self.update()

    def set_state(self, state: str) -> None:
        if state != self._state:
            self._state = state
            if state == "recording":
                self._history = deque([0.0] * BAR_COUNT, maxlen=BAR_COUNT)
                self._reassert_topmost()
            self.update()

    def _reassert_topmost(self) -> None:
        """Games re-claim the top of the z-order when they launch; shove the
        pill back above borderless-fullscreen windows each time recording
        starts. (Exclusive fullscreen still covers everything — no cure.)"""
        if not _WIN32:
            return
        try:
            HWND_TOPMOST = -1
            flags = 0x0001 | 0x0002 | 0x0010  # NOSIZE | NOMOVE | NOACTIVATE
            _user32.SetWindowPos(
                int(self.winId()), HWND_TOPMOST, 0, 0, 0, 0, flags)
        except Exception:
            pass

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
        if self._badge:
            self._draw_badge(p)
        self._draw_mic_status(p)

    def _draw_mic_status(self, p: QPainter) -> None:
        top = self.height() - self.MIC_STATUS_HEIGHT
        muted = self._mic_muted
        color = QColor("#ff6b6b" if muted is True else "#62d98b" if muted is False else "#e9bc72")
        p.save()
        p.translate(self.width() / 2, top)
        p.setPen(QPen(color, 1.5))
        p.setBrush(Qt.BrushStyle.NoBrush)
        # A compact, centered icon only; the window remains fully transparent.
        p.drawRoundedRect(QRectF(-2.5, 1, 5, 8), 2.5, 2.5)
        p.drawArc(QRectF(-5, 4, 10, 8), 180 * 16, 180 * 16)
        p.drawLine(0, 12, 0, 14)
        p.drawLine(-3, 14, 3, 14)
        if muted is True:
            p.drawLine(-6, 1, 6, 14)
        p.restore()

    def _draw_bars(self, p: QPainter) -> None:
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(240, 90, 90))
        total_w = BAR_COUNT * BAR_WIDTH + (BAR_COUNT - 1) * BAR_GAP
        x0 = (self.width() - total_w) // 2
        max_h = self._content_height()
        cy = max_h / 2
        for i, level in enumerate(self._history):
            scaled = min(level * self._scale, 1.0)
            h = max(BAR_MIN_HEIGHT, scaled * max_h)
            x = x0 + i * (BAR_WIDTH + BAR_GAP)
            y = cy - h / 2
            p.drawRoundedRect(QRectF(x, y, BAR_WIDTH, h), BAR_WIDTH / 2, BAR_WIDTH / 2)

    def _draw_dot(self, p: QPainter) -> None:
        color, radius = self._dot_style()
        cx = self.width() / 2
        cy = self._content_height() / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(color)
        p.drawEllipse(QRectF(cx - radius, cy - radius, radius * 2, radius * 2))

    def _draw_badge(self, p: QPainter) -> None:
        """"WHISPER" in amber directly beneath the level bars — the tell that
        this dictation is NOT going through the cloud engine."""
        p.setFont(self._badge_font)
        p.setPen(QColor(*OVERLAY_BADGE_COLOR))
        p.drawText(
            QRectF(0, self._content_height(), self.width(), OVERLAY_BADGE_HEIGHT),
            int(Qt.AlignmentFlag.AlignCenter),
            OVERLAY_BADGE_TEXT,
        )

    def _dot_style(self) -> tuple[QColor, float]:
        if self._state == "transcribing":
            r = 2.4 + (sin(self._pulse_t * 7) + 1) * 1.1
            return QColor(96, 168, 255), r
        if self._state == "done":
            return QColor(96, 200, 138), 2.6
        if self._state == "error":
            return QColor(230, 160, 80), 2.6
        # "empty" — Whisper returned no text (silent / too-short / VAD-rejected).
        # Distinct orange so the user can tell "we tried, got nothing" apart
        # from "pasted successfully into a window you're not looking at".
        # Purely a render state — no audio-pipeline side effects.
        if self._state == "empty":
            return QColor(255, 165, 0), 2.8
        return QColor(140, 144, 156, 170), 1.8


# ===========================================================================
# auto-update — one-click in-app upgrade from GitHub Releases
# ===========================================================================
log_update = logging.getLogger("update")


def _is_newer_version(remote: str, local: str) -> bool:
    """True if `remote` represents a strictly newer version than `local`.
    Uses packaging.version so pre-release suffixes (1.2.0-rc1) sort correctly
    BELOW the matching release. Falls back to a naive tuple comparison if
    packaging.version is unavailable for any reason."""
    remote = (remote or "").lstrip("v")
    local = (local or "").lstrip("v")
    try:
        from packaging.version import Version
        return Version(remote) > Version(local)
    except Exception:
        def _parts(v: str) -> tuple:
            out: list[int] = []
            for chunk in v.split("."):
                digits = "".join(c for c in chunk if c.isdigit())
                out.append(int(digits) if digits else 0)
            return tuple(out)
        return _parts(remote) > _parts(local)


def _pid_alive(pid: int) -> bool:
    """Return True iff a process with the given PID is currently running.
    Used by the update helper to know when the old app has fully exited.

    Uses WaitForSingleObject(handle, 0) on Windows rather than GetExitCodeProcess,
    because the latter returns STILL_ACTIVE (259) — which is also a legitimate
    exit code, so a process that exited with code 259 would look "alive".
    """
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    try:
        import ctypes
    except Exception:
        return False
    PROCESS_SYNCHRONIZE = 0x00100000
    WAIT_OBJECT_0 = 0   # signaled — process has exited
    WAIT_TIMEOUT = 0x102
    h = ctypes.windll.kernel32.OpenProcess(PROCESS_SYNCHRONIZE, False, pid)
    if not h:
        # OpenProcess fails when the PID doesn't exist OR when we lack
        # permissions; treat both as "not alive" (correct for our parent).
        return False
    try:
        result = ctypes.windll.kernel32.WaitForSingleObject(h, 0)
        return result == WAIT_TIMEOUT  # still running == not signaled
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
    if not _is_newer_version(tag, __version__):
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


def _download_file(url: str, dest: Path, *, timeout: float,
                   max_bytes: int) -> None:
    """Stream `url` to `dest` with a per-read socket timeout and a hard size
    cap (enforced WHILE downloading, so a mis-published asset can't fill the
    disk before the post-download check)."""
    req = urllib.request.Request(url, headers={"User-Agent": f"Tellur/{__version__}"})
    total = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise OSError(
                    f"archive exceeds the {max_bytes // (1024 * 1024)} MB cap")
            out.write(chunk)


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


# Relative paths in the install dir we should NOT overwrite during update —
# user may have customized them. Matched against the POSIX-style relative
# path from the unpacked archive root, not by basename, so we don't
# accidentally preserve a same-named file in some future subdirectory.
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

        # 2b) If the archive has a single top-level directory (e.g. a GitHub
        # "Source code (zip)" download wraps everything in `tellur-1.2.2/`),
        # treat that as the real source root so we don't deposit a folder
        # inside install_dir.
        source_root = staging_dir
        top_items = [p for p in staging_dir.iterdir()]
        if (
            len(top_items) == 1
            and top_items[0].is_dir()
            and top_items[0].name.lower().startswith(("tellur", APP_NAME.lower()))
        ):
            source_root = top_items[0]
            w(f"detected wrapper dir; using {source_root.name} as source root")

        # 3) copy files over the install dir, skipping preserved user files.
        copied = 0
        skipped = 0
        for src in source_root.rglob("*"):
            if not src.is_file():
                continue
            rel = src.relative_to(source_root)
            rel_posix = rel.as_posix()
            dst = install_dir / rel
            if rel_posix in _PRESERVE_FILES and dst.exists():
                w(f"preserved: {rel_posix}")
                skipped += 1
                continue
            try:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
            except OSError as e:
                w(f"copy failed for {rel_posix}: {e}")
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
    quit_requested = pyqtSignal()      # post-install restart → App.quit

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
            # daemon: a stalled download must not pin the process open after
            # Quit (no tray, no hotkeys, still running).
            daemon=True, name="update-install",
        ).start()

    def _install_worker(self, info: dict) -> None:
        try:
            # Refuse oversized release assets — sane defense even though
            # we're downloading from a trusted repo, in case a release is
            # ever misconfigured or a CDN serves a wrong file.
            claimed_bytes = int(info.get("zip_size") or 0)
            if claimed_bytes > UPDATE_MAX_ZIP_BYTES:
                self.install_failed.emit(
                    f"Update archive too large "
                    f"({claimed_bytes // (1024 * 1024)} MB > "
                    f"{UPDATE_MAX_ZIP_BYTES // (1024 * 1024)} MB cap). "
                    f"Refusing to download."
                )
                return
            self.state_changed.emit(f"Downloading v{info['version']}…")
            tmpdir = Path(tempfile.mkdtemp(prefix="tellur_dl_"))
            zip_path = tmpdir / f"Tellur-{info['version']}.zip"
            try:
                _download_file(info["zip_url"], zip_path,
                               timeout=UPDATE_DOWNLOAD_TIMEOUT_SEC,
                               max_bytes=UPDATE_MAX_ZIP_BYTES)
            except Exception as e:
                log_update.exception("download failed")
                shutil.rmtree(tmpdir, ignore_errors=True)
                self.install_failed.emit(f"Download failed: {e}")
                return

            # quick integrity check
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    bad = zf.testzip()
                    if bad is not None:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                        self.install_failed.emit(f"Downloaded archive is corrupt ({bad})")
                        return
            except zipfile.BadZipFile:
                shutil.rmtree(tmpdir, ignore_errors=True)
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
            # Give the helper a beat to start polling our PID, then quit —
            # through App.quit (tray hidden, passthrough closed, volumes
            # restored), not a bare QApplication.quit.
            QTimer.singleShot(400, self.quit_requested.emit)
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
    refresh_ui_requested = pyqtSignal()

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

        refresh_act = QAction("Refresh UI", self)
        refresh_act.setToolTip("Re-create the overlay and reset stuck UI state")
        refresh_act.triggered.connect(self.refresh_ui_requested)
        menu.addAction(refresh_act)

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
# Theme palettes — every QSS color is sourced from here so the light / dark
# stylesheets are produced from the same template with different inputs.
# Keys are stable; values are CSS color literals.
THEME_PALETTES: dict[str, dict[str, str]] = {
    "dark": {
        "bg":              "#1a1b1f",
        "bg_alt":          "#15161a",
        "border":          "#2a2b30",
        "border_strong":   "#3a3b42",
        "row_hover":       "#20212a",
        "row_selected":    "#2c3a55",
        "row_separator":   "#22232a",
        "text":            "#e6e8ee",
        "text_strong":     "#ffffff",
        "text_dim":        "#9a9da6",
        "text_muted":      "#6a6e78",
        "text_form_label": "#c0c4cc",
        "text_header":     "#d8dbe2",
        "header_underline":"#2b2e36",
        "btn_bg":          "#2a2b30",
        "btn_hover":       "#34353c",
        "btn_pressed":     "#4a4b54",
        "accent":          "#6080ff",
        "accent_hover":    "#7a98ff",
        "accent_selection":"#5078ff",
        "danger":          "#e07c7c",
        "success":         "#6bd47b",
        "icon_btn_hover":  "#2c2d34",
        "tab_hover":       "#ccd0d8",
    },
    "light": {
        "bg":              "#f7f7fa",
        "bg_alt":          "#ffffff",
        "border":          "#d6d8dc",
        "border_strong":   "#b8bcc4",
        "row_hover":       "#eef0f6",
        "row_selected":    "#cdd9ff",
        "row_separator":   "#e4e6ea",
        "text":            "#1a1b21",
        "text_strong":     "#000000",
        "text_dim":        "#5a5e68",
        "text_muted":      "#7a7e88",
        "text_form_label": "#33363c",
        "text_header":     "#1a1b21",
        "header_underline":"#d6d8dc",
        "btn_bg":          "#ffffff",
        "btn_hover":       "#eef0f6",
        "btn_pressed":     "#d9dde4",
        "accent":          "#3057d8",
        "accent_hover":    "#4068e8",
        "accent_selection":"#3057d8",
        "danger":          "#c44a4a",
        "success":         "#2c8c3b",
        "icon_btn_hover":  "#dde0e6",
        "tab_hover":       "#3a3d44",
    },
}


def build_panel_qss(theme: str) -> str:
    """Render the panel stylesheet for the named theme. Driven entirely by
    the THEME_PALETTES dict so adding a new theme is a one-entry addition."""
    p = THEME_PALETTES.get(theme) or THEME_PALETTES["dark"]
    return f"""
QWidget#mainPanel {{
    background-color: {p['bg']};
    color: {p['text']};
    font-family: "Segoe UI";
    font-size: 10pt;
}}
/* Inner containers need explicit backgrounds — plain QWidgets fall back
   to the system palette otherwise, which leaves a dark inner area when
   the light theme is active. */
QWidget#mainPanel QWidget#settingsContent {{
    background-color: {p['bg']};
}}
QWidget#mainPanel QScrollArea {{
    background-color: {p['bg']};
    border: none;
}}
QWidget#mainPanel QScrollArea > QWidget > QWidget {{
    background-color: {p['bg']};
}}
QWidget#mainPanel QLabel {{ color: {p['text']}; background: transparent; }}
QWidget#mainPanel QLabel#sectionHeader {{
    font-weight: 600;
    color: {p['text_header']};
    font-size: 10.5pt;
    padding-top: 4px;
    padding-bottom: 2px;
    border-bottom: 1px solid {p['header_underline']};
}}
QWidget#mainPanel QLabel#sectionHint {{
    color: {p['text_muted']};
    font-size: 9pt;
}}
QWidget#mainPanel QLabel#formLabel {{
    color: {p['text_form_label']};
}}
QWidget#mainPanel QLabel#valueDim {{
    color: {p['text_dim']};
}}
QWidget#mainPanel QLabel#footerLabel {{
    color: {p['text_muted']};
    font-size: 9pt;
}}
QWidget#mainPanel QListWidget {{
    background-color: {p['bg_alt']};
    border: 1px solid {p['border']};
    border-radius: 4px;
    outline: 0;
}}
QWidget#mainPanel QListWidget::item {{
    padding: 0;
    border: none;
}}
QWidget#mainPanel QListWidget::item:hover {{ background-color: {p['row_hover']}; }}
QWidget#mainPanel QListWidget::item:selected {{
    background-color: {p['row_selected']};
    color: {p['text_strong']};
}}
QWidget#mainPanel QPushButton {{
    background-color: {p['btn_bg']};
    color: {p['text']};
    border: 1px solid {p['border_strong']};
    border-radius: 4px;
    padding: 6px 14px;
    min-width: 70px;
}}
QWidget#mainPanel QPushButton:hover {{ background-color: {p['btn_hover']}; }}
QWidget#mainPanel QPushButton:pressed {{ background-color: {p['btn_pressed']}; }}
QWidget#mainPanel QPushButton#dangerButton {{ color: {p['danger']}; }}
QWidget#mainPanel QTextEdit {{
    background-color: {p['bg_alt']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 4px;
    padding: 8px;
    selection-background-color: {p['accent_selection']};
    selection-color: {p['text_strong']};
}}
QWidget#mainPanel QTabWidget::pane {{
    border: none;
    background-color: {p['bg']};
}}
QWidget#mainPanel QTabBar::tab {{
    background-color: transparent;
    color: {p['text_dim']};
    padding: 8px 18px;
    border: none;
}}
QWidget#mainPanel QTabBar::tab:selected {{
    color: {p['text']};
    border-bottom: 2px solid {p['accent']};
}}
QWidget#mainPanel QTabBar::tab:hover:!selected {{ color: {p['tab_hover']}; }}
QWidget#mainPanel QCheckBox {{ color: {p['text']}; spacing: 8px; }}
QWidget#mainPanel QSlider::groove:horizontal {{
    background: {p['border']}; height: 4px; border-radius: 2px;
}}
QWidget#mainPanel QSlider::handle:horizontal {{
    background: {p['accent']}; width: 14px; margin: -5px 0; border-radius: 7px;
}}
QWidget#mainPanel QSlider::handle:horizontal:hover {{ background: {p['accent_hover']}; }}
QWidget#mainPanel QComboBox {{
    background-color: {p['bg_alt']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 4px;
    padding: 4px 8px;
    selection-background-color: {p['row_selected']};
}}
QWidget#mainPanel QComboBox:hover {{ border: 1px solid {p['border_strong']}; }}
QWidget#mainPanel QComboBox::drop-down {{ border: none; width: 22px; }}
QWidget#mainPanel QComboBox QAbstractItemView {{
    background-color: {p['bg_alt']};
    color: {p['text']};
    border: 1px solid {p['border']};
    selection-background-color: {p['row_selected']};
    selection-color: {p['text_strong']};
    outline: 0;
}}
QWidget#mainPanel QProgressBar {{
    background-color: {p['bg_alt']};
    border: 1px solid {p['border']};
    border-radius: 4px;
    text-align: center;
    color: {p['text']};
    font-size: 9pt;
}}
QWidget#mainPanel QProgressBar::chunk {{
    background-color: {p['accent']};
    border-radius: 3px;
}}
QWidget#mainPanel QLineEdit {{
    background-color: {p['bg_alt']};
    color: {p['text']};
    border: 1px solid {p['border']};
    border-radius: 4px;
    padding: 6px 10px;
    selection-background-color: {p['row_selected']};
}}
QWidget#mainPanel QLineEdit:focus {{ border: 1px solid {p['accent']}; }}
QWidget#historyRow {{
    background-color: transparent;
    border-bottom: 1px solid {p['row_separator']};
}}
QWidget#historyRow QLabel#historyRowText {{
    color: {p['text']};
    font-size: 10pt;
    background: transparent;
}}
QWidget#historyRow QLabel#historyRowMeta {{
    color: {p['text_muted']};
    font-size: 9pt;
    background: transparent;
}}
QWidget#noteRow {{
    background-color: transparent;
    border-bottom: 1px solid {p['row_separator']};
}}
QWidget#noteRow QLabel#historyRowText {{
    color: {p['text']};
    font-size: 10pt;
    background: transparent;
}}
QWidget#noteRow QLabel#historyRowMeta {{
    color: {p['text_muted']};
    font-size: 9pt;
    background: transparent;
}}
QWidget#noteRow QLabel#noteAppChip {{
    color: {p['text_dim']};
    font-size: 8.5pt;
    background: transparent;
    padding: 1px 6px;
    border: 1px solid {p['border']};
    border-radius: 7px;
}}
QWidget#noteRow QLabel#noteFlagChip {{
    font-size: 8.5pt;
    padding: 1px 6px;
    border-radius: 7px;
    color: white;
    font-weight: 600;
}}
QWidget#noteRow QLabel#noteFlagChip[flag="critical"]  {{ background-color: {p['danger']}; }}
QWidget#noteRow QLabel#noteFlagChip[flag="important"] {{ background-color: {p['accent']}; }}
QWidget#noteRow QLabel#noteFlagChip[flag="followup"]  {{ background-color: {p['success']}; }}
QWidget#noteRow QLabel#noteFlagChip[flag="random"]    {{ background-color: {p['text_muted']}; }}
QPushButton#iconButton {{
    background: transparent;
    border: none;
    border-radius: 4px;
    padding: 0;
    margin: 0;
    min-width: 0;
}}
QPushButton#iconButton:hover {{
    background-color: {p['icon_btn_hover']};
}}
"""


# Back-compat alias — old code paths that referenced PANEL_QSS still work.
PANEL_QSS = build_panel_qss("dark")


class ElidedLabel(QLabel):
    """QLabel that ellipsizes on the right when its rendered width is less
    than the full text's width. Qt's default QLabel hard-clips with no
    indicator, which made the history rows ugly under tight space."""

    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self._full_text = text
        self.setWordWrap(False)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setMinimumWidth(40)
        self._render()

    def setText(self, text: str) -> None:
        self._full_text = text
        self._render()

    def fullText(self) -> str:
        return self._full_text

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    def _render(self) -> None:
        fm = self.fontMetrics()
        avail = max(self.width() - 4, 0)
        elided = fm.elidedText(self._full_text, Qt.TextElideMode.ElideRight, avail)
        super().setText(elided)


class IconButton(QToolButton):
    """Compact flat icon button with a custom-painted glyph. QToolButton +
    autoRaise gives true zero-chrome rendering until hover, unlike
    QPushButton.setFlat which still draws a frame on some Windows styles."""

    PIN_OFF = "pin_off"
    PIN_ON = "pin_on"
    COPY = "copy"
    DELETE = "delete"

    SIZE = 22

    def __init__(self, icon_type: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._icon_type = icon_type
        self.setAutoRaise(True)
        self.setFixedSize(self.SIZE, self.SIZE)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setObjectName("iconButton")
        # Force background transparency regardless of platform style.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setStyleSheet("border: none; background: transparent;")

    def set_icon_type(self, icon_type: str) -> None:
        self._icon_type = icon_type
        self.update()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        hovered = self.underMouse()
        if self._icon_type == IconButton.PIN_ON:
            base = QColor(244, 197, 66)  # gold for an active pin
        elif hovered:
            base = QColor(232, 234, 240)
        else:
            base = QColor(150, 155, 165)

        r = self.rect()
        cx, cy = r.center().x(), r.center().y()
        if self._icon_type in (IconButton.PIN_OFF, IconButton.PIN_ON):
            self._paint_star(p, cx, cy, base, filled=(self._icon_type == IconButton.PIN_ON))
        elif self._icon_type == IconButton.COPY:
            self._paint_copy(p, cx, cy, base)
        elif self._icon_type == IconButton.DELETE:
            self._paint_delete(p, cx, cy, base)
        p.end()

    @staticmethod
    def _paint_star(p: QPainter, cx: int, cy: int, color: QColor, filled: bool) -> None:
        from math import cos, sin, pi
        from PyQt6.QtGui import QPen
        outer = 6.0
        inner = 2.6
        points = []
        for i in range(10):
            angle = -pi / 2 + i * pi / 5
            r = outer if i % 2 == 0 else inner
            points.append((cx + r * cos(angle), cy + r * sin(angle)))
        path = QPainterPath()
        path.moveTo(*points[0])
        for x, y in points[1:]:
            path.lineTo(x, y)
        path.closeSubpath()
        if filled:
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(color)
            p.drawPath(path)
        else:
            pen = QPen(color, 1.3)
            p.setPen(pen)
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawPath(path)

    @staticmethod
    def _paint_copy(p: QPainter, cx: int, cy: int, color: QColor) -> None:
        from PyQt6.QtGui import QPen
        pen = QPen(color, 1.3)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        p.setPen(pen)
        p.setBrush(Qt.BrushStyle.NoBrush)
        # Back doc: up-left
        p.drawRoundedRect(QRectF(cx - 5, cy - 6, 7, 9), 1.3, 1.3)
        # Front doc: down-right, painted opaque to hide the overlap.
        # Use the panel background colour for fill so the back doc's
        # corner peeks out from behind, giving the "two stacked" feel.
        p.setBrush(QColor(26, 27, 31))
        p.drawRoundedRect(QRectF(cx - 2, cy - 3, 7, 9), 1.3, 1.3)
        # Restroke the front doc outline since the fill overdrew the stroke.
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRoundedRect(QRectF(cx - 2, cy - 3, 7, 9), 1.3, 1.3)

    @staticmethod
    def _paint_delete(p: QPainter, cx: int, cy: int, color: QColor) -> None:
        from PyQt6.QtGui import QPen
        pen = QPen(color, 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        p.setPen(pen)
        s = 4.5
        p.drawLine(int(cx - s), int(cy - s), int(cx + s), int(cy + s))
        p.drawLine(int(cx + s), int(cy - s), int(cx - s), int(cy + s))


class HistoryRow(QWidget):
    """Custom row widget for the History list. Shows ellipsized text +
    meta line + a single Copy icon button on the right.

    Pinning and per-row delete were intentionally removed: pinning added
    visual clutter for marginal value, and per-row delete buttons are
    a misclick hazard. Delete now lives in the History tab's top toolbar
    and confirms before destroying anything.

    Double-click on the text starts inline edit.
    """

    copy_requested = pyqtSignal(dict)
    edit_started = pyqtSignal(dict)

    def __init__(self, entry: dict):
        super().__init__()
        self.entry = entry
        self.setObjectName("historyRow")
        # Plain QWidget ignores QSS background/border by default; enable so
        # the row separator and any future row-level styling actually paint.
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        # Fix row height so text + meta + button share the same vertical
        # center.
        self.setFixedHeight(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 14, 0)   # 14px right inset, not at the edge
        layout.setSpacing(10)

        # Left: ellipsized text preview, vertically centered.
        self.text_label = ElidedLabel(self._full_text())
        self.text_label.setObjectName("historyRowText")
        self.text_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        layout.addWidget(self.text_label, stretch=1)

        # Meta column: fixed minimum width, right-aligned + vertically
        # centered, so the copy button lines up across rows regardless of
        # meta text length.
        self._meta_label = QLabel(self._meta_text())
        self._meta_label.setObjectName("historyRowMeta")
        self._meta_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._meta_label.setMinimumWidth(110)
        layout.addWidget(self._meta_label)

        # Single copy button on the right.
        copy_btn = IconButton(IconButton.COPY)
        copy_btn.setToolTip("Copy this transcription")
        copy_btn.clicked.connect(lambda: self.copy_requested.emit(self.entry))
        layout.addWidget(copy_btn)

    def _full_text(self) -> str:
        # Newlines flattened to spaces; ElidedLabel handles truncation.
        return (self.entry.get("text") or "").replace("\n", " ")

    def _meta_text(self) -> str:
        text = self.entry.get("text") or ""
        word_count = len(text.split())
        when = humanize_time(self.entry.get("ts", 0))
        edited = "  ·  edited" if self.entry.get("edited") else ""
        return f"{word_count} word{'' if word_count == 1 else 's'}  ·  {when}{edited}"

    def refresh(self) -> None:
        """Re-render based on possibly-mutated entry dict (e.g. after inline edit)."""
        self.text_label.setText(self._full_text())
        self._meta_label.setText(self._meta_text())

    def mouseDoubleClickEvent(self, event) -> None:
        # Forward to text-label area only: double-click on body starts edit.
        if self.text_label.geometry().contains(event.position().toPoint()):
            self.edit_started.emit(self.entry)
            return
        super().mouseDoubleClickEvent(event)


class NoteRow(QWidget):
    """Custom row widget for the Notes list. Same shape as HistoryRow but
    shows two trailing chips: an auto-flag (foreground app at capture time)
    and an optional manual flag (critical/important/etc.). Both chips render
    inline via QSS object names so the theme palette controls their color.
    """

    copy_requested = pyqtSignal(dict)

    def __init__(self, entry: dict):
        super().__init__()
        self.entry = entry
        self.setObjectName("noteRow")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.setFixedHeight(34)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 14, 0)
        layout.setSpacing(10)

        self.text_label = ElidedLabel(self._full_text())
        self.text_label.setObjectName("historyRowText")
        self.text_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
        )
        layout.addWidget(self.text_label, stretch=1)

        # App-flag chip — always present, dim/neutral color.
        self._app_chip = QLabel(self._app_text())
        self._app_chip.setObjectName("noteAppChip")
        self._app_chip.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._app_chip.setMinimumWidth(64)
        layout.addWidget(self._app_chip)

        # Manual flag chip — only visible when a flag is set; color follows
        # the flag (set via dynamic property so QSS can theme it).
        self._flag_chip = QLabel("")
        self._flag_chip.setObjectName("noteFlagChip")
        self._flag_chip.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._flag_chip.setMinimumWidth(0)
        layout.addWidget(self._flag_chip)

        self._meta_label = QLabel(self._meta_text())
        self._meta_label.setObjectName("historyRowMeta")
        self._meta_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        self._meta_label.setMinimumWidth(90)
        layout.addWidget(self._meta_label)

        copy_btn = IconButton(IconButton.COPY)
        copy_btn.setToolTip("Copy this note")
        copy_btn.clicked.connect(lambda: self.copy_requested.emit(self.entry))
        layout.addWidget(copy_btn)

        self._apply_flag_style()

    def _full_text(self) -> str:
        return (self.entry.get("text") or "").replace("\n", " ")

    def _app_text(self) -> str:
        app = (self.entry.get("app") or "").strip()
        return app or "—"

    def _meta_text(self) -> str:
        text = self.entry.get("text") or ""
        word_count = len(text.split())
        when = humanize_time(self.entry.get("ts", 0))
        return f"{word_count} word{'' if word_count == 1 else 's'}  ·  {when}"

    def _apply_flag_style(self) -> None:
        flag = (self.entry.get("flag") or "").strip().lower()
        if flag:
            self._flag_chip.setText(flag)
            self._flag_chip.setVisible(True)
        else:
            self._flag_chip.setText("")
            self._flag_chip.setVisible(False)
        # Drive QSS via dynamic property so theme palettes can color per-flag.
        self._flag_chip.setProperty("flag", flag or "none")
        # Force re-polish so the property change takes effect.
        self._flag_chip.style().unpolish(self._flag_chip)
        self._flag_chip.style().polish(self._flag_chip)

    def refresh(self) -> None:
        self.text_label.setText(self._full_text())
        self._app_chip.setText(self._app_text())
        self._meta_label.setText(self._meta_text())
        self._apply_flag_style()


class HistoryList(QListWidget):
    """QListWidget subclass that constrains every row widget to the
    viewport's width.

    Without this, items installed via `setItemWidget` render at their
    `sizeHint().width()`, which for a long-text row is much larger than
    the panel. Result: the meta column and action buttons get pushed
    past the right edge and become invisible (you'd see a horizontal
    scrollbar). Re-fitting on every resize keeps everything visible and
    forces the text label to ellipsize correctly.
    """

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.refit_rows()

    def refit_rows(self) -> None:
        w = self.viewport().width()
        if w <= 0:
            return
        for i in range(self.count()):
            item = self.item(i)
            widget = self.itemWidget(item)
            if widget is None:
                continue
            h = widget.height() if widget.height() > 0 else 34
            item.setSizeHint(QSize(w, h))
            widget.setFixedWidth(w)


def keyseq_to_combo(seq: "QKeySequence") -> str:
    """Translate a Qt key sequence (e.g. 'Ctrl+Meta+Z') to the format the
    `keyboard` library uses ('ctrl+windows+z'). Empty sequence → ''."""
    if seq.isEmpty():
        return ""
    s = seq.toString().strip().lower()
    # Qt names the Windows key 'meta'; keyboard library calls it 'windows'.
    parts = [("windows" if p == "meta" else p) for p in s.split("+")]
    return "+".join(parts)


def combo_to_keyseq(combo: str) -> "QKeySequence":
    """Inverse of keyseq_to_combo — turn 'ctrl+windows+z' into a
    QKeySequence Qt can render in a QKeySequenceEdit."""
    combo = (combo or "").strip().lower()
    if not combo:
        return QKeySequence()
    parts = []
    for p in combo.split("+"):
        p = p.strip()
        if not p:
            continue
        if p == "windows":
            parts.append("Meta")
        else:
            parts.append(p.capitalize())
    return QKeySequence("+".join(parts))


class _NoWheelComboBox(QComboBox):
    """QComboBox that doesn't consume mouse wheel events. Wheel events get
    ignore()'d so the parent scroll area handles them — otherwise scrolling
    through a settings page can silently change combo selections when the
    cursor passes over them."""
    def wheelEvent(self, event) -> None:  # noqa: N802 — Qt signature
        event.ignore()


class _NoWheelSlider(QSlider):
    """QSlider that doesn't consume mouse wheel events (same rationale as
    _NoWheelComboBox)."""
    def wheelEvent(self, event) -> None:  # noqa: N802 — Qt signature
        event.ignore()


class MainPanel(QWidget):
    """The 'open from the tray' window. Tabs: History, Notes, Settings."""

    model_switch_requested = pyqtSignal(str)
    refresh_ui_requested = pyqtSignal()
    # Test-button results are produced on worker threads; these deliver them to
    # the GUI thread. (QTimer.singleShot does NOT fire from a thread that has no
    # event loop, so a worker cannot update widgets directly or via singleShot.)
    _eleven_test_done = pyqtSignal(bool, str)
    _llm_test_done = pyqtSignal(bool, str)

    def __init__(
        self,
        log: TranscriptLog,
        notes: "NotesLog",
        settings: Settings,
        updater: "Updater | None" = None,
        recorder: "AudioRecorder | None" = None,
        passthrough: "PassthroughMonitor | None" = None,
        parent: QObject | None = None,
    ):
        super().__init__()
        self._log = log
        self._notes = notes
        self._settings = settings
        self._updater = updater
        self._recorder = recorder
        self._passthrough = passthrough
        # Worker threads can't touch widgets directly — route test results here
        # (cross-thread signal emit → queued slot on the GUI thread).
        self._eleven_test_done.connect(self._set_eleven_test_result)
        self._llm_test_done.connect(self._set_llm_test_result)

        self.setObjectName("mainPanel")
        self.setWindowTitle(APP_NAME)
        if ICON_PATH.exists():
            self.setWindowIcon(QIcon(str(ICON_PATH)))
        self._current_theme = ""  # will be set by _apply_theme
        self._apply_theme(self._settings.theme)
        # React to live theme changes (Settings → Appearance dropdown), but
        # defer the heavy restyle to the next event-loop tick. Without the
        # defer, picking "Light" from the Theme combo would interrupt Qt's
        # own click-processing on that combo — the first click silently
        # got eaten and the user needed a second click for it to register.
        # By the time the QTimer fires, the combo has finished its event.
        self._settings.changed.connect(
            lambda: QTimer.singleShot(
                0, lambda: self._apply_theme(self._settings.theme)
            )
        )
        # Fixed default size — tabs are independently sized; Settings scrolls
        # internally so it never forces the window taller. Width gives the
        # Settings content enough breathing room that no horizontal
        # scrollbar appears (which would otherwise trigger when Qt
        # auto-scrolls a freshly-focused button into view).
        self.resize(680, 580)

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(0)

        tabs = QTabWidget()
        tabs.addTab(self._build_history_tab(), "History")
        tabs.addTab(self._build_notes_tab(), "Notes")
        # Settings tab is wrapped in a QScrollArea (built inside the helper)
        # so the window doesn't have to grow to fit every section.
        tabs.addTab(self._build_settings_tab(), "Settings")
        # Default to History tab on open — it's the more common use case.
        tabs.setCurrentIndex(0)
        root.addWidget(tabs)

        # subscribe to log changes
        log.entry_added.connect(self._on_entry_added)
        log.entry_changed.connect(self._on_entry_changed)
        log.entry_removed.connect(self._on_entry_removed)
        log.cleared.connect(self._refresh_list)
        self._refresh_list()

        # subscribe to notes-log changes
        notes.entry_added.connect(self._on_note_added)
        notes.entry_changed.connect(self._on_note_changed)
        notes.entry_removed.connect(self._on_note_removed)
        notes.cleared.connect(self._refresh_notes_list)
        self._refresh_notes_list()

        # refresh time labels every 30s so "just now" rolls to "1m ago" etc.
        self._tick = QTimer(self)
        self._tick.setInterval(30_000)
        self._tick.timeout.connect(self._refresh_visible_timestamps)
        self._tick.start()

        # Live mic meter — only ticks while the panel is visible (showEvent
        # starts it, hideEvent stops it) so we don't keep an audio stream
        # open in the background.
        self._meter_tick = QTimer(self)
        self._meter_tick.setInterval(int(1000 / LEVEL_PUSH_HZ))
        self._meter_tick.timeout.connect(self._tick_level_meter)

    # --- history tab ----------------------------------------------------

    def _build_history_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(8)

        # Top toolbar: [ Delete selected ]  [ Search ............... ]
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        self._delete_selected_btn = QPushButton("Delete selected")
        self._delete_selected_btn.setToolTip("Delete the currently selected transcription (confirms first)")
        self._delete_selected_btn.setEnabled(False)
        self._delete_selected_btn.clicked.connect(self._delete_selected_confirm)
        top_bar.addWidget(self._delete_selected_btn)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search transcripts…  (Ctrl+F)")
        self._search.textChanged.connect(lambda _s: self._refresh_list())
        self._search.setClearButtonEnabled(True)
        top_bar.addWidget(self._search, stretch=1)
        layout.addLayout(top_bar)

        # The list itself. Items each get a HistoryRow widget installed via
        # setItemWidget. Custom HistoryList resizes row widgets to viewport.
        self._list = HistoryList()
        self._list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._list.itemSelectionChanged.connect(self._on_selection)
        self._list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._list.setUniformItemSizes(True)
        self._list.installEventFilter(self)
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

    # --- list construction & refresh -----------------------------------

    def _sorted_filtered_entries(self) -> list[dict]:
        """Apply search filter and sort newest-first."""
        query = self._search.text().strip().lower() if hasattr(self, "_search") else ""
        all_entries = self._log.entries()
        if query:
            all_entries = [e for e in all_entries if query in (e.get("text") or "").lower()]
        # _entries is in append order (oldest → newest); reverse for display.
        return list(reversed(all_entries))

    def _install_row(self, item: "QListWidgetItem", entry: dict) -> None:
        """Build a HistoryRow widget for an entry and install it on the list item.
        Row width is set to viewport width here AND on every list resize so the
        meta column and action button stay visible at the right edge."""
        row = HistoryRow(entry)
        row.copy_requested.connect(self._on_row_copy)
        row.edit_started.connect(self._on_row_edit_start)
        item.setData(Qt.ItemDataRole.UserRole, entry)
        # Use the current viewport width if known; HistoryList.resizeEvent
        # will re-fit on every resize anyway.
        w = self._list.viewport().width() if self._list.viewport().width() > 0 else 580
        item.setSizeHint(QSize(w, 34))
        row.setFixedWidth(w)
        self._list.setItemWidget(item, row)

    def _refresh_list(self) -> None:
        self._list.clear()
        for entry in self._sorted_filtered_entries():
            item = QListWidgetItem()
            self._list.addItem(item)
            self._install_row(item, entry)
        # Refit immediately after population so initial render uses the
        # actual viewport width (which may be larger than our 580 default).
        self._list.refit_rows()
        if self._list.count():
            self._list.setCurrentRow(0)
        else:
            self._detail.setPlainText("")

    def _refresh_visible_timestamps(self) -> None:
        for i in range(self._list.count()):
            item = self._list.item(i)
            widget = self._list.itemWidget(item)
            if isinstance(widget, HistoryRow):
                widget.refresh()
        if hasattr(self, "_notes_list"):
            for i in range(self._notes_list.count()):
                item = self._notes_list.item(i)
                widget = self._notes_list.itemWidget(item)
                if isinstance(widget, NoteRow):
                    widget.refresh()

    def _find_row(self, entry: dict) -> "tuple[QListWidgetItem | None, HistoryRow | None]":
        for i in range(self._list.count()):
            item = self._list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is entry:
                return item, self._list.itemWidget(item)
        return None, None

    def _on_entry_added(self, entry: dict) -> None:
        # Re-sort on add: pinned-first ordering means insert is non-trivial.
        # Cheap to just rebuild for our cap of 500.
        self._refresh_list()

    def _on_entry_changed(self, entry: dict) -> None:
        # Pin toggle changes the row's group; re-sort the whole list.
        self._refresh_list()
        # Try to re-select the same entry so the detail pane keeps showing it.
        item, _ = self._find_row(entry)
        if item is not None:
            self._list.setCurrentItem(item)

    def _on_entry_removed(self, entry: dict) -> None:
        # Full refresh — same pattern as add/changed handlers. The previous
        # surgical takeItem path was unreliable when combined with
        # setItemWidget (the item's data identity check could mismatch on
        # rebuilt rows). Rebuild is fast enough for our 500-entry cap.
        log_app.debug("on_entry_removed: rebuilding list")
        self._refresh_list()

    def _on_selection(self) -> None:
        items = self._list.selectedItems()
        if not items:
            self._detail.setPlainText("")
            if hasattr(self, "_delete_selected_btn"):
                self._delete_selected_btn.setEnabled(False)
            return
        entry = items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self._detail.setPlainText(entry.get("text", ""))
            if hasattr(self, "_delete_selected_btn"):
                self._delete_selected_btn.setEnabled(True)

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
            "Clear all transcription history? This cannot be undone.\n\n"
            "Pinned items will also be removed.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._log.clear()

    # --- per-row actions ------------------------------------------------

    def _on_row_copy(self, entry: dict) -> None:
        self._copy_to_clipboard(entry.get("text", ""), source="row-copy")

    def _delete_with_confirm(self, entry: dict) -> None:
        """Confirm with a Yes/No dialog, then delete. Used by the top-bar
        button, the Delete key, and the right-click menu — every delete path
        confirms so misclicks can't destroy data."""
        text = entry.get("text", "")
        preview = text if len(text) <= 80 else text[:77] + "…"
        reply = QMessageBox.question(
            self, "Delete transcription",
            f"Delete this transcription?\n\n“{preview}”\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._log.delete_entry(entry)

    def _delete_selected_confirm(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        entry = items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self._delete_with_confirm(entry)

    def _on_row_edit_start(self, entry: dict) -> None:
        current = entry.get("text", "")
        new_text, ok = QInputDialog.getMultiLineText(
            self, "Edit transcription",
            "Fix typos, then click OK. The edited text replaces the original.",
            current,
        )
        if ok and new_text.strip() and new_text != current:
            self._log.update_text(entry, new_text)

    def _on_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        entry = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, dict):
            return
        menu = QMenu(self)

        copy_act = QAction("Copy", self)
        copy_act.triggered.connect(lambda: self._copy_to_clipboard(entry.get("text", ""), "ctx-copy"))
        menu.addAction(copy_act)

        copy_ts_act = QAction("Copy with timestamp", self)
        def _copy_ts() -> None:
            ts = entry.get("ts", 0)
            stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts)) if ts else ""
            text = entry.get("text", "")
            self._copy_to_clipboard(f"[{stamp}] {text}" if stamp else text, "ctx-copy-ts")
        copy_ts_act.triggered.connect(_copy_ts)
        menu.addAction(copy_ts_act)

        copy_md_act = QAction("Copy as markdown quote", self)
        def _copy_md() -> None:
            text = entry.get("text", "")
            lines = ["> " + ln for ln in text.splitlines() or [""]]
            self._copy_to_clipboard("\n".join(lines), "ctx-copy-md")
        copy_md_act.triggered.connect(_copy_md)
        menu.addAction(copy_md_act)

        menu.addSeparator()

        edit_act = QAction("Edit text…", self)
        edit_act.triggered.connect(lambda: self._on_row_edit_start(entry))
        menu.addAction(edit_act)

        menu.addSeparator()

        del_act = QAction("Delete…", self)
        del_act.triggered.connect(lambda: self._delete_with_confirm(entry))
        menu.addAction(del_act)

        menu.exec(self._list.mapToGlobal(pos))

    # --- notes tab ------------------------------------------------------

    def _build_notes_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 14, 0, 0)
        layout.setSpacing(8)

        hint = QLabel(
            "Notes are captured with the <b>Notes hotkey</b> (default <b>Ctrl+Win+Z</b>; "
            "remap in Settings → Hotkeys). Press to start, press again to stop. "
            "Notes are <i>not</i> pasted into the focused window. Right-click a note to flag it."
        )
        hint.setObjectName("sectionHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        self._notes_delete_btn = QPushButton("Delete selected")
        self._notes_delete_btn.setEnabled(False)
        self._notes_delete_btn.clicked.connect(self._delete_selected_note_confirm)
        top_bar.addWidget(self._notes_delete_btn)

        self._notes_flag_filter = _NoWheelComboBox()
        self._notes_flag_filter.addItem("All flags", "")
        self._notes_flag_filter.addItem("Unflagged only", "__none__")
        for f in NOTE_FLAGS:
            self._notes_flag_filter.addItem(f.capitalize(), f)
        self._notes_flag_filter.currentIndexChanged.connect(
            lambda _i: self._refresh_notes_list()
        )
        top_bar.addWidget(self._notes_flag_filter)

        self._notes_search = QLineEdit()
        self._notes_search.setPlaceholderText("Search notes…")
        self._notes_search.textChanged.connect(lambda _s: self._refresh_notes_list())
        self._notes_search.setClearButtonEnabled(True)
        top_bar.addWidget(self._notes_search, stretch=1)
        layout.addLayout(top_bar)

        self._notes_list = HistoryList()
        self._notes_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self._notes_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._notes_list.customContextMenuRequested.connect(self._on_notes_context_menu)
        self._notes_list.itemSelectionChanged.connect(self._on_notes_selection)
        self._notes_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._notes_list.setUniformItemSizes(True)
        layout.addWidget(self._notes_list, stretch=2)

        layout.addWidget(QLabel("Selected:"))
        self._notes_detail = QTextEdit()
        self._notes_detail.setReadOnly(True)
        self._notes_detail.setPlaceholderText("Click a note above to view its full text.")
        layout.addWidget(self._notes_detail, stretch=1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        copy_sel = QPushButton("Copy selected")
        copy_sel.clicked.connect(self._copy_selected_note)
        actions.addWidget(copy_sel)
        actions.addStretch()
        clear = QPushButton("Clear notes")
        clear.clicked.connect(self._clear_notes)
        actions.addWidget(clear)
        layout.addLayout(actions)

        return w

    def _sorted_filtered_notes(self) -> list[dict]:
        query = (self._notes_search.text().strip().lower()
                 if hasattr(self, "_notes_search") else "")
        flag_pick = (self._notes_flag_filter.currentData()
                     if hasattr(self, "_notes_flag_filter") else "")
        all_entries = self._notes.entries()
        if query:
            all_entries = [
                e for e in all_entries
                if query in (e.get("text") or "").lower()
                or query in (e.get("app") or "").lower()
            ]
        if flag_pick == "__none__":
            all_entries = [e for e in all_entries if not (e.get("flag") or "")]
        elif flag_pick:
            all_entries = [e for e in all_entries if (e.get("flag") or "") == flag_pick]
        return list(reversed(all_entries))

    def _install_note_row(self, item: "QListWidgetItem", entry: dict) -> None:
        row = NoteRow(entry)
        row.copy_requested.connect(self._on_note_row_copy)
        item.setData(Qt.ItemDataRole.UserRole, entry)
        w = self._notes_list.viewport().width() if self._notes_list.viewport().width() > 0 else 580
        item.setSizeHint(QSize(w, 34))
        row.setFixedWidth(w)
        self._notes_list.setItemWidget(item, row)

    def _refresh_notes_list(self) -> None:
        if not hasattr(self, "_notes_list"):
            return
        self._notes_list.clear()
        for entry in self._sorted_filtered_notes():
            item = QListWidgetItem()
            self._notes_list.addItem(item)
            self._install_note_row(item, entry)
        self._notes_list.refit_rows()
        if self._notes_list.count():
            self._notes_list.setCurrentRow(0)
        else:
            self._notes_detail.setPlainText("")

    def _find_note_row(self, entry: dict) -> "tuple[QListWidgetItem | None, NoteRow | None]":
        for i in range(self._notes_list.count()):
            item = self._notes_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) is entry:
                return item, self._notes_list.itemWidget(item)
        return None, None

    def _on_note_added(self, _entry: dict) -> None:
        self._refresh_notes_list()

    def _on_note_changed(self, entry: dict) -> None:
        # Flag change can affect filter membership — rebuild whole list and
        # reselect the same entry if it survived the filter.
        self._refresh_notes_list()
        item, _ = self._find_note_row(entry)
        if item is not None:
            self._notes_list.setCurrentItem(item)

    def _on_note_removed(self, _entry: dict) -> None:
        self._refresh_notes_list()

    def _on_notes_selection(self) -> None:
        items = self._notes_list.selectedItems()
        entry = items[0].data(Qt.ItemDataRole.UserRole) if items else None
        if isinstance(entry, dict):
            self._notes_detail.setPlainText(entry.get("text", ""))
            self._notes_delete_btn.setEnabled(True)
        else:
            self._notes_detail.setPlainText("")
            self._notes_delete_btn.setEnabled(False)

    def _on_note_row_copy(self, entry: dict) -> None:
        self._copy_to_clipboard(entry.get("text", ""), source="note-row-copy")

    def _copy_selected_note(self) -> None:
        items = self._notes_list.selectedItems()
        if not items:
            return
        entry = items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self._copy_to_clipboard(entry.get("text", ""), source="note-copy-selected")

    def _clear_notes(self) -> None:
        reply = QMessageBox.question(
            self, "Clear notes",
            "Delete all notes? This cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._notes.clear()

    def _delete_note_with_confirm(self, entry: dict) -> None:
        text = entry.get("text", "")
        preview = text if len(text) <= 80 else text[:77] + "…"
        reply = QMessageBox.question(
            self, "Delete note",
            f"Delete this note?\n\n“{preview}”\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._notes.delete_entry(entry)

    def _delete_selected_note_confirm(self) -> None:
        items = self._notes_list.selectedItems()
        if not items:
            return
        entry = items[0].data(Qt.ItemDataRole.UserRole)
        if isinstance(entry, dict):
            self._delete_note_with_confirm(entry)

    def _on_notes_context_menu(self, pos) -> None:
        item = self._notes_list.itemAt(pos)
        if item is None:
            return
        entry = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(entry, dict):
            return
        menu = QMenu(self)

        copy_act = QAction("Copy", self)
        copy_act.triggered.connect(
            lambda: self._copy_to_clipboard(entry.get("text", ""), "note-ctx-copy")
        )
        menu.addAction(copy_act)

        menu.addSeparator()

        flag_menu = menu.addMenu("Flag as")
        current_flag = (entry.get("flag") or "").lower()
        none_act = QAction("None (clear flag)", self)
        none_act.setCheckable(True)
        none_act.setChecked(current_flag == "")
        none_act.triggered.connect(lambda: self._notes.set_flag(entry, ""))
        flag_menu.addAction(none_act)
        flag_menu.addSeparator()
        for f in NOTE_FLAGS:
            act = QAction(f.capitalize(), self)
            act.setCheckable(True)
            act.setChecked(current_flag == f)
            act.triggered.connect(lambda _checked=False, flag=f: self._notes.set_flag(entry, flag))
            flag_menu.addAction(act)

        menu.addSeparator()

        del_act = QAction("Delete…", self)
        del_act.triggered.connect(lambda: self._delete_note_with_confirm(entry))
        menu.addAction(del_act)

        menu.exec(self._notes_list.mapToGlobal(pos))

    def eventFilter(self, obj, event) -> bool:
        """List keyboard shortcuts: Enter=copy, Delete=delete-with-confirm,
        Ctrl+F=focus search."""
        from PyQt6.QtCore import QEvent
        if obj is self._list and event.type() == QEvent.Type.KeyPress:
            key = event.key()
            mods = event.modifiers()
            items = self._list.selectedItems()
            entry = items[0].data(Qt.ItemDataRole.UserRole) if items else None
            if entry is not None and isinstance(entry, dict):
                if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                    self._copy_to_clipboard(entry.get("text", ""), "key-enter")
                    return True
                if key == Qt.Key.Key_Delete:
                    self._delete_with_confirm(entry)
                    return True
            if key == Qt.Key.Key_F and mods & Qt.KeyboardModifier.ControlModifier:
                self._search.setFocus()
                self._search.selectAll()
                return True
        return super().eventFilter(obj, event)

    def _apply_theme(self, theme: str) -> None:
        """Swap the panel's stylesheet to the requested theme. Idempotent —
        a no-op when the theme is already active. Forces a style refresh
        on every styled descendant so objectName-driven rules re-render."""
        if theme not in THEME_PALETTES:
            theme = "dark"
        if theme == self._current_theme:
            return
        self._current_theme = theme
        self.setStyleSheet(build_panel_qss(theme))
        # Qt caches polished state per widget — unpolish/polish forces a
        # re-evaluation of QSS so newly-named objects pick up the new colors
        # immediately (without this, some labels stay on the prior theme
        # until they're next interacted with).
        for w in self.findChildren(QWidget):
            w.style().unpolish(w)
            w.style().polish(w)

    # --- settings tab ---------------------------------------------------

    # Common label width for left-hand labels in form rows. Picked so the
    # longest label ("Microphone", "Endpoint", "Default prompt", "Input gain")
    # all align without truncation at the default font size.
    _FORM_LABEL_WIDTH = 110

    def _section_header(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionHeader")
        return lbl

    def _section_hint(self, text: str, *, rich: bool = False) -> QLabel:
        lbl = QLabel(text)
        lbl.setObjectName("sectionHint")
        if rich:
            lbl.setTextFormat(Qt.TextFormat.RichText)
        lbl.setWordWrap(True)
        return lbl

    def _form_row(self, label_text: str, control: QWidget) -> QHBoxLayout:
        """Build a label-control row with consistent label width so every
        section's left edge of controls aligns."""
        row = QHBoxLayout()
        row.setSpacing(10)
        row.setContentsMargins(0, 0, 0, 0)
        lbl = QLabel(label_text)
        lbl.setObjectName("formLabel")
        lbl.setFixedWidth(self._FORM_LABEL_WIDTH)
        lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(lbl)
        # Let the control expand horizontally inside the row.
        try:
            control.setSizePolicy(QSizePolicy.Policy.Expanding, control.sizePolicy().verticalPolicy())
        except Exception:
            pass
        row.addWidget(control, stretch=1)
        return row

    def _make_device_combo(self) -> QComboBox:
        """Create the input-device combo (used during settings build)."""
        self._device_combo = _NoWheelComboBox()
        self._populate_device_combo()
        self._device_combo.currentIndexChanged.connect(self._on_device_changed)
        return self._device_combo

    def _build_settings_tab(self) -> QWidget:
        # Inner content widget — all sections go here, then we wrap it in a
        # QScrollArea so a long Settings tab doesn't force the window taller.
        content = QWidget()
        content.setObjectName("settingsContent")
        # Plain QWidget ignores QSS-defined backgrounds by default; enable
        # styled background painting so the theme's bg actually fills the
        # content area (otherwise it falls back to the system dark color,
        # which broke the light theme).
        content.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(content)
        # Left padding kept generous so checkbox indicators and button edges
        # don't get clipped by the scroll-area viewport. Right padding is a
        # bit larger to leave room for the vertical scrollbar.
        layout.setContentsMargins(14, 16, 18, 14)
        layout.setSpacing(20)

        # --- appearance section (v2.0) ---
        appearance_box = QVBoxLayout()
        appearance_box.setSpacing(10)
        appearance_box.addWidget(self._section_header("Appearance"))
        self._theme_combo = _NoWheelComboBox()
        self._theme_combo.addItem("Dark", userData="dark")
        self._theme_combo.addItem("Light", userData="light")
        self._theme_combo.setCurrentIndex(
            0 if self._settings.theme == "dark" else 1
        )
        self._theme_combo.currentIndexChanged.connect(self._on_theme_changed)
        appearance_box.addLayout(self._form_row("Theme", self._theme_combo))
        layout.addLayout(appearance_box)

        # --- general toggles section ---
        gen_box = QVBoxLayout()
        gen_box.setSpacing(10)
        gen_box.addWidget(self._section_header("General"))

        self._auto_paste_cb = QCheckBox("Auto-paste transcription into focused window")
        self._auto_paste_cb.setChecked(self._settings.auto_paste)
        self._auto_paste_cb.toggled.connect(self._on_auto_paste_toggled)
        gen_box.addWidget(self._auto_paste_cb)

        # Voice commands & smart formatting (the v1.4 feature, single toggle)
        self._voice_commands_cb = QCheckBox("Voice commands & punctuation dictation")
        self._voice_commands_cb.setChecked(self._settings.voice_commands)
        self._voice_commands_cb.toggled.connect(self._on_voice_commands_toggled)
        gen_box.addWidget(self._voice_commands_cb)
        gen_box.addWidget(self._section_hint(
            "When on, dictated words like \"comma\", \"period\", \"new line\", \"open quote\", "
            "\"close quote\" become actual punctuation. \"Scratch that\" / \"delete that\" "
            "drops the previous sentence. First letter of each sentence is auto-capitalized."
        ))

        # Start-on-login (v2.2.2) — toggles the elevated "Tellur" scheduled
        # task when it exists (v2.3.3), else writes an HKCU\...\Run entry
        # pointing at this install's run.bat.
        self._start_on_login_cb = QCheckBox("Start Tellur when I log into Windows")
        self._start_on_login_cb.setChecked(self._settings.start_on_login)
        self._start_on_login_cb.toggled.connect(self._on_start_on_login_toggled)
        gen_box.addWidget(self._start_on_login_cb)
        gen_box.addWidget(self._section_hint(
            "Uses the elevated \"Tellur\" scheduled task when present — starts "
            "silently at login with no UAC prompt, and paste works in admin "
            "windows. Without the task it falls back to a user-level Run "
            "registry key (one UAC prompt at login because <code>run.bat</code> "
            "self-elevates; set <code>TELLUR_NO_ELEVATE=1</code> to skip the "
            "prompt and run unelevated)."
        ))
        layout.addLayout(gen_box)

        # --- hotkey mode section ---
        hk_box = QVBoxLayout()
        hk_box.setSpacing(10)
        hk_box.addWidget(self._section_header("Push-to-talk mode"))

        self._hk_mode_combo = _NoWheelComboBox()
        self._hk_mode_combo.addItem("Hold — speak while Ctrl+Win is held", userData="hold")
        self._hk_mode_combo.addItem("Toggle — tap Ctrl+Win to start, tap again to stop", userData="toggle")
        cur_idx = 0 if self._settings.hotkey_mode == "hold" else 1
        self._hk_mode_combo.setCurrentIndex(cur_idx)
        self._hk_mode_combo.currentIndexChanged.connect(self._on_hk_mode_changed)
        hk_box.addWidget(self._hk_mode_combo)

        hk_box.addWidget(self._section_hint(
            "Hold is the classic press-to-talk feel. Toggle is great for longer dictations "
            "so you don't have to keep the keys pressed. In toggle mode, press Esc to abandon "
            "the current recording without transcribing."
        ))
        layout.addLayout(hk_box)

        # --- hotkeys (editable + fixed-reference) ---
        ref_box = QVBoxLayout()
        ref_box.setSpacing(10)
        ref_box.addWidget(self._section_header("Hotkeys"))
        ref_box.addWidget(self._section_hint(
            "<b>Ctrl + Win</b> (push-to-talk) and <b>Esc</b> (cancel) are fixed. "
            "Click a field below, press the new combo, then it saves automatically. "
            "Modifier-free single keys (like <i>F2</i>) work but may collide with other apps."
        ))
        # Built each via a small helper so editing one set rebinds itself
        # against current Settings without restarting Tellur.
        self._hotkey_edits: dict[str, QKeySequenceEdit] = {}
        for slot, label, tip in (
            ("note",    "Notes (toggle)",
             "Start/stop a voice note (saved to the Notes tab; not pasted)"),
            ("repaste", "Re-paste last",
             "Paste the most recent transcript into the focused window again"),
            ("llm",     "Apply AI prompt",
             "Run the default AI prompt on the last transcript and paste the result"),
            ("quit",    "Quit Tellur",
             "Exit Tellur"),
        ):
            current = getattr(self._settings, f"hotkey_{slot}", "")
            row_widget = QWidget()
            row_inner = QHBoxLayout(row_widget)
            row_inner.setContentsMargins(0, 0, 0, 0)
            row_inner.setSpacing(8)
            edit = QKeySequenceEdit(combo_to_keyseq(current))
            edit.setToolTip(tip)
            edit.editingFinished.connect(
                lambda s=slot, e=edit: self._on_hotkey_changed(s, e)
            )
            self._hotkey_edits[slot] = edit
            row_inner.addWidget(edit, stretch=1)
            reset_btn = QPushButton("Reset")
            reset_btn.setToolTip("Restore this hotkey to its default")
            reset_btn.clicked.connect(lambda _checked=False, s=slot: self._reset_hotkey(s))
            row_inner.addWidget(reset_btn)
            ref_box.addLayout(self._form_row(label, row_widget))
        layout.addLayout(ref_box)

        # --- audio ducking (v2.2) ---
        duck_box = QVBoxLayout()
        duck_box.setSpacing(10)
        duck_box.addWidget(self._section_header("Lower other apps while recording"))
        duck_box.addWidget(self._section_hint(
            "When you start recording, drop the volume of every other app "
            "(Discord, browser, music, games) to a low level so background "
            "audio doesn't compete with your voice. Volume snaps back when "
            "you release the key. Tellur itself is never ducked."
        ))
        self._duck_enabled_cb = QCheckBox("Enabled")
        self._duck_enabled_cb.setChecked(self._settings.duck_enabled)
        self._duck_enabled_cb.toggled.connect(self._on_duck_enabled_toggled)
        duck_box.addLayout(self._form_row("Audio ducking", self._duck_enabled_cb))

        self._duck_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self._duck_slider.setRange(0, 50)
        self._duck_slider.setValue(int(self._settings.duck_level_pct))
        self._duck_slider.valueChanged.connect(self._on_duck_level_changed)
        self._duck_label = QLabel(f"{int(self._settings.duck_level_pct)}%")
        self._duck_label.setFixedWidth(40)
        self._duck_label.setObjectName("valueDim")
        duck_widget = QWidget()
        duck_inner = QHBoxLayout(duck_widget)
        duck_inner.setContentsMargins(0, 0, 0, 0)
        duck_inner.setSpacing(8)
        duck_inner.addWidget(self._duck_slider, stretch=1)
        duck_inner.addWidget(self._duck_label)
        duck_box.addLayout(self._form_row("Lower to", duck_widget))
        if not AudioDucker.available():
            warn = self._section_hint(
                "<b>pycaw is not available</b> — install it (<i>pip install pycaw</i>) "
                "into the Tellur venv to enable ducking."
            )
            duck_box.addWidget(warn)
            self._duck_enabled_cb.setEnabled(False)
            self._duck_slider.setEnabled(False)
        layout.addLayout(duck_box)

        # --- audio input section (v1.6) ---
        audio_box = QVBoxLayout()
        audio_box.setSpacing(10)
        audio_box.addWidget(self._section_header("Audio input"))

        # Device picker.
        audio_box.addLayout(self._form_row("Microphone", self._make_device_combo()))

        # Live input level meter — only animates while Settings is visible
        # and the recorder isn't currently capturing for a transcription.
        self._level_meter = QProgressBar()
        self._level_meter.setRange(0, 1000)
        self._level_meter.setTextVisible(False)
        self._level_meter.setFixedHeight(10)
        self._level_meter.setValue(0)
        audio_box.addLayout(self._form_row("Level", self._level_meter))

        # Gain slider + value label, packed in a sub-row.
        self._gain_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self._gain_slider.setRange(10, 400)
        self._gain_slider.setValue(int(self._settings.input_gain * 100))
        self._gain_slider.valueChanged.connect(self._on_gain_changed)
        self._gain_label = QLabel(f"{self._settings.input_gain:.2f}x")
        self._gain_label.setFixedWidth(56)
        self._gain_label.setObjectName("valueDim")
        gain_widget = QWidget()
        gain_inner = QHBoxLayout(gain_widget)
        gain_inner.setContentsMargins(0, 0, 0, 0)
        gain_inner.setSpacing(8)
        gain_inner.addWidget(self._gain_slider, stretch=1)
        gain_inner.addWidget(self._gain_label)
        audio_box.addLayout(self._form_row("Input gain", gain_widget))

        audio_box.addWidget(self._section_hint(
            "Boost quiet mics with gain. If Whisper struggles on soft speech, push gain up. "
            "If your audio clips or distorts on loud bursts, push it down."
        ))
        layout.addLayout(audio_box)

        # --- mic passthrough section (v2.7) ---
        pt_box = QVBoxLayout()
        pt_box.setSpacing(10)
        pt_box.addWidget(self._section_header("Mic passthrough"))

        self._pt_enabled_cb = QCheckBox("Enabled")
        self._pt_enabled_cb.setChecked(self._settings.passthrough_enabled)
        self._pt_enabled_cb.toggled.connect(self._on_pt_enabled_toggled)
        pt_box.addLayout(self._form_row("Hear-through", self._pt_enabled_cb))

        self._pt_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self._pt_slider.setRange(0, 100)
        self._pt_slider.setValue(int(self._settings.passthrough_level_pct))
        self._pt_slider.valueChanged.connect(self._on_pt_level_changed)
        self._pt_label = QLabel(f"{int(self._settings.passthrough_level_pct)}%")
        self._pt_label.setFixedWidth(56)
        self._pt_label.setObjectName("valueDim")
        pt_widget = QWidget()
        pt_inner = QHBoxLayout(pt_widget)
        pt_inner.setContentsMargins(0, 0, 0, 0)
        pt_inner.setSpacing(8)
        pt_inner.addWidget(self._pt_slider, stretch=1)
        pt_inner.addWidget(self._pt_label)
        pt_box.addLayout(self._form_row("Passthrough level", pt_widget))

        pt_box.addLayout(self._form_row("Play into", self._make_pt_output_combo()))

        self._pt_mute_cb = QCheckBox("Mute while dictating")
        self._pt_mute_cb.setChecked(self._settings.passthrough_mute_while_dictating)
        self._pt_mute_cb.toggled.connect(self._on_pt_mute_toggled)
        pt_box.addLayout(self._form_row("While recording", self._pt_mute_cb))

        pt_box.addWidget(self._section_hint(
            "Mixes your mic into your headphones so you can hear the room without "
            "taking them out. This level is separate from the Windows mic volume, "
            "so turning it down does not make you quieter to anyone else. "
            "Passthrough is played into the chosen output, so any system EQ on that "
            "device (Equalizer APO and the like) applies to it too \u2014 start low. "
            "On open-back headphones or speakers a high level can feed back. "
            "If you hear yourself echoing, pick the same device's WASAPI entry \u2014 "
            "it buffers far less than MME."
        ))
        layout.addLayout(pt_box)

        # --- overlay (visual-only) ---
        sens_box = QVBoxLayout()
        sens_box.setSpacing(10)
        sens_box.addWidget(self._section_header("Overlay"))
        self._sens_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        self._sens_slider.setRange(20, 400)
        self._sens_slider.setValue(int(self._settings.sensitivity))
        self._sens_slider.valueChanged.connect(self._on_sens_changed)
        self._sens_label = QLabel(str(int(self._settings.sensitivity)))
        self._sens_label.setFixedWidth(40)
        self._sens_label.setObjectName("valueDim")
        sens_widget = QWidget()
        sens_inner = QHBoxLayout(sens_widget)
        sens_inner.setContentsMargins(0, 0, 0, 0)
        sens_inner.setSpacing(8)
        sens_inner.addWidget(self._sens_slider, stretch=1)
        sens_inner.addWidget(self._sens_label)
        sens_box.addLayout(self._form_row("Meter sensitivity", sens_widget))
        sens_box.addWidget(self._section_hint(
            "Mic-meter sensitivity on the on-screen overlay only — does not affect transcription."
        ))
        layout.addLayout(sens_box)

        # --- transcription model section ---
        model_box = QVBoxLayout()
        model_box.setSpacing(10)
        model_box.addWidget(self._section_header("Transcription model"))
        model_box.addWidget(self._section_hint(
            "Switching downloads the model on first use. The download is cached, "
            "so future switches between models you've already used are instant."
        ))

        self._model_combo = _NoWheelComboBox()
        self._populate_model_combo()
        self._model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        model_box.addWidget(self._model_combo)

        self._model_status = QLabel(f"Active: {self._settings.model_name}")
        self._model_status.setObjectName("valueDim")
        model_box.addWidget(self._model_status)

        self._model_progress = QProgressBar()
        self._model_progress.setRange(0, 100)
        self._model_progress.setValue(0)
        self._model_progress.setTextVisible(True)
        self._model_progress.setFormat("%p%")
        self._model_progress.setVisible(False)
        self._model_progress.setFixedHeight(14)
        model_box.addWidget(self._model_progress)

        layout.addLayout(model_box)

        # --- transcription engine section (v2.3) — local vs ElevenLabs cloud ---
        stt_box = QVBoxLayout()
        stt_box.setSpacing(10)
        stt_box.addWidget(self._section_header("Transcription engine"))
        stt_box.addWidget(self._section_hint(
            "Local Whisper is fully offline and the default. ElevenLabs Scribe "
            "sends your audio to ElevenLabs for transcription — it needs an internet "
            "connection and an API key with Speech-to-Text access."
        ))

        self._stt_backend_combo = _NoWheelComboBox()
        self._stt_backend_combo.addItem(
            "Auto — local when GPU is free, cloud when it's busy",
            userData="auto")
        self._stt_backend_combo.addItem("Local Whisper (offline)", userData="local")
        self._stt_backend_combo.addItem("ElevenLabs Scribe (cloud)", userData="elevenlabs")
        idx = self._stt_backend_combo.findData(self._settings.stt_backend)
        self._stt_backend_combo.setCurrentIndex(idx if idx >= 0 else 1)
        self._stt_backend_combo.currentIndexChanged.connect(self._on_stt_backend_changed)
        stt_box.addLayout(self._form_row("Engine", self._stt_backend_combo))

        # Live truth about what will actually serve the next dictation. Before
        # this, a drained key was only visible in the log: "Test key" passes on
        # a 1-second probe even when the balance can't cover a real clip, so the
        # UI said ✓ while every dictation quietly ran on local Whisper.
        self._stt_engine_status = QLabel("")
        self._stt_engine_status.setObjectName("valueDim")
        self._stt_engine_status.setWordWrap(True)
        stt_box.addWidget(self._stt_engine_status)
        self._stt_status_timer = QTimer(self)
        self._stt_status_timer.setInterval(5000)
        self._stt_status_timer.timeout.connect(self._refresh_stt_status)
        self._stt_status_timer.start()
        self._refresh_stt_status()

        self._eleven_clean_cb = QCheckBox(
            "Clean up cloud transcripts (drop filler words and false starts)")
        self._eleven_clean_cb.setToolTip(
            "Asks ElevenLabs to leave out \"um\", \"uh\" and restarted words "
            "instead of writing them down literally. Local Whisper already "
            "writes this way. Turn off for word-for-word transcripts."
        )
        self._eleven_clean_cb.setChecked(self._settings.eleven_clean_transcript)
        self._eleven_clean_cb.toggled.connect(self._on_eleven_clean_toggled)
        stt_box.addWidget(self._eleven_clean_cb)

        self._eleven_key_edit = QLineEdit(self._settings.eleven_api_key)
        self._eleven_key_edit.setPlaceholderText("ElevenLabs API key (Speech-to-Text access)")
        self._eleven_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._eleven_key_edit.editingFinished.connect(self._on_eleven_key_changed)
        stt_box.addLayout(self._form_row("ElevenLabs API key", self._eleven_key_edit))

        self._eleven_key2_edit = QLineEdit(self._settings.eleven_api_key_backup)
        self._eleven_key2_edit.setPlaceholderText(
            "Backup key — used automatically when the primary runs out of credits"
        )
        self._eleven_key2_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._eleven_key2_edit.editingFinished.connect(self._on_eleven_key2_changed)
        stt_box.addLayout(self._form_row("Backup API key", self._eleven_key2_edit))

        # Keys 3..N. Each ElevenLabs key carries its own monthly allowance, so
        # stacking keys is how the monthly ceiling gets raised; two fixed slots
        # was the real limit we hit. One per line, tried in order after the two
        # above. Saved on a debounce so typing doesn't hammer settings.json.
        self._eleven_keys_extra_edit = QTextEdit()
        self._eleven_keys_extra_edit.setPlainText(
            "\n".join(self._settings.eleven_api_keys_extra)
        )
        self._eleven_keys_extra_edit.setPlaceholderText(
            "One key per line — tried in order after the primary and backup"
        )
        self._eleven_keys_extra_edit.setFixedHeight(64)
        self._eleven_keys_extra_edit.setAcceptRichText(False)
        self._extra_keys_save_timer = QTimer(self)
        self._extra_keys_save_timer.setSingleShot(True)
        self._extra_keys_save_timer.setInterval(800)
        self._extra_keys_save_timer.timeout.connect(self._on_eleven_keys_extra_changed)
        self._eleven_keys_extra_edit.textChanged.connect(
            self._extra_keys_save_timer.start
        )
        stt_box.addLayout(self._form_row("Extra API keys", self._eleven_keys_extra_edit))

        eleven_test_row = QHBoxLayout()
        eleven_test_row.setSpacing(10)
        self._eleven_test_btn = QPushButton("Test key")
        self._eleven_test_btn.clicked.connect(self._on_eleven_test_clicked)
        eleven_test_row.addWidget(self._eleven_test_btn)
        self._eleven_test_status = QLabel("")
        self._eleven_test_status.setObjectName("valueDim")
        eleven_test_row.addWidget(self._eleven_test_status, stretch=1)
        stt_box.addLayout(eleven_test_row)

        layout.addLayout(stt_box)

        # --- AI post-processing section (v1.7) ---
        llm_box = QVBoxLayout()
        llm_box.setSpacing(10)
        llm_box.addWidget(self._section_header("AI post-processing (optional)"))
        llm_box.addWidget(self._section_hint(
            "Send transcripts to a local (or remote) OpenAI-compatible LLM to clean them up, "
            "rewrite as bullets/email/Slack, or summarize. Works with LM Studio, Ollama, "
            "llama.cpp server, vLLM, etc. Stays 100% local if your endpoint is local."
        ))

        self._llm_enabled_cb = QCheckBox("Enable AI post-processing")
        self._llm_enabled_cb.setChecked(self._settings.llm_enabled)
        self._llm_enabled_cb.toggled.connect(self._on_llm_enabled_toggled)
        llm_box.addWidget(self._llm_enabled_cb)

        self._llm_url_edit = QLineEdit(self._settings.llm_base_url)
        self._llm_url_edit.setPlaceholderText("http://localhost:1234/v1")
        self._llm_url_edit.editingFinished.connect(self._on_llm_url_changed)
        llm_box.addLayout(self._form_row("Endpoint", self._llm_url_edit))

        self._llm_model_edit = QLineEdit(self._settings.llm_model)
        self._llm_model_edit.setPlaceholderText("often optional for LM Studio; required for Ollama")
        self._llm_model_edit.editingFinished.connect(self._on_llm_model_changed)
        llm_box.addLayout(self._form_row("Model", self._llm_model_edit))

        self._llm_key_edit = QLineEdit(self._settings.llm_api_key)
        self._llm_key_edit.setPlaceholderText("leave blank for local endpoints")
        self._llm_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._llm_key_edit.editingFinished.connect(self._on_llm_key_changed)
        llm_box.addLayout(self._form_row("API key", self._llm_key_edit))

        self._llm_prompt_combo = _NoWheelComboBox()
        for p in BUILTIN_LLM_PROMPTS:
            self._llm_prompt_combo.addItem(p["label"], userData=p["id"])
        try:
            cur_idx = next(i for i, p in enumerate(BUILTIN_LLM_PROMPTS)
                           if p["id"] == self._settings.llm_default_prompt)
        except StopIteration:
            cur_idx = 0
        self._llm_prompt_combo.setCurrentIndex(cur_idx)
        self._llm_prompt_combo.currentIndexChanged.connect(self._on_llm_prompt_changed)
        llm_box.addLayout(self._form_row("Default prompt", self._llm_prompt_combo))

        self._llm_auto_cb = QCheckBox("Auto-apply default prompt to every transcript")
        self._llm_auto_cb.setChecked(self._settings.llm_auto_apply)
        self._llm_auto_cb.toggled.connect(self._on_llm_auto_toggled)
        llm_box.addWidget(self._llm_auto_cb)

        # Test connection button + status.
        test_row = QHBoxLayout()
        test_row.setSpacing(10)
        self._llm_test_btn = QPushButton("Test connection")
        self._llm_test_btn.clicked.connect(self._on_llm_test_clicked)
        test_row.addWidget(self._llm_test_btn)
        self._llm_test_status = QLabel("")
        self._llm_test_status.setObjectName("valueDim")
        test_row.addWidget(self._llm_test_status, stretch=1)
        llm_box.addLayout(test_row)

        llm_box.addWidget(self._section_hint(
            "Press <b>Ctrl+Win+L</b> to apply the default prompt to your most recent transcript.",
            rich=True,
        ))

        layout.addLayout(llm_box)

        # --- integration & automation section (v1.8) ---
        intg_box = QVBoxLayout()
        intg_box.setSpacing(10)
        intg_box.addWidget(self._section_header("Integration & automation"))

        self._send_enter_cb = QCheckBox("Press Enter after paste (chat-app mode)")
        self._send_enter_cb.setToolTip(
            "Useful for Slack/Discord/Teams where messages need an Enter to send. "
            "Skip for code editors and docs where Enter would insert a newline you don't want."
        )
        self._send_enter_cb.setChecked(self._settings.send_after_paste)
        self._send_enter_cb.toggled.connect(self._on_send_enter_toggled)
        intg_box.addWidget(self._send_enter_cb)

        # Terminal-aware paste (v2.3).
        self._term_paste_cb = QCheckBox(
            "Terminal-aware paste (CMD / PowerShell / Claude Code consoles)"
        )
        self._term_paste_cb.setToolTip(
            "Classic console windows can't accept a simulated Ctrl+V while a "
            "TUI app (like Claude Code) is running in them, so Tellur types "
            "the transcript there instead. A few terminal emulators that "
            "reserve a different paste chord (mintty, Alacritty, WezTerm) get "
            "theirs; everything else keeps plain Ctrl+V. "
            "Turn off to always use plain Ctrl+V."
        )
        self._term_paste_cb.setChecked(self._settings.terminal_aware_paste)
        self._term_paste_cb.toggled.connect(self._on_terminal_paste_toggled)
        intg_box.addWidget(self._term_paste_cb)
        intg_box.addWidget(self._section_hint(
            "In classic console windows the transcript is <i>typed</i>, with line "
            "breaks flattened to spaces — so dictation can never press Enter in "
            "your shell. Claude Code works best in Windows Terminal, where normal "
            "Ctrl+V paste applies.",
            rich=True,
        ))

        # Markdown daily save.
        self._md_enabled_cb = QCheckBox("Save each transcript to a daily markdown file")
        self._md_enabled_cb.setChecked(self._settings.markdown_save_enabled)
        self._md_enabled_cb.toggled.connect(self._on_md_enabled_toggled)
        intg_box.addWidget(self._md_enabled_cb)

        md_row = QHBoxLayout()
        md_row.setSpacing(8)
        self._md_folder_edit = QLineEdit(self._settings.markdown_folder)
        self._md_folder_edit.setPlaceholderText(str(DEFAULT_MARKDOWN_DIR))
        self._md_folder_edit.editingFinished.connect(self._on_md_folder_changed)
        md_row.addWidget(self._md_folder_edit, stretch=1)
        md_open_btn = QPushButton("Open")
        md_open_btn.setToolTip("Open the markdown folder in Explorer")
        md_open_btn.clicked.connect(self._on_md_open_clicked)
        md_row.addWidget(md_open_btn)
        md_row_widget = QWidget()
        md_row_widget.setLayout(md_row)
        intg_box.addLayout(self._form_row("Folder", md_row_widget))
        intg_box.addWidget(self._section_hint(
            "Leave folder blank to use the default under your Tellur data directory. "
            "One <code>YYYY-MM-DD.md</code> file per local-time day; each transcript gets "
            "a <code>##</code> heading with its timestamp.",
            rich=True,
        ))

        # Webhook.
        self._wh_enabled_cb = QCheckBox("POST every transcript to a webhook URL")
        self._wh_enabled_cb.setChecked(self._settings.webhook_enabled)
        self._wh_enabled_cb.toggled.connect(self._on_wh_enabled_toggled)
        intg_box.addWidget(self._wh_enabled_cb)

        self._wh_url_edit = QLineEdit(self._settings.webhook_url)
        self._wh_url_edit.setPlaceholderText("https://example.com/hook")
        self._wh_url_edit.editingFinished.connect(self._on_wh_url_changed)
        intg_box.addLayout(self._form_row("URL", self._wh_url_edit))

        self._wh_tpl_edit = QLineEdit(self._settings.webhook_template)
        self._wh_tpl_edit.setPlaceholderText('{"text": "{text}", "raw": "{raw}", "ts": {ts}}')
        self._wh_tpl_edit.editingFinished.connect(self._on_wh_template_changed)
        intg_box.addLayout(self._form_row("Body template", self._wh_tpl_edit))

        intg_box.addWidget(self._section_hint(
            "Placeholders: <b>{text}</b> = final transcript, <b>{raw}</b> = Whisper's raw output, "
            "<b>{ts}</b> = Unix timestamp. If the rendered body parses as JSON it's sent as "
            "<code>application/json</code>; otherwise as <code>text/plain</code>. "
            "Failures are logged and swallowed — the webhook never blocks dictation.",
            rich=True,
        ))

        layout.addLayout(intg_box)

        # --- privacy & data management section (v1.9) ---
        priv_box = QVBoxLayout()
        priv_box.setSpacing(10)
        priv_box.addWidget(self._section_header("Privacy & data"))

        self._save_history_cb = QCheckBox("Save transcript history")
        self._save_history_cb.setToolTip(
            "When off, transcripts are NOT written to history.json. The current "
            "in-memory list still tracks recent entries for hotkeys like Ctrl+Win+B "
            "/ Ctrl+Win+L during this session, but nothing is persisted to disk."
        )
        self._save_history_cb.setChecked(self._settings.save_history)
        self._save_history_cb.toggled.connect(self._on_save_history_toggled)
        priv_box.addWidget(self._save_history_cb)

        # Retention slider/spinner (use a slider for simplicity).
        retention_widget = QWidget()
        retention_row = QHBoxLayout(retention_widget)
        retention_row.setContentsMargins(0, 0, 0, 0)
        retention_row.setSpacing(8)
        self._retention_slider = _NoWheelSlider(Qt.Orientation.Horizontal)
        # 0..365 days. 0 = keep forever.
        self._retention_slider.setRange(0, 365)
        self._retention_slider.setValue(int(self._settings.history_retention_days))
        self._retention_slider.valueChanged.connect(self._on_retention_changed)
        self._retention_label = QLabel(self._format_retention(self._settings.history_retention_days))
        self._retention_label.setFixedWidth(110)
        self._retention_label.setObjectName("valueDim")
        retention_row.addWidget(self._retention_slider, stretch=1)
        retention_row.addWidget(self._retention_label)
        priv_box.addLayout(self._form_row("Auto-delete after", retention_widget))

        # Export + clear + data folder.
        action_row = QHBoxLayout()
        action_row.setSpacing(8)
        self._export_btn = QPushButton("Export history…")
        self._export_btn.clicked.connect(self._on_export_history_clicked)
        action_row.addWidget(self._export_btn)
        self._import_dict_btn = QPushButton("Import dictionary…")
        self._import_dict_btn.setToolTip(
            "Merge a replacements.json from another machine / teammate into your own."
        )
        self._import_dict_btn.clicked.connect(self._on_import_dict_clicked)
        action_row.addWidget(self._import_dict_btn)
        self._open_data_btn = QPushButton("Open data folder")
        self._open_data_btn.clicked.connect(self._on_open_data_clicked)
        action_row.addWidget(self._open_data_btn)
        priv_box.addLayout(action_row)

        clear_row = QHBoxLayout()
        clear_row.setSpacing(8)
        self._clear_all_btn = QPushButton("Clear all data…")
        self._clear_all_btn.setToolTip(
            "Wipe history.json AND your replacements dictionary. settings.json is kept."
        )
        self._clear_all_btn.setObjectName("dangerButton")
        self._clear_all_btn.clicked.connect(self._on_clear_all_clicked)
        clear_row.addWidget(self._clear_all_btn)
        clear_row.addStretch()
        priv_box.addLayout(clear_row)

        priv_box.addWidget(self._section_hint(
            "All Tellur data lives locally in your data folder. Set retention to 0 to keep history forever; "
            "any other value runs a one-shot purge at app startup and after each new transcript. "
            "Export creates a portable copy in the format you choose; Clear all data only removes "
            "<code>history.json</code> and <code>replacements.json</code> — your settings survive."
        , rich=True))

        layout.addLayout(priv_box)

        # --- per-context vocabulary section (v2.0) ---
        ctx_box = QVBoxLayout()
        ctx_box.setSpacing(10)
        ctx_box.addWidget(self._section_header("Per-app vocabulary"))
        ctx_box.addWidget(self._section_hint(
            "Drop a JSON file named after an app's executable into "
            "<code>replacements.d/</code> to swap in a tailored dictionary "
            "whenever that app is the foreground window. "
            "Example: <code>code.json</code> wins when VS Code is focused, "
            "<code>slack.json</code> wins when Slack is. Entries merge with "
            "your base <code>replacements.json</code>; per-app entries win "
            "on conflict. Same JSON format as the main dictionary."
        , rich=True))

        ctx_actions = QHBoxLayout()
        ctx_actions.setSpacing(8)
        self._open_ctx_dir_btn = QPushButton("Open per-app folder")
        self._open_ctx_dir_btn.clicked.connect(self._on_open_ctx_dir_clicked)
        ctx_actions.addWidget(self._open_ctx_dir_btn)
        self._show_ctx_btn = QPushButton("Show current app")
        self._show_ctx_btn.setToolTip(
            "Identify the foreground window's process name (use this to know "
            "what to name your JSON file)."
        )
        self._show_ctx_btn.clicked.connect(self._on_show_ctx_clicked)
        ctx_actions.addWidget(self._show_ctx_btn)
        ctx_actions.addStretch()
        ctx_box.addLayout(ctx_actions)

        self._ctx_status = QLabel("")
        self._ctx_status.setObjectName("valueDim")
        ctx_box.addWidget(self._ctx_status)

        layout.addLayout(ctx_box)

        # --- software updates section ---
        if self._updater is not None:
            update_box = QVBoxLayout()
            update_box.setSpacing(10)
            update_box.addWidget(self._section_header("Software updates"))

            row = QHBoxLayout()
            row.setSpacing(10)
            self._update_status = QLabel(f"Tellur {__version__} — checking for updates…")
            self._update_status.setObjectName("valueDim")
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

        # Troubleshooting ------------------------------------------------
        tb_box = QVBoxLayout()
        tb_box.setSpacing(10)
        tb_box.addWidget(self._section_header("Troubleshooting"))
        tb_box.addWidget(self._section_hint(
            "If the overlay disappears, the recording state seems stuck, or "
            "the tray icon misbehaves, click <b>Refresh UI</b> to rebuild the "
            "visible UI without restarting Tellur. The Whisper model stays "
            "loaded so this is fast."
        ))
        tb_row = QHBoxLayout()
        tb_row.setSpacing(10)
        refresh_btn = QPushButton("Refresh UI")
        refresh_btn.setToolTip("Re-create the overlay and reset any wedged UI state")
        refresh_btn.clicked.connect(self.refresh_ui_requested)
        tb_row.addWidget(refresh_btn)
        tb_row.addStretch()
        tb_box.addLayout(tb_row)
        layout.addLayout(tb_box)

        layout.addStretch()

        footer = QLabel(
            f"{APP_NAME} {__version__}  ·  Ctrl+Win to talk  ·  Esc cancel  ·  "
            f"hotkeys are editable above (defaults: Ctrl+Win+Z note, "
            f"Ctrl+Win+B re-paste, Ctrl+Win+L AI prompt, Ctrl+Win+Q quit)"
        )
        footer.setObjectName("footerLabel")
        footer.setWordWrap(True)
        layout.addWidget(footer)

        # Wrap content in a scroll area so a tall settings list doesn't force
        # the whole window taller. The scroll area is what the tab actually
        # displays; `content` is the inner widget that holds every section.
        scroll = QScrollArea()
        scroll.setObjectName("settingsScroll")
        scroll.setWidget(content)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        # Horizontal bar is AsNeeded (not AlwaysOff). At the default window
        # size all content fits, so the bar stays hidden. If the user
        # shrinks the window below the content's natural width, the bar
        # appears so cut-off controls can still be reached by scrolling —
        # instead of silently shifting on click via Qt's focus-follow.
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return scroll

    def _on_auto_paste_toggled(self, checked: bool) -> None:
        self._settings.auto_paste = bool(checked)
        self._settings.save()

    def _on_voice_commands_toggled(self, checked: bool) -> None:
        self._settings.voice_commands = bool(checked)
        self._settings.save()
        log_app.info("voice commands %s", "enabled" if checked else "disabled")

    def _on_start_on_login_toggled(self, checked: bool) -> None:
        # Settings.save() reconciles the registry entry — see apply_start_on_login.
        self._settings.start_on_login = bool(checked)
        self._settings.save()

    def _on_hk_mode_changed(self, _idx: int) -> None:
        mode = self._hk_mode_combo.currentData() or "hold"
        if mode == self._settings.hotkey_mode:
            return
        self._settings.hotkey_mode = mode
        self._settings.save()  # emits `changed` for live-apply
        log_app.info("hotkey mode -> %s", mode)

    # --- editable hotkey + ducking handlers ---------------------------

    _HOTKEY_DEFAULTS = {
        "quit":    "ctrl+windows+q",
        "repaste": "ctrl+windows+b",
        "llm":     "ctrl+windows+l",
        "note":    "ctrl+windows+z",
    }

    def _on_hotkey_changed(self, slot: str, edit: "QKeySequenceEdit") -> None:
        new_combo = keyseq_to_combo(edit.keySequence())
        if not new_combo:
            # User cleared the field — fall back to the default rather than
            # leaving the slot unbound (which would silently disable it).
            new_combo = self._HOTKEY_DEFAULTS.get(slot, "")
            edit.setKeySequence(combo_to_keyseq(new_combo))
        attr = f"hotkey_{slot}"
        if getattr(self._settings, attr, "") == new_combo:
            return
        setattr(self._settings, attr, new_combo)
        self._settings.save()  # triggers HotkeyWatcher.apply_settings via changed
        log_app.info("hotkey %s -> %s", slot, new_combo)

    def _reset_hotkey(self, slot: str) -> None:
        default = self._HOTKEY_DEFAULTS.get(slot, "")
        edit = self._hotkey_edits.get(slot)
        if edit is None:
            return
        edit.setKeySequence(combo_to_keyseq(default))
        self._on_hotkey_changed(slot, edit)

    def _on_duck_enabled_toggled(self, checked: bool) -> None:
        self._settings.duck_enabled = bool(checked)
        self._settings.save()
        log_app.info("audio ducking %s", "enabled" if checked else "disabled")

    def _on_duck_level_changed(self, value: int) -> None:
        v = max(0, min(100, int(value)))
        if hasattr(self, "_duck_label"):
            self._duck_label.setText(f"{v}%")
        if v == self._settings.duck_level_pct:
            return
        self._settings.duck_level_pct = v
        self._settings.save()

    # --- mic passthrough handlers --------------------------------------

    def _make_pt_output_combo(self) -> QComboBox:
        self._pt_output_combo = _NoWheelComboBox()
        self._populate_pt_output_combo()
        self._pt_output_combo.currentIndexChanged.connect(self._on_pt_output_changed)
        return self._pt_output_combo

    def _populate_pt_output_combo(self) -> None:
        """(Re)build the passthrough output dropdown, same shape as the
        microphone picker: 'System default' first, then real output devices."""
        target = self._settings.passthrough_output_name
        self._pt_output_combo.blockSignals(True)
        self._pt_output_combo.clear()
        self._pt_output_combo.addItem("System default", userData=None)
        idx_to_select = 0
        devices = list_output_devices()
        for i, (d, key) in enumerate(zip(devices, output_device_keys(devices)), start=1):
            label = d["name"]
            if d.get("api"):
                label += f"  [{d['api'].replace('Windows ', '')}]"
            if d["default"]:
                label += "  (system default)"
            self._pt_output_combo.addItem(label, userData=key)
            if target and key == target:
                idx_to_select = i
        self._pt_output_combo.setCurrentIndex(idx_to_select)
        self._pt_output_combo.blockSignals(False)

    def _on_pt_enabled_toggled(self, checked: bool) -> None:
        self._settings.passthrough_enabled = bool(checked)
        self._settings.save()   # emits `changed` -> App._apply_passthrough_settings
        log_app.info("mic passthrough %s", "enabled" if checked else "disabled")

    def _on_pt_level_changed(self, value: int) -> None:
        v = max(0, min(100, int(value)))
        if hasattr(self, "_pt_label"):
            self._pt_label.setText(f"{v}%")
        if v == self._settings.passthrough_level_pct:
            return
        self._settings.passthrough_level_pct = v
        # Apply live so the slider is audible while dragging, then debounce the
        # write — the same shape as the input-gain slider above.
        if self._passthrough is not None:
            self._passthrough.set_level_pct(v)
        if not hasattr(self, "_pt_save_timer"):
            self._pt_save_timer = QTimer(self)
            self._pt_save_timer.setSingleShot(True)
            self._pt_save_timer.timeout.connect(self._settings.save)
        self._pt_save_timer.start(300)

    def _on_pt_output_changed(self, _idx: int) -> None:
        name = self._pt_output_combo.currentData()  # str or None
        if name == self._settings.passthrough_output_name:
            return
        self._settings.passthrough_output_name = name
        self._settings.save()
        log_app.info("passthrough output -> %s", name if name else "system default")

    def _on_pt_mute_toggled(self, checked: bool) -> None:
        self._settings.passthrough_mute_while_dictating = bool(checked)
        self._settings.save()

    # --- transcription engine (ElevenLabs Scribe) handlers -------------

    def _on_stt_backend_changed(self, _idx: int) -> None:
        backend = self._stt_backend_combo.currentData() or "local"
        if backend == self._settings.stt_backend:
            return
        self._settings.stt_backend = backend
        self._settings.save()
        log_app.info("transcription backend -> %s", backend)
        self._refresh_stt_status()

    def _on_eleven_clean_toggled(self, checked: bool) -> None:
        self._settings.eleven_clean_transcript = bool(checked)
        self._settings.save()
        log_app.info("cloud transcript clean-up %s", "on" if checked else "off")

    def _on_eleven_key_changed(self) -> None:
        new_key = self._eleven_key_edit.text().strip()
        if new_key == self._settings.eleven_api_key:
            return
        self._settings.eleven_api_key = new_key
        self._settings.save()

    def _on_eleven_key2_changed(self) -> None:
        new_key = self._eleven_key2_edit.text().strip()
        if new_key == self._settings.eleven_api_key_backup:
            return
        self._settings.eleven_api_key_backup = new_key
        self._settings.save()
        self._refresh_stt_status()

    def _on_eleven_keys_extra_changed(self) -> None:
        keys = [ln.strip() for ln in
                self._eleven_keys_extra_edit.toPlainText().splitlines() if ln.strip()]
        if keys == self._settings.eleven_api_keys_extra:
            return
        self._settings.eleven_api_keys_extra = keys
        self._settings.save()
        log_app.info("ElevenLabs extra keys -> %d configured (pool size %d)",
                     len(keys), len(self._settings.eleven_keys()))
        self._refresh_stt_status()

    def _on_eleven_test_clicked(self) -> None:
        self._eleven_test_btn.setEnabled(False)
        self._eleven_test_status.setText("Testing…")
        self._eleven_test_status.setStyleSheet("")
        # Read the fields straight from the UI (don't depend on editingFinished /
        # the debounce timer) and persist them so real dictation uses the same
        # values. Extras are included, so the button tests the whole pool.
        primary = self._eleven_key_edit.text().strip()
        backup = self._eleven_key2_edit.text().strip()
        extra = [ln.strip() for ln in
                 self._eleven_keys_extra_edit.toPlainText().splitlines() if ln.strip()]
        if (primary != self._settings.eleven_api_key
                or backup != self._settings.eleven_api_key_backup
                or extra != self._settings.eleven_api_keys_extra):
            self._settings.eleven_api_key = primary
            self._settings.eleven_api_key_backup = backup
            self._settings.eleven_api_keys_extra = extra
            self._settings.save()
        model_id = self._settings.eleven_model_id or ELEVEN_DEFAULT_MODEL
        labels = ["Primary", "Backup"] + ["Key %d" % (i + 3) for i in range(len(extra))]
        targets = [(lbl, k) for lbl, k in zip(labels, [primary, backup] + extra) if k]
        if not targets:
            self._set_eleven_test_result(False, "Enter an API key first")
            return
        log_app.info("eleven test: starting (%d key(s), model=%s)", len(targets), model_id)

        # ~1s quiet tone per key — a real round-trip that proves each key + its
        # Speech-to-Text permission. A key that passes is cleared from the
        # exhausted list (re-enabled), which is how a refilled key comes back.
        # Result is delivered via _eleven_test_done (NOT QTimer.singleShot, which
        # never fires from this event-loop-less worker thread).
        def worker():
            n = int(SAMPLE_RATE * 1.0)
            t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
            tone = (0.02 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
            parts: list[str] = []
            all_ok = True
            for label, key in targets:
                try:
                    ElevenLabsScribeClient(key, model_id=model_id, timeout=20.0).transcribe(tone)
                    self._settings.clear_eleven_exhausted(key)
                    parts.append(f"{label} ✓")
                    log_app.info("eleven test: %s OK", label)
                except ElevenAuthError as e:
                    # The client maps 401/402/429 to this (not HTTPError).
                    all_ok = False
                    if e.code == 429:
                        why = "rate-limited"
                    elif e.status == "quota_exceeded":
                        why = "out of credits"
                    else:
                        why = f"unauthorized ({e.status or e.code})"
                    parts.append(f"{label} ✗ ({why})")
                    log_app.warning("eleven test: %s HTTP %s %s", label, e.code, e.status)
                except urllib.error.HTTPError as e:
                    all_ok = False
                    why = ("forbidden" if e.code == 403 else f"HTTP {e.code}")
                    parts.append(f"{label} ✗ ({why})")
                    log_app.warning("eleven test: %s HTTP %s %s", label, e.code, e.reason)
                except urllib.error.URLError as e:
                    all_ok = False
                    parts.append(f"{label} ✗ (no connection)")
                    log_app.warning("eleven test: %s URLError %s", label, e.reason)
                except Exception:
                    all_ok = False
                    parts.append(f"{label} ✗")
                    log_app.exception("eleven test: %s failed", label)
            self._eleven_test_done.emit(all_ok, " · ".join(parts))

        threading.Thread(target=worker, daemon=True, name="eleven-test").start()

    def _set_eleven_test_result(self, ok: bool, message: str) -> None:
        self._eleven_test_btn.setEnabled(True)
        self._eleven_test_status.setText(message)
        palette = THEME_PALETTES.get(self._settings.theme) or THEME_PALETTES["dark"]
        color = palette["success"] if ok else palette["danger"]
        self._eleven_test_status.setStyleSheet(f"color: {color};")
        self._refresh_stt_status()

    def _refresh_stt_status(self) -> None:
        """Repaint the engine-status line: which backend is live, and the state
        of each ElevenLabs key. Cheap enough to run on a 5 s timer."""
        if not hasattr(self, "_stt_engine_status"):
            return
        s = self._settings
        engine, reason = s.active_stt_engine()
        palette = THEME_PALETTES.get(s.theme) or THEME_PALETTES["dark"]
        if engine == "local":
            head = "Now using: LOCAL WHISPER — %s" % reason
            # Cloud selected but serving local = something is wrong (red).
            # Local deliberately selected, or auto routing = informational.
            color = (palette["danger"] if s.stt_backend == "elevenlabs"
                     else palette["text_dim"])
        else:
            head = "Now using: %s" % reason
            color = palette["success"]
        bits: list[str] = []
        now = time.time()
        cooling = s.eleven_exhausted_active()
        for i, key in enumerate(s.eleven_keys()):
            label = "Key #%d (…%s)" % (i + 1, key[-4:])
            known = s.eleven_credits_known(key)
            if key in cooling:
                benched = s.eleven_exhausted_at.get(key, 0.0)
                until = s.eleven_retry_after.get(
                    key, benched + ELEVEN_EXHAUSTED_RETRY_SEC)
                left = max(0.0, until - now)
                if left >= 3600:
                    when = "retries in %dh %dm" % (left // 3600, (left % 3600) // 60)
                elif left >= 60:
                    when = "retries in %dm" % (left // 60)
                else:
                    when = "retries in %ds" % int(left)
                # A short total cooldown means we benched it for back-pressure
                # (429), not a spent quota — don't mislabel a healthy key.
                short = (until - benched) <= (ELEVEN_RATELIMIT_RETRY_SEC * 2)
                detail = ("rate-limited, %s" if short else "out of credits, %s") % when
                if known is not None:
                    detail += " · %.0f credits left" % known
            elif known is not None:
                detail = "%.0f credits left" % known
            else:
                detail = "ready"
            bits.append("%s: %s" % (label, detail))
        if not bits:
            bits.append("No ElevenLabs key configured.")
        self._stt_engine_status.setText(head + "\n" + "  ·  ".join(bits))
        self._stt_engine_status.setStyleSheet("color: %s;" % color)

    # --- LLM post-processing handlers ----------------------------------

    def _on_llm_enabled_toggled(self, checked: bool) -> None:
        self._settings.llm_enabled = bool(checked)
        self._settings.save()
        log_app.info("LLM post-processing %s", "enabled" if checked else "disabled")

    def _on_llm_url_changed(self) -> None:
        new_url = self._llm_url_edit.text().strip() or "http://localhost:1234/v1"
        if new_url == self._settings.llm_base_url:
            return
        self._settings.llm_base_url = new_url
        self._settings.save()

    def _on_llm_model_changed(self) -> None:
        new_model = self._llm_model_edit.text().strip()
        if new_model == self._settings.llm_model:
            return
        self._settings.llm_model = new_model
        self._settings.save()

    def _on_llm_key_changed(self) -> None:
        new_key = self._llm_key_edit.text()
        if new_key == self._settings.llm_api_key:
            return
        self._settings.llm_api_key = new_key
        self._settings.save()

    def _on_llm_prompt_changed(self, _idx: int) -> None:
        pid = self._llm_prompt_combo.currentData() or "cleanup"
        if pid == self._settings.llm_default_prompt:
            return
        self._settings.llm_default_prompt = pid
        self._settings.save()

    def _on_llm_auto_toggled(self, checked: bool) -> None:
        self._settings.llm_auto_apply = bool(checked)
        self._settings.save()
        log_app.info("LLM auto-apply %s", "on" if checked else "off")

    def _on_llm_test_clicked(self) -> None:
        self._llm_test_btn.setEnabled(False)
        self._llm_test_status.setText("Testing…")
        # Clear any green/red inline style from a previous run so the
        # objectName-driven QSS (theme-aware dim color) takes effect.
        self._llm_test_status.setStyleSheet("")

        # Run in a worker thread so the UI doesn't freeze on slow endpoints.
        # Result is delivered via _llm_test_done (NOT QTimer.singleShot, which
        # never fires from this event-loop-less worker thread).
        def worker():
            client = LLMClient(
                base_url=self._settings.llm_base_url,
                model=self._settings.llm_model,
                api_key=self._settings.llm_api_key,
            )
            try:
                ok = client.ping()
                if not ok:
                    self._llm_test_done.emit(False, "Endpoint reachable but returned non-2xx")
                    return
                # Also try a tiny chat to surface model-not-found / 401 etc.
                _ = client.chat(
                    "You are a test responder. Reply with the single word OK.",
                    "ping",
                )
                self._llm_test_done.emit(True, "Connected ✓")
            except urllib.error.URLError as e:
                self._llm_test_done.emit(False, f"Can't reach endpoint: {e.reason}")
            except Exception as e:
                self._llm_test_done.emit(False, f"Failed: {e}")

        threading.Thread(target=worker, daemon=True, name="llm-test").start()

    def _set_llm_test_result(self, ok: bool, message: str) -> None:
        self._llm_test_btn.setEnabled(True)
        self._llm_test_status.setText(message)
        # Source the color from the current theme palette so success/fail
        # contrast properly against both light and dark backgrounds.
        palette = THEME_PALETTES.get(self._settings.theme) or THEME_PALETTES["dark"]
        color = palette["success"] if ok else palette["danger"]
        self._llm_test_status.setStyleSheet(f"color: {color};")

    # --- appearance handler --------------------------------------------

    def _on_theme_changed(self, _idx: int) -> None:
        new_theme = self._theme_combo.currentData() or "dark"
        if new_theme == self._settings.theme:
            return
        self._settings.theme = new_theme
        self._settings.save()  # emits `changed` → _apply_theme via connection
        log_app.info("theme -> %s", new_theme)

    # --- privacy & data management handlers ----------------------------

    @staticmethod
    def _format_retention(days: int) -> str:
        if days <= 0:
            return "forever"
        if days == 1:
            return "1 day"
        if days < 30:
            return f"{days} days"
        if days < 60:
            return "1 month"
        if days < 365:
            return f"~{days // 30} months"
        return "1 year"

    def _on_save_history_toggled(self, checked: bool) -> None:
        self._settings.save_history = bool(checked)
        self._settings.save()
        log_app.info("save_history %s", "on" if checked else "off")

    def _on_retention_changed(self, value: int) -> None:
        self._settings.history_retention_days = int(value)
        self._retention_label.setText(self._format_retention(value))
        # debounce save
        if not hasattr(self, "_ret_save_timer"):
            self._ret_save_timer = QTimer(self)
            self._ret_save_timer.setSingleShot(True)
            self._ret_save_timer.timeout.connect(self._settings.save)
        self._ret_save_timer.start(400)

    def _on_export_history_clicked(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        entries = self._log.entries()
        if not entries:
            QMessageBox.information(self, "Export history",
                                    "No transcripts to export yet.")
            return
        default_name = f"tellur-history-{time.strftime('%Y%m%d-%H%M%S')}"
        path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export history",
            str(Path.home() / default_name),
            "Markdown (*.md);;Plain text (*.txt);;JSON (*.json);;CSV (*.csv)",
        )
        if not path:
            return
        # Pick format from selected filter (more reliable than user extension).
        ext_map = {
            "Markdown (*.md)": "md",
            "Plain text (*.txt)": "txt",
            "JSON (*.json)": "json",
            "CSV (*.csv)": "csv",
        }
        fmt = ext_map.get(selected_filter, Path(path).suffix.lstrip(".") or "txt")
        if not Path(path).suffix:
            path = path + "." + fmt
        try:
            n = export_history(entries, Path(path), fmt)
            QMessageBox.information(self, "Export complete",
                                    f"Exported {n} entries to:\n{path}")
        except Exception as e:
            log_app.exception("history export failed")
            QMessageBox.warning(self, "Export failed", f"{type(e).__name__}: {e}")

    def _on_import_dict_clicked(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Import dictionary (replacements.json)",
            str(Path.home()),
            "JSON (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
            if not isinstance(data, dict):
                raise ValueError("file must contain a JSON object of {key: value} mappings")
        except Exception as e:
            QMessageBox.warning(self, "Import failed", f"{type(e).__name__}: {e}")
            return
        # Merge: imported entries win on conflict.
        local_path = Path(__file__).resolve().parent / REPLACEMENTS_FILE
        merged: dict = {}
        try:
            if local_path.exists():
                existing = json.loads(local_path.read_text(encoding="utf-8-sig"))
                if isinstance(existing, dict):
                    merged.update(existing)
        except Exception:
            log_app.exception("failed to read existing replacements during import")
        added = 0
        overwritten = 0
        for k, v in data.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            if k in merged and merged[k] != v:
                overwritten += 1
            elif k not in merged:
                added += 1
            merged[k] = v
        try:
            local_path.write_text(
                json.dumps(merged, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as e:
            QMessageBox.warning(self, "Import failed", f"Couldn't write: {e}")
            return
        log_app.info("dictionary import: +%d new, %d overwritten, %d total",
                     added, overwritten, len(merged))
        QMessageBox.information(
            self, "Import complete",
            f"Added {added} new rules, overwrote {overwritten} existing.\n"
            f"Dictionary now contains {len(merged)} entries.\n\n"
            f"(Reload happens automatically on the next transcription.)",
        )

    def _on_open_data_clicked(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(str(DATA_DIR))  # noqa: SLF001
        except Exception:
            log_app.exception("failed to open data folder")

    def _on_clear_all_clicked(self) -> None:
        reply = QMessageBox.question(
            self,
            "Clear all data?",
            "This will permanently delete:\n\n"
            "• Your transcript history (history.json)\n"
            "• Your custom dictionary (replacements.json)\n\n"
            "Settings will be kept. This cannot be undone.\n\nProceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Clear history through TranscriptLog so listeners refresh.
        self._log.clear()
        # Wipe the on-disk replacements file. The in-memory Replacements
        # object will see it gone on its next reload (per-transcription).
        try:
            (Path(__file__).resolve().parent / REPLACEMENTS_FILE).unlink(missing_ok=True)
        except Exception:
            log_app.exception("failed to delete replacements.json")
        QMessageBox.information(
            self, "Cleared",
            "History and dictionary cleared. Settings retained.",
        )

    # --- per-context vocabulary handlers -------------------------------

    def _on_open_ctx_dir_clicked(self) -> None:
        try:
            PER_CONTEXT_DICT_DIR.mkdir(parents=True, exist_ok=True)
            os.startfile(str(PER_CONTEXT_DICT_DIR))  # noqa: SLF001
        except Exception:
            log_app.exception("failed to open per-context folder")

    def _on_show_ctx_clicked(self) -> None:
        ctx = get_foreground_process_basename()
        if ctx is None:
            self._ctx_status.setText(
                "Couldn't identify the foreground app — try clicking somewhere "
                "outside Tellur first, then come back and click this button."
            )
            return
        existing = PER_CONTEXT_DICT_DIR / f"{ctx}.json"
        if existing.exists():
            self._ctx_status.setText(
                f"Foreground app: <b>{ctx}</b> — overlay file already exists "
                f"at <code>{existing.name}</code>."
            )
        else:
            self._ctx_status.setText(
                f"Foreground app: <b>{ctx}</b> — create "
                f"<code>{ctx}.json</code> in the per-app folder to add an overlay."
            )
        self._ctx_status.setTextFormat(Qt.TextFormat.RichText)

    # --- integration & automation handlers -----------------------------

    def _on_send_enter_toggled(self, checked: bool) -> None:
        self._settings.send_after_paste = bool(checked)
        self._settings.save()
        log_app.info("send-after-paste %s", "on" if checked else "off")

    def _on_terminal_paste_toggled(self, checked: bool) -> None:
        self._settings.terminal_aware_paste = bool(checked)
        self._settings.save()
        log_app.info("terminal-aware paste %s", "on" if checked else "off")

    def _on_md_enabled_toggled(self, checked: bool) -> None:
        self._settings.markdown_save_enabled = bool(checked)
        self._settings.save()
        log_app.info("markdown daily save %s", "on" if checked else "off")

    def _on_md_folder_changed(self) -> None:
        new_folder = self._md_folder_edit.text().strip()
        if new_folder == self._settings.markdown_folder:
            return
        self._settings.markdown_folder = new_folder
        self._settings.save()

    def _on_md_open_clicked(self) -> None:
        folder = self._settings.markdown_folder.strip() or str(DEFAULT_MARKDOWN_DIR)
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            os.startfile(folder)  # noqa: SLF001 — Windows-only API, by design
        except Exception:
            log_app.exception("failed to open markdown folder")

    def _on_wh_enabled_toggled(self, checked: bool) -> None:
        self._settings.webhook_enabled = bool(checked)
        self._settings.save()
        log_app.info("webhook %s", "on" if checked else "off")

    def _on_wh_url_changed(self) -> None:
        new_url = self._wh_url_edit.text().strip()
        if new_url == self._settings.webhook_url:
            return
        self._settings.webhook_url = new_url
        self._settings.save()

    def _on_wh_template_changed(self) -> None:
        new_tpl = self._wh_tpl_edit.text()
        if new_tpl == self._settings.webhook_template:
            return
        self._settings.webhook_template = new_tpl
        self._settings.save()

    # --- audio input handlers ------------------------------------------

    def _populate_device_combo(self) -> None:
        """(Re)build the input device dropdown. First entry is always
        'System default'; rest are real input devices, with the default
        device tagged."""
        target = self._settings.input_device_name
        self._device_combo.blockSignals(True)
        self._device_combo.clear()
        self._device_combo.addItem("System default", userData=None)
        idx_to_select = 0
        for i, d in enumerate(list_input_devices(), start=1):
            label = d["name"]
            if d["default"]:
                label += "  (system default)"
            self._device_combo.addItem(label, userData=d["name"])
            if target and d["name"] == target:
                idx_to_select = i
        self._device_combo.setCurrentIndex(idx_to_select)
        self._device_combo.blockSignals(False)

    def _on_device_changed(self, _idx: int) -> None:
        name = self._device_combo.currentData()  # str or None
        if name == self._settings.input_device_name:
            return
        self._settings.input_device_name = name
        self._settings.save()  # emits `changed` → App._apply_audio_settings
        log_app.info("input device -> %s", name if name else "system default")

    def _on_gain_changed(self, value: int) -> None:
        gain = max(0.1, min(10.0, value / 100.0))
        self._gain_label.setText(f"{gain:.2f}x")
        self._settings.input_gain = gain
        # debounce saves
        if not hasattr(self, "_gain_save_timer"):
            self._gain_save_timer = QTimer(self)
            self._gain_save_timer.setSingleShot(True)
            self._gain_save_timer.timeout.connect(self._settings.save)
        self._gain_save_timer.start(300)
        # Apply gain live without waiting for the save (settings.changed only
        # fires from save()), so the live meter reflects the new value now.
        if self._recorder is not None:
            self._recorder.set_gain(gain)

    def _tick_level_meter(self) -> None:
        if self._recorder is None:
            return
        # Convert RMS (≈0..1) to the 0..1000 range with sensitivity-scaled
        # multiplier so the meter responds similarly to the overlay bars.
        level = self._recorder.monitor_level * float(self._settings.sensitivity) * 10
        level = max(0, min(1000, int(level)))
        self._level_meter.setValue(level)

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

    # --- model picker handlers -----------------------------------------

    def _populate_model_combo(self) -> None:
        """(Re)build the model dropdown, annotating cached models with ✓."""
        current = self._model_combo.currentData() if self._model_combo.count() else None
        target = current or self._settings.model_name

        self._model_combo.blockSignals(True)
        self._model_combo.clear()
        for m in KNOWN_MODELS:
            label = m["label"]
            if m["name"] == DEFAULT_MODEL:
                label += " (default)"
            cached_suffix = "  ·  ✓ downloaded" if is_model_cached(m["name"]) else ""
            text = (
                f"{label}  ·  {_format_size_mb(m['size_mb'])}  ·  "
                f"{m['summary']}{cached_suffix}"
            )
            self._model_combo.addItem(text, userData=m["name"])
        idx = next(
            (i for i, m in enumerate(KNOWN_MODELS) if m["name"] == target),
            next(i for i, m in enumerate(KNOWN_MODELS) if m["name"] == DEFAULT_MODEL),
        )
        self._model_combo.setCurrentIndex(idx)
        self._model_combo.blockSignals(False)

    def _on_model_combo_changed(self, idx: int) -> None:
        name = self._model_combo.itemData(idx)
        if not name or name == self._settings.model_name:
            return
        self.model_switch_requested.emit(name)

    def on_model_switch_state(self, state: str, message: str) -> None:
        """Called by App when a model-switch worker reports progress."""
        if not hasattr(self, "_model_status"):
            return
        self._model_status.setText(message)
        if state == "loading":
            self._model_combo.setEnabled(False)
        elif state == "ready":
            self._model_combo.setEnabled(True)
            # Refresh dropdown so the just-downloaded model gets its ✓ marker.
            self._populate_model_combo()
        elif state == "error":
            # Revert the dropdown to the model that's actually still active.
            self._model_combo.blockSignals(True)
            idx = next(
                (i for i, m in enumerate(KNOWN_MODELS)
                 if m["name"] == self._settings.model_name),
                0,
            )
            self._model_combo.setCurrentIndex(idx)
            self._model_combo.blockSignals(False)
            self._model_combo.setEnabled(True)

    def on_model_download_progress(self, pct: int) -> None:
        """Show/hide and update the download progress bar.

        Contract for pct:
          -2  → indeterminate (busy animation, e.g. connecting or loading)
          -1  → hide
           0..100 → determinate percentage
        """
        if not hasattr(self, "_model_progress"):
            return
        bar = self._model_progress
        if pct == -1:
            bar.setVisible(False)
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setFormat("%p%")
        elif pct < 0:
            # Indeterminate: range (0,0) makes Qt animate a sliding chunk.
            bar.setRange(0, 0)
            bar.setFormat("")
            bar.setVisible(True)
        else:
            if bar.maximum() == 0:
                bar.setRange(0, 100)
                bar.setFormat("%p%")
            bar.setValue(pct)
            bar.setVisible(True)

    # --- window behavior -----------------------------------------------

    def closeEvent(self, event) -> None:
        # Closing the window just hides it — tray quit (or Ctrl+Win+Q) actually exits.
        event.ignore()
        self.hide()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        # Start the live mic monitor + meter tick when the panel is shown.
        if self._recorder is not None:
            self._recorder.start_monitor()   # deferred if a recording is live
            self._meter_tick.start()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        # Stop the live monitor + meter when the panel is hidden.
        self._meter_tick.stop()
        if hasattr(self, "_level_meter"):
            self._level_meter.setValue(0)
        if self._recorder is not None:
            self._recorder.stop_monitor()


# ===========================================================================
# orchestrator
# ===========================================================================
class App(QObject):
    ui_state = pyqtSignal(str)
    # "local" | "elevenlabs" — drives the overlay's engine badge. Emitted from
    # worker threads, so it must stay a signal rather than a direct call.
    ui_engine = pyqtSignal(str)
    ui_level = pyqtSignal(float)
    model_switch_state = pyqtSignal(str, str)   # (state, message)
    model_download_progress = pyqtSignal(int)   # 0..100, or -1 to hide bar

    def __init__(self):
        super().__init__()

        # Load settings FIRST so we can pick the user's chosen model up-front.
        here = Path(__file__).resolve().parent
        self.install_dir = here
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.settings = Settings(SETTINGS_FILE)
        self.log = TranscriptLog(HISTORY_FILE)
        self.notes = NotesLog(NOTES_FILE)
        self.replacements = Replacements(
            here / REPLACEMENTS_FILE,
            per_context_dir=PER_CONTEXT_DICT_DIR,
        )
        # Per-context vocabulary (v2.0): captured on hotkey press so the
        # foreground app at the start of the recording is what gets matched
        # — by the time transcription finishes the user may have switched
        # windows.
        self._current_context: str | None = None
        self.voice_commands = VoiceCommandProcessor()
        self.updater = Updater(self.install_dir)

        self.recorder = AudioRecorder()
        self.ducker = AudioDucker()
        self.passthrough = PassthroughMonitor()
        # Crash-safety: if a previous run was killed/hung mid-duck, restore the
        # apps it left quiet. Background thread so a pycaw hiccup can't delay
        # startup and won't fight an app the user already turned back up.
        threading.Thread(
            target=self.ducker.recover, daemon=True, name="duck-recover",
        ).start()
        self.engine = WhisperEngine(self.settings.model_name, settings=self.settings)
        self.overlay = Overlay()
        self.hotkey = HotkeyWatcher(settings=self.settings)
        self.injector = TextInjector(settings=self.settings)

        # Apply persisted audio settings to the recorder.
        self._apply_audio_settings()
        self.settings.changed.connect(self._apply_audio_settings)
        self._apply_passthrough_settings()
        self.settings.changed.connect(self._apply_passthrough_settings)

        # Apply persisted settings to the overlay; react to live changes.
        self.overlay.set_scale(self.settings.sensitivity)
        self.settings.changed.connect(
            lambda: self.overlay.set_scale(self.settings.sensitivity)
        )

        # Tray + panel — panel is hidden by default; opened from the tray.
        self.panel = MainPanel(self.log, self.notes, self.settings, self.updater,
                               self.recorder, self.passthrough)
        self.tray = TrayIcon(self.updater, self)
        if not QSystemTrayIcon.isSystemTrayAvailable():
            log_app.warning("system tray not available — tray icon will be inactive")
        self.tray.show_panel.connect(self._show_panel)
        self.tray.copy_last.connect(self._copy_last_to_clipboard)
        self.tray.quit_requested.connect(self.quit)
        self.tray.install_update.connect(self.updater.install_async)
        self.tray.refresh_ui_requested.connect(self.refresh_ui)
        self.updater.update_found.connect(self.tray.set_update_available)
        self.updater.quit_requested.connect(self.quit)

        # Model-switching: panel asks → App orchestrates → panel renders progress.
        self.panel.model_switch_requested.connect(self.switch_model_async)
        self.panel.refresh_ui_requested.connect(self.refresh_ui)
        self.model_switch_state.connect(self.panel.on_model_switch_state)
        self.model_download_progress.connect(self.panel.on_model_download_progress)

        self.hotkey.pressed.connect(self.on_press)
        self.hotkey.released.connect(self.on_release)
        self.hotkey.quit_requested.connect(self.quit)
        self.hotkey.cancel_requested.connect(self.on_cancel)
        self.hotkey.repaste_requested.connect(self.on_repaste_last)
        self.hotkey.llm_apply_requested.connect(self.on_llm_apply_last)
        self.hotkey.note_toggle_requested.connect(self.on_note_toggle)
        # Note recording state — separate from dictation recording. A note
        # toggle press flips this; on release we transcribe and save to
        # NotesLog (no paste).
        self._note_recording = False
        self._note_cancel = False
        # Apply hotkey mode from settings and react to live changes.
        self.hotkey.set_mode(self.settings.hotkey_mode)
        self.settings.changed.connect(
            lambda: self.hotkey.set_mode(self.settings.hotkey_mode)
        )
        # Re-bind configurable hotkeys whenever Settings is saved (user
        # edited a hotkey in the Settings tab).
        self.settings.changed.connect(self.hotkey.apply_settings)
        self.ui_state.connect(self.overlay.set_state)
        self.ui_engine.connect(self.overlay.set_engine)
        self.ui_level.connect(self.overlay.push_level)

        self._level_timer = QTimer(self)
        self._level_timer.setInterval(int(1000 / LEVEL_PUSH_HZ))
        self._level_timer.timeout.connect(self._tick_level)

        # serialize GPU calls; generation token lets stale state emits be skipped
        self._engine_lock = threading.Lock()
        # Generation counter — read/written across main and worker threads, so
        # protected by its own lock. Use the helpers _bump_gen / _is_latest.
        self._gen_lock = threading.Lock()
        self._latest_gen = 0

        # Cancel flag — set by on_cancel (Esc) during recording to skip transcribe.
        self._cancel_current_recording = False

        # Keep the comparatively large local Whisper model lazy when the cloud
        # backend is configured.  WhisperEngine.transcribe() already calls
        # load() on the local fallback path, so eagerly warming it here only
        # reserves several GB of commit for the entire login session even when
        # every transcription is served by ElevenLabs.
        # For "auto", warm up only if the GPU is free right now: an idle GPU
        # means the next clip decodes locally and should not pay the load cost;
        # a busy GPU at startup (logged in mid-game) means clips go to the
        # cloud anyway, and warming up would grab VRAM from the busy app.
        cloud_stt_ready = (
            self.settings.stt_backend == "elevenlabs"
            and bool(self.settings.eleven_keys())
        ) or (
            self.settings.stt_backend == "auto"
            and GPU_MONITOR.busy_reason() is not None
        )
        if cloud_stt_ready:
            log_engine.info(
                "local Whisper warm-up deferred (cloud STT will serve; the "
                "model loads on the first local decode)"
            )
        else:
            threading.Thread(
                target=self._warm_up,
                daemon=True,
                name="model-warmup",
            ).start()

        # Kick off a one-shot update check ~3s after startup so it doesn't
        # compete with model loading for network/CPU.
        QTimer.singleShot(3000, self.updater.check_async)

    def start(self) -> None:
        self.overlay.show()
        self.tray.show()
        self.hotkey.start()
        # Apply history retention on startup so a long downtime doesn't leave
        # very old entries hanging around when the user has set a cap.
        days = self.settings.history_retention_days
        if days > 0:
            self.log.purge_older_than(days * 86400)
        # Reconcile start-on-login registry entry with the saved setting.
        # Handles two cases: (1) install dir was moved, so the prior Run
        # value points at a stale path; (2) registry entry was cleared by
        # another tool. Cheap idempotent — only writes when state differs.
        if self.settings.start_on_login != is_start_on_login_enabled():
            apply_start_on_login(self.settings.start_on_login)
        log_app.info("started — tray icon visible")

    def quit(self) -> None:
        log_app.info("quit requested")
        # Restore other-app audio if we exit mid-recording — otherwise the
        # user's Discord / VLC / browser stays stuck at 5% until they
        # adjust it manually.
        try:
            self.ducker.restore()
        except Exception:
            log_app.debug("quit: ducker.restore failed", exc_info=True)
        # Close the passthrough streams explicitly — PortAudio callbacks run on
        # their own threads and can outlive an abrupt interpreter shutdown.
        try:
            self.passthrough.stop()
        except Exception:
            log_app.debug("quit: passthrough.stop failed", exc_info=True)
        # Release the global keyboard hooks (Ctrl+Win polls + the
        # Ctrl+Win+Q hotkey). The `keyboard` library spawns a low-level
        # Windows hook thread that can leak across abnormal exits otherwise.
        try:
            keyboard.unhook_all()
        except Exception:
            log_app.debug("keyboard.unhook_all failed", exc_info=True)
        # Hide tray icon explicitly so it disappears immediately rather than waiting
        # for Windows tray cleanup on process exit.
        try:
            self.tray.hide()
        except Exception:
            pass
        # Persist anything the async savers still hold (daemon threads die
        # with the interpreter otherwise — the last entry could be lost).
        for lg in (self.log, self.notes):
            try:
                lg.flush()
            except Exception:
                log_app.debug("quit: log flush failed", exc_info=True)
        QApplication.instance().quit()

    def refresh_ui(self) -> None:
        """Reset the overlay + tray + hotkey state when the visible UI gets
        wedged (overlay disappeared, recording state stuck, tray icon
        missing). Safe to call from the tray menu or the Settings button.
        Does NOT touch the Whisper model or in-flight transcription threads —
        a transcribe in progress will still complete and emit its result; we
        only rebuild the surface visuals + hotkey watchers around it.
        """
        log_app.info("refresh_ui: rebuilding overlay + tray + hotkey state")
        # 1. Stop the live-level timer so we're not pushing into an overlay
        #    we're about to replace.
        try:
            self._level_timer.stop()
        except Exception:
            log_app.debug("refresh_ui: level timer stop failed", exc_info=True)

        # 2. Bump generation so any in-flight transcribe stops emitting UI
        #    state — its result still gets saved to history/notes, but it
        #    won't paint "done" / "idle" over our fresh overlay.
        self._bump_gen()

        # 2b. Restore any ducked audio sessions — we're tearing down the
        #     recording surface, so leaving other apps' volume at 5% would
        #     be a nasty surprise.
        try:
            self.ducker.restore()
        except Exception:
            log_audio.debug("refresh_ui: ducker.restore failed", exc_info=True)

        # 3. If we were mid-recording (either dictation or note), abandon
        #    cleanly. Audio buffer is dropped; mic stream is closed.
        if self._note_recording:
            try:
                self.recorder.stop()
            except Exception:
                log_audio.debug("refresh_ui: recorder.stop failed", exc_info=True)
            self._note_recording = False
            self._note_cancel = False
            self.hotkey.set_note_in_progress(False)
        if self.hotkey.recording:
            self._cancel_current_recording = True
            try:
                self.recorder.stop()
            except Exception:
                log_audio.debug("refresh_ui: recorder.stop failed", exc_info=True)
            self.hotkey.force_release()

        # 4. Rebuild the overlay. Just hiding+showing isn't enough when the
        #    Qt window has gone funky (state lost, parent destroyed); a
        #    fresh widget is the most reliable reset.
        old_overlay = self.overlay
        try:
            self.ui_state.disconnect(old_overlay.set_state)
        except Exception:
            pass
        try:
            self.ui_engine.disconnect(old_overlay.set_engine)
        except Exception:
            pass
        try:
            self.ui_level.disconnect(old_overlay.push_level)
        except Exception:
            pass
        try:
            old_overlay.hide()
            old_overlay.deleteLater()
        except Exception:
            log_app.debug("refresh_ui: overlay teardown noise", exc_info=True)
        self.overlay = Overlay()
        self.overlay.set_scale(self.settings.sensitivity)
        self.ui_state.connect(self.overlay.set_state)
        self.ui_engine.connect(self.overlay.set_engine)
        self.ui_level.connect(self.overlay.push_level)
        self.overlay.show()
        self.ui_state.emit("idle")

        # 5. Make sure the tray icon is still visible — Explorer can drop tray
        #    icons after an Explorer.exe restart and we'd never know.
        try:
            if not self.tray.isVisible():
                self.tray.show()
        except Exception:
            log_app.debug("refresh_ui: tray re-show failed", exc_info=True)

        # 6. Force-rebuild the panel's history + notes lists in case a list
        #    rendering hiccup left rows in a bad state.
        try:
            self.panel._refresh_list()
            self.panel._refresh_notes_list()
        except Exception:
            log_app.debug("refresh_ui: panel refresh failed", exc_info=True)

        log_app.info("refresh_ui: done")

    def _show_panel(self) -> None:
        # Position once per session: horizontally centered on the primary
        # screen, vertically a bit above the midline (so the window feels
        # eye-level rather than sinking toward the taskbar). Once placed,
        # we leave it alone — if the user drags it, that position sticks
        # across subsequent hides/shows.
        if not getattr(self, "_panel_positioned", False):
            screen = QGuiApplication.primaryScreen()
            if screen is not None:
                geo = screen.availableGeometry()  # excludes the taskbar
                target_w = self.panel.width() or 680
                target_h = self.panel.height() or 580
                # Cap height to the available work-area so the bottom never
                # ends up under the taskbar on small screens.
                if target_h > geo.height() - 32:
                    target_h = geo.height() - 32
                    self.panel.resize(target_w, target_h)
                x = geo.center().x() - target_w // 2
                # Nudge ~8% of the work-area height above center so the
                # window feels like a mid-screen popup, not a bottom drawer.
                y = geo.center().y() - target_h // 2 - int(geo.height() * 0.08)
                # Clamp so we never start partially off-screen.
                x = max(geo.left() + 8, min(x, geo.right() - target_w - 8))
                y = max(geo.top() + 8, min(y, geo.bottom() - target_h - 8))
                self.panel.move(x, y)
            self._panel_positioned = True
        self.panel.show()
        self.panel.raise_()
        self.panel.activateWindow()

    def _apply_audio_settings(self) -> None:
        """Reconcile recorder state with the latest Settings.input_*.
        Resolves the persisted device NAME back to a PortAudio index (so the
        choice survives index reshuffles between sessions / device hotplug)."""
        # Resolve device name → index. None means "follow system default".
        target_name = self.settings.input_device_name
        idx: int | None = None
        if target_name:
            for d in list_input_devices():
                if d["name"] == target_name:
                    idx = d["index"]
                    break
            if idx is None:
                log_audio.warning(
                    "configured input device %r not found — using system default",
                    target_name,
                )
        self.recorder.set_device(idx)
        self.recorder.set_gain(self.settings.input_gain)

    def _apply_passthrough_settings(self) -> None:
        """Reconcile the passthrough monitor with the latest Settings. Resolves
        both device NAMES back to PortAudio indexes, for the same reason
        _apply_audio_settings does: indexes reshuffle between sessions."""
        in_idx = None
        if self.settings.input_device_name:
            for d in list_input_devices():
                if d["name"] == self.settings.input_device_name:
                    in_idx = d["index"]
                    break
        out_idx = find_output_device(self.settings.passthrough_output_name)
        if self.settings.passthrough_output_name and out_idx is None:
            log_audio.warning(
                "configured passthrough output %r not found — using system default",
                self.settings.passthrough_output_name,
            )

        self.passthrough.set_level_pct(self.settings.passthrough_level_pct)
        self.passthrough.set_devices(in_idx, out_idx)

        if self.settings.passthrough_enabled and not self.passthrough.running:
            if not self.passthrough.start():
                # Turning the setting back off keeps the checkbox honest rather
                # than showing "on" for a monitor that never opened.
                self.settings.passthrough_enabled = False
                # This also runs from __init__, before the panel is built, so
                # the checkbox is only corrected once it actually exists.
                panel = getattr(self, "panel", None)
                if panel is not None and hasattr(panel, "_pt_enabled_cb"):
                    panel._pt_enabled_cb.setChecked(False)
        elif not self.settings.passthrough_enabled and self.passthrough.running:
            self.passthrough.stop()

    def _duck_if_enabled(self) -> None:
        """Lower other apps' volume if the user has enabled ducking. No-op
        otherwise — pycaw isn't even imported until we get here."""
        if not self.settings.duck_enabled:
            return
        level = max(0, min(100, int(self.settings.duck_level_pct))) / 100.0
        try:
            self.ducker.duck(level=level)
        except Exception:
            log_audio.exception("ducker.duck raised")

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
            # Report the resting state once load is done so the UI can drop
            # any "Loading…" indicator the settings tab might be showing.
            self.model_switch_state.emit("ready", f"Active: {self.engine.name}")
        except Exception:
            log_engine.exception("model load failed")
            self.ui_state.emit("error")
            # Auto-reset the overlay to idle after a few seconds so it doesn't
            # stay orange forever. The next dictation attempt will retry the
            # load and either succeed or re-emit error.
            def _reset() -> None:
                time.sleep(3.0)
                self.ui_state.emit("idle")
            threading.Thread(
                target=_reset, daemon=True, name="warmup-error-reset",
            ).start()

    def switch_model_async(self, new_name: str) -> None:
        if new_name == self.engine.name:
            return
        threading.Thread(
            target=self._switch_model_worker, args=(new_name,),
            daemon=True, name=f"model-switch-{new_name}",
        ).start()

    def _switch_model_worker(self, new_name: str) -> None:
        size_mb = next(
            (m["size_mb"] for m in KNOWN_MODELS if m["name"] == new_name), 100,
        )
        cached = is_model_cached(new_name)

        if cached:
            log_engine.info("model %s already cached; instant switch", new_name)
            self.model_switch_state.emit("loading", f"Switching to {new_name}…")
            self.model_download_progress.emit(-1)
        else:
            # Pre-download phase: HF Hub does API + ETag + connection setup
            # before any bytes flow. Show an indeterminate bar so the user
            # sees activity rather than a frozen 0%.
            log_engine.info(
                "model %s not cached; downloading (~%d MB)", new_name, size_mb,
            )
            self.model_switch_state.emit(
                "loading",
                f"Connecting to download {new_name} "
                f"({_format_size_mb(size_mb)})…",
            )
            self.model_download_progress.emit(-2)  # indeterminate

            repo_id = _MODEL_REPO.get(
                new_name, f"Systran/faster-whisper-{new_name}",
            )
            try:
                downloaded = self._download_with_progress(
                    repo_id, new_name, size_mb,
                )
            except Exception as e:
                log_engine.exception("download failed for %s", new_name)
                self.model_switch_state.emit(
                    "error", f"Download failed: {e}",
                )
                self.model_download_progress.emit(-1)
                return

            if not downloaded:
                # HF API didn't accept our progress hook (older version);
                # fall back to letting WhisperModel handle the download.
                # Show indeterminate so it's clear something's happening.
                self.model_switch_state.emit(
                    "loading", f"Downloading {new_name}…",
                )
                self.model_download_progress.emit(-2)

            # Final: load weights from cache into VRAM (1–3 s for large
            # models; no per-byte progress available so we stay
            # indeterminate).
            self.model_switch_state.emit(
                "loading", f"Loading {new_name} into memory…",
            )
            self.model_download_progress.emit(-2)

        try:
            self.engine.switch_to(new_name)
        except Exception as e:
            log_engine.exception("model switch failed")
            self.model_switch_state.emit("error", f"Failed to load {new_name}: {e}")
            self.model_download_progress.emit(-1)
            return

        self.settings.model_name = new_name
        self.settings.save()
        self.model_download_progress.emit(-1)
        self.model_switch_state.emit("ready", f"Active: {new_name}")

    def _download_with_progress(
        self, repo_id: str, model_name: str, expected_mb: int,
    ) -> bool:
        """Trigger an HF snapshot download with polling-based UI feedback.

        We do NOT hook into tqdm — HF Hub's xet backend bypasses tqdm_class for
        the actual byte transfer, so we'd never see progress. Polling the
        model's cache directory works regardless of transport, including xet
        (which still writes the final blob into the model's blobs/ dir).

        Returns True on success.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            log_engine.warning("huggingface_hub unavailable; cannot pre-download")
            return False

        cache_dir = _model_cache_dir(model_name)
        expected_bytes = max(expected_mb * 1024 * 1024, 1)
        stop_event = threading.Event()
        app_self = self

        def _poll() -> None:
            last_pct = -1
            # 500 KB threshold: gates the transition from the caller's
            # "Connecting…" indeterminate state to our determinate bar.
            BYTES_THRESHOLD = 500_000
            while not stop_event.is_set():
                current = _dir_size_bytes(cache_dir)
                if current >= BYTES_THRESHOLD:
                    pct = min(int((current / expected_bytes) * 100), 99)
                    if pct != last_pct:
                        current_mb = current // (1024 * 1024)
                        app_self.model_switch_state.emit(
                            "loading",
                            f"Downloading {model_name}…  "
                            f"{current_mb} / {expected_mb} MB",
                        )
                        app_self.model_download_progress.emit(pct)
                        last_pct = pct
                stop_event.wait(0.25)

        threading.Thread(
            target=_poll, daemon=True, name=f"model-poll-{model_name}",
        ).start()
        log_engine.info("downloading %s (repo=%s)", model_name, repo_id)
        try:
            snapshot_download(repo_id=repo_id)
        finally:
            stop_event.set()
        log_engine.info("download finished for %s", model_name)
        return True

    # --- generation counter helpers (lock-protected against cross-thread reads)

    def _bump_gen(self) -> int:
        with self._gen_lock:
            self._latest_gen += 1
            return self._latest_gen

    def _is_latest(self, gen: int) -> bool:
        with self._gen_lock:
            return gen == self._latest_gen

    def on_press(self) -> None:
        # Bump gen FIRST so any pending reset timer from a prior release becomes
        # stale and skips its emit — otherwise it can clobber our "recording".
        gen = self._bump_gen()
        # Reset the per-recording cancel flag; Esc during this session will set it.
        self._cancel_current_recording = False
        # Capture the foreground app NOW so per-context vocab matches what the
        # user is actually dictating into — by the time transcription returns
        # they may have switched windows.
        self._current_context = get_foreground_process_basename()
        log_hotkey.debug("press (gen=%d, mode=%s, context=%s)",
                         gen, self.settings.hotkey_mode, self._current_context)
        # Silence the hear-through monitor before opening the recording
        # stream: hearing yourself while dictating is distracting, and it also
        # takes the passthrough off the mic if this driver dislikes a second
        # reader. Muting (rather than stopping) keeps the resume instant.
        if self.settings.passthrough_mute_while_dictating:
            self.passthrough.set_muted(True)
        try:
            self.recorder.start()
        except Exception:
            log_audio.exception("mic start failed")
            # A driver that refuses a second reader on the same device is the
            # likeliest cause, so drop the monitor entirely and try once more.
            if self.passthrough.running:
                log_audio.info("retrying mic start without the passthrough monitor")
                self.passthrough.stop()
                try:
                    self.recorder.start()
                except Exception:
                    log_audio.exception("mic start failed again")
                    self.passthrough.set_muted(False)
                    self.ui_state.emit("error")
                    return
            else:
                self.passthrough.set_muted(False)
                self.ui_state.emit("error")
                return
        self._duck_if_enabled()
        self.ui_state.emit("recording")
        # Clear any badge from the previous dictation and make no claim yet.
        # The badge is confirmatory only: it appears after the transcribe
        # returns and tells us local Whisper is what actually ran, never as
        # a prediction that could be wrong if a key turns out to work.
        self.ui_engine.emit("pending")
        self._level_timer.start()
        self._preload_local_model()

    def _preload_local_model(self) -> None:
        """Auto mode: if the local model isn't in memory and the GPU is free,
        start loading it now, while the user is still talking.

        The model is absent after a VRAM release or a deferred warm-up, and
        the old code only noticed at key release — so that clip waited out a
        ~1.7 s load (2.06 s median to paste vs 0.35 s) with the pill stuck on
        "transcribing". engine.load() is idempotent and serialized on its own
        build lock, so the decode simply joins a load already in flight."""
        if self.settings.stt_backend != "auto" or self.engine.model_resident:
            return
        t = getattr(self, "_preload_thread", None)
        if t is not None and t.is_alive():
            return
        try:
            if GPU_MONITOR.busy_reason(need_vram=True) is not None:
                return      # this clip is headed for the cloud anyway
        except Exception:
            return

        def _load() -> None:
            try:
                self.engine.load()
            except Exception:
                log_engine.exception("local model preload failed")

        log_engine.debug("auto backend: preloading the local model during recording")
        self._preload_thread = threading.Thread(
            target=_load, daemon=True, name="model-preload",
        )
        self._preload_thread.start()

    def on_release(self) -> None:
        self._level_timer.stop()
        # Unmute first so the room is audible again the instant the key is
        # released, rather than after the transcription round-trip.
        self.passthrough.set_muted(False)
        if self.settings.passthrough_enabled and not self.passthrough.running:
            # The monitor was torn down to free the mic for this recording.
            self.passthrough.start()
        # Restore other apps' volume immediately on release — doing it here
        # (rather than after transcription) keeps the perceived ducking
        # window tight to the actual recording window.
        self.ducker.restore()
        try:
            audio = self.recorder.stop()
        except Exception:
            log_audio.exception("mic stop failed")
            self.ui_state.emit("error")
            return
        save_debug_recording(audio, self.settings.keep_recordings)
        if self._cancel_current_recording:
            log_hotkey.info("release: recording cancelled — skipping transcribe")
            self._cancel_current_recording = False
            self.ui_state.emit("idle")
            return
        if audio.size == 0:
            log_hotkey.debug("release (no audio)")
            self.ui_state.emit("idle")
            return
        gen = self._bump_gen()
        log_hotkey.debug("release → transcribe (gen=%d, %.2fs)",
                         gen, audio.size / SAMPLE_RATE)
        self.ui_state.emit("transcribing")
        threading.Thread(
            target=self._process_audio, args=(audio, gen, self._current_context),
            daemon=True, name=f"transcribe-{gen}",
        ).start()

    def on_cancel(self) -> None:
        """Esc pressed — abandon the current recording without transcribing.
        Only acts when we're actually recording (dictation OR a note);
        otherwise Esc behaves normally for whatever app the user is in."""
        if self._note_recording:
            log_hotkey.info("Esc: cancel current note recording")
            self._note_cancel = True
            self._end_note_recording()
            return
        if not self.hotkey.recording:
            return
        log_hotkey.info("Esc: cancel current recording")
        self._cancel_current_recording = True
        self.hotkey.force_release()

    def on_repaste_last(self) -> None:
        """Ctrl+Win+V: take the last transcript and paste it into whatever
        window has focus right now. Handy when an auto-paste landed in the
        wrong window or you want to use the same dictation again."""
        last = self.log.last()
        if not last:
            log_app.info("repaste-last: no transcript yet")
            return
        text = (last.get("text") or "").strip()
        if not text:
            return
        log_app.info("repaste-last (%d chars)", len(text))
        self._cancel_dictation_for_chord("repaste")
        # Off the GUI thread: paste() waits for the Ctrl+Win(+B) release
        # before it types, and must not stall the event loop doing so.
        threading.Thread(
            target=self.injector.paste, args=(text,),
            kwargs={"press_enter": self.settings.send_after_paste},
            daemon=True, name="repaste",
        ).start()

    def _cancel_dictation_for_chord(self, what: str) -> None:
        """Ctrl+Win+<key> chords are pressed on top of the push-to-talk
        modifiers, so the PTT poll has already started a dictation by the
        time the chord fires. Abandon it — the user wanted the chord, not a
        0.3 s recording of them reaching for the key (which then decoded to
        junk and pasted on top of the chord's own paste)."""
        if self.hotkey.recording:
            log_hotkey.info("%s: cancelling in-flight dictation", what)
            self._cancel_current_recording = True
            self.hotkey.force_release()

    def on_llm_apply_last(self) -> None:
        """Ctrl+Win+L: run the configured default LLM prompt on the most
        recent transcript and paste the result. Does NOT replace the
        history entry — appends a new entry tagged with the prompt id so
        the original raw dictation stays available for reference."""
        if not self.settings.llm_enabled:
            log_app.info("llm-apply: LLM post-processing is disabled in Settings")
            self.ui_state.emit("error")
            self._schedule_reset(self._bump_gen())
            return
        last = self.log.last()
        if not last:
            log_app.info("llm-apply: no transcript yet")
            return
        text = (last.get("text") or "").strip()
        if not text:
            return
        prompt = llm_prompt_by_id(self.settings.llm_default_prompt) \
                 or BUILTIN_LLM_PROMPTS[0]
        log_app.info("llm-apply: prompt=%s on %d chars", prompt["id"], len(text))
        self._cancel_dictation_for_chord("llm-apply")
        self.ui_state.emit("transcribing")  # reuse the blue pulse for "thinking"
        threading.Thread(
            target=self._llm_apply_worker,
            args=(text, prompt, True),  # paste=True, add_to_history=True
            daemon=True,
            name="llm-apply",
        ).start()

    def on_note_toggle(self) -> None:
        """Ctrl+Win+N: toggle a note recording. First press starts; second
        press stops, transcribes, and saves to the Notes tab (no paste).
        Auto-flags the note with the foreground app's basename so notes can
        later be searched/grouped by where you were when you captured them.
        """
        if self._note_recording:
            self._end_note_recording()
            return
        # If a dictation is currently active, drop it cleanly so the audio
        # stream is free and we don't end up with a phantom partial paste.
        if self.hotkey.recording:
            log_hotkey.info("note toggle: cancelling in-flight dictation")
            self._cancel_current_recording = True
            self.hotkey.force_release()
        self._begin_note_recording()

    def _begin_note_recording(self) -> None:
        gen = self._bump_gen()
        self._note_cancel = False
        self._current_context = get_foreground_process_basename()
        log_hotkey.info("note begin (gen=%d, context=%s)", gen, self._current_context)
        if self.settings.passthrough_mute_while_dictating:
            self.passthrough.set_muted(True)
        try:
            self.recorder.start()
        except Exception:
            log_audio.exception("note: mic start failed")
            if self.passthrough.running:
                log_audio.info("retrying note mic start without the passthrough monitor")
                self.passthrough.stop()
                try:
                    self.recorder.start()
                except Exception:
                    log_audio.exception("note: mic start failed again")
                    self.passthrough.set_muted(False)
                    self.ui_state.emit("error")
                    return
            else:
                self.passthrough.set_muted(False)
                self.ui_state.emit("error")
                return
        self._note_recording = True
        self.hotkey.set_note_in_progress(True)
        self._duck_if_enabled()
        self.ui_state.emit("recording")
        # Clear any badge from the previous dictation and make no claim yet.
        # The badge is confirmatory only: it appears after the transcribe
        # returns and tells us local Whisper is what actually ran, never as
        # a prediction that could be wrong if a key turns out to work.
        self.ui_engine.emit("pending")
        self._level_timer.start()

    def _end_note_recording(self) -> None:
        self._level_timer.stop()
        self.ducker.restore()
        self.passthrough.set_muted(False)
        if self.settings.passthrough_enabled and not self.passthrough.running:
            self.passthrough.start()
        try:
            audio = self.recorder.stop()
        except Exception:
            log_audio.exception("note: mic stop failed")
            self.ui_state.emit("error")
            self._note_recording = False
            self.hotkey.set_note_in_progress(False)
            return
        save_debug_recording(audio, self.settings.keep_recordings)
        self._note_recording = False
        self.hotkey.set_note_in_progress(False)
        if self._note_cancel:
            log_hotkey.info("note: cancelled — skipping transcribe")
            self._note_cancel = False
            self.ui_state.emit("idle")
            return
        if audio.size == 0:
            log_hotkey.debug("note: no audio")
            self.ui_state.emit("idle")
            return
        gen = self._bump_gen()
        log_hotkey.debug("note → transcribe (gen=%d, %.2fs)",
                         gen, audio.size / SAMPLE_RATE)
        self.ui_state.emit("transcribing")
        threading.Thread(
            target=self._process_note_audio, args=(audio, gen, self._current_context),
            daemon=True, name=f"note-transcribe-{gen}",
        ).start()

    def _process_note_audio(self, audio: np.ndarray, gen: int, app: str | None) -> None:
        if not self._is_latest(gen):
            log_app.debug("note transcribe(gen=%d) stale before start — skip", gen)
            return
        prompt = build_initial_prompt(self.log)
        hotwords = ", ".join(self.replacements.vocab) or None
        with self._engine_lock:
            try:
                raw = self.engine.transcribe(audio, initial_prompt=prompt, hotwords=hotwords)
                # engine.last_backend is authoritative: the pool may have
                # fallen back to local Whisper after we predicted cloud.
                self.ui_engine.emit(self.engine.last_backend)
            except Exception:
                log_engine.exception("note transcribe failed (gen=%d)", gen)
                if self._is_latest(gen):
                    self.ui_state.emit("error")
                    self._schedule_reset(gen)
                return
        raw = strip_audio_event_tags(raw)
        if not raw:
            # Empty transcribe — flash orange so a too-short / silent press is
            # visually distinct from a successful paste landing off-screen.
            if self._is_latest(gen):
                self.ui_state.emit("empty")
                self._schedule_reset(gen)
            return
        text = self._polish(raw, app)
        # Notes are NEVER pasted (that's the whole point — silent capture).
        # No LLM auto-apply either; notes go in raw-clean so the user can
        # decide later what to do with them.
        self.notes.add(text, app=app, raw=raw)
        log_app.info("note saved: %d chars (app=%s)", len(text), app or "?")
        if self._is_latest(gen):
            self.ui_state.emit("done")
            self._schedule_reset(gen)

    def _llm_apply_worker(self, text: str, prompt: dict, paste: bool) -> None:
        gen = self._bump_gen()
        client = LLMClient(
            base_url=self.settings.llm_base_url,
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key,
        )
        try:
            result = client.chat(prompt["system"], text)
        except Exception as e:
            log_app.exception("LLM call failed: %s", e)
            if self._is_latest(gen):
                self.ui_state.emit("error")
                self._schedule_reset(gen)
            return
        result = (result or "").strip()
        if not result:
            log_app.warning("LLM returned empty result — leaving original transcript alone")
            if self._is_latest(gen):
                self.ui_state.emit("idle")
            return
        log_app.info("LLM apply: %d -> %d chars (prompt=%s)",
                     len(text), len(result), prompt["id"])
        if paste:
            self.injector.paste(result, press_enter=self.settings.send_after_paste)
        # Append as a new history entry tagged with the source prompt so the
        # user can see what was done and retrieve the LLM-cleaned version
        # later from the History tab.
        if self.settings.save_history:
            self.log.add(result, raw=text)
        self._dispatch_integrations(result, raw=text)
        if self._is_latest(gen):
            self.ui_state.emit("done")
            self._schedule_reset(gen)

    def _dispatch_integrations(self, text: str, *, raw: str = "") -> None:
        """Best-effort fan-out to the v1.8 sinks (markdown daily file, webhook).
        Sinks run on a worker thread so a slow webhook never blocks the
        transcript pipeline; markdown writes are fast enough to inline but
        we batch them in the same thread for consistency."""
        if not text.strip():
            return
        if not (self.settings.markdown_save_enabled or self.settings.webhook_enabled):
            return
        ts = time.time()
        # Snapshot settings up-front so a concurrent save() doesn't cause
        # the worker to see a partial update.
        md_on = self.settings.markdown_save_enabled
        md_folder = self.settings.markdown_folder or str(DEFAULT_MARKDOWN_DIR)
        wh_on = self.settings.webhook_enabled
        wh_url = self.settings.webhook_url
        wh_template = self.settings.webhook_template

        def worker():
            if md_on:
                try:
                    MarkdownDailySink(Path(md_folder)).write(text, ts=ts, raw=raw)
                except Exception:
                    log_app.exception("markdown sink dispatch failed")
            if wh_on and wh_url:
                try:
                    WebhookSink(wh_url, wh_template).post(text, ts=ts, raw=raw)
                except Exception:
                    log_app.exception("webhook sink dispatch failed")

        threading.Thread(target=worker, daemon=True, name="integrations").start()

    def _tick_level(self) -> None:
        self.ui_level.emit(self.recorder.level)

    def _polish(self, raw: str, context: str | None) -> str:
        """Dictionary → voice commands → smart defaults. Never raises: a bad
        dictionary entry or a regex edge case must not cost the user the
        transcript, so any failure logs and returns what we had so far."""
        text = raw
        try:
            text = self.replacements.apply(raw, context=context)
            if self.settings.voice_commands:
                before_vc = text
                text = self.voice_commands.process(text)
                if text != before_vc:
                    log_app.info("voice commands: %r -> %r", before_vc, text)
            # Always-on smart polish: capitalize first letter, ensure
            # terminal punctuation, strip orphan trailing commas.
            text = apply_smart_defaults(text)
        except Exception:
            log_app.exception("post-processing failed — pasting unpolished text")
        if text != raw:
            log_app.info("rewrite %r -> %r", raw, text)
        return text

    def _process_audio(self, audio: np.ndarray, gen: int,
                       context: str | None = None) -> None:
        # Bail early if this audio is already stale (user pressed again).
        if not self._is_latest(gen):
            log_app.debug("transcribe(gen=%d) stale before start — skip", gen)
            return
        # Per-app dictionary context is captured at release time and handed
        # in, because self._current_context already belongs to the NEXT
        # recording if the user pressed again before this decode finished.
        if context is None:
            context = self._current_context
        prompt = build_initial_prompt(self.log)
        hotwords = ", ".join(self.replacements.vocab) or None
        with self._engine_lock:
            try:
                raw = self.engine.transcribe(audio, initial_prompt=prompt, hotwords=hotwords)
                # engine.last_backend is authoritative: the pool may have
                # fallen back to local Whisper after we predicted cloud.
                self.ui_engine.emit(self.engine.last_backend)
            except Exception:
                log_engine.exception("transcribe failed (gen=%d)", gen)
                if self._is_latest(gen):
                    self.ui_state.emit("error")
                    self._schedule_reset(gen)
                return

        raw = strip_audio_event_tags(raw)
        if not raw:
            # Empty transcribe — flash orange so a too-short / silent press is
            # visually distinct from a successful paste landing off-screen.
            if self._is_latest(gen):
                self.ui_state.emit("empty")
                self._schedule_reset(gen)
            return

        text = self._polish(raw, context)

        # Auto-apply LLM post-processing: replaces the transcript before paste
        # and replaces the history entry's text (raw still captures Whisper's
        # original output for forensic purposes).
        if (self.settings.llm_enabled and self.settings.llm_auto_apply
                and self._is_latest(gen)):
            prompt = llm_prompt_by_id(self.settings.llm_default_prompt) \
                     or BUILTIN_LLM_PROMPTS[0]
            client = LLMClient(
                base_url=self.settings.llm_base_url,
                model=self.settings.llm_model,
                api_key=self.settings.llm_api_key,
            )
            try:
                llm_result = client.chat(prompt["system"], text)
                llm_result = (llm_result or "").strip()
                if llm_result:
                    log_app.info("auto-LLM: %r -> %r (prompt=%s)",
                                 text, llm_result, prompt["id"])
                    text = llm_result
            except Exception:
                log_app.exception("auto-LLM failed — falling back to raw transcript")

        if self.settings.auto_paste:
            self.injector.paste(text, press_enter=self.settings.send_after_paste)
        if self.settings.save_history:
            self.log.add(text, raw=raw)
        self._dispatch_integrations(text, raw=raw)

        if self._is_latest(gen):
            self.ui_state.emit("done")
            self._schedule_reset(gen)

    def _schedule_reset(self, gen: int) -> None:
        def reset():
            time.sleep(RESET_AFTER_DONE_SEC)
            if self._is_latest(gen):
                self.ui_state.emit("idle")
            else:
                log_app.debug("reset(gen=%d) stale — skip", gen)
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


_SINGLE_INSTANCE_HANDLE = None


def _acquire_single_instance() -> bool:
    """Hold a named mutex for the life of the process; False if another
    process already holds it. Non-Windows: always True."""
    global _SINGLE_INSTANCE_HANDLE
    if not _WIN32:
        return True
    try:
        import ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p)
        k32.CreateMutexW.restype = ctypes.c_void_p
        name = f"Local\\{APP_NAME}-single-instance"
        # run.bat terminates the previous instance and launches this one in
        # the same breath; TerminateProcess is asynchronous, so the old
        # mutex can still be alive for a moment. Retry briefly before
        # concluding someone else really owns it.
        for _attempt in range(8):
            handle = k32.CreateMutexW(None, False, name)
            err = ctypes.get_last_error()
            if not handle:
                return True      # couldn't create one at all — don't block startup
            if err != 183:       # 183 = ERROR_ALREADY_EXISTS
                _SINGLE_INSTANCE_HANDLE = handle   # keep it alive; released at exit
                return True
            k32.CloseHandle(handle)
            time.sleep(0.5)
        return False
    except Exception:
        return True


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
    # Single instance. Two Tellurs means two pastes per dictation, two
    # duckers fighting over volumes and two writers on settings/history.
    # run.bat kills older instances before launching, but a second launch
    # through any other path (double-clicking tellur.py, an overlapping
    # updater restart) had nothing stopping it.
    if not _acquire_single_instance():
        startup.warning("another %s instance is already running — exiting", APP_NAME)
        listener.stop()
        return 0
    # Elevation is load-bearing: unelevated, UIPI blocks both the hotkey hook
    # and paste whenever an admin window (elevated terminal) has focus. Log
    # it so "dictation dead in terminal X" is diagnosable from the log alone.
    if sys.platform == "win32":
        try:
            import ctypes
            _elevated = bool(ctypes.windll.shell32.IsUserAnAdmin())
        except Exception:
            _elevated = False
        startup.info(
            "running elevated: %s",
            "yes" if _elevated
            else "NO — hotkey and paste won't reach admin windows (UIPI)",
        )

    os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false")

    qapp = QApplication(sys.argv)
    qapp.setQuitOnLastWindowClosed(False)
    qapp.setApplicationName(APP_NAME)
    qapp.setApplicationDisplayName(APP_NAME)
    qapp.setApplicationVersion(__version__)
    apply_app_branding(qapp)

    try:
        app = App()
        app.start()
    except Exception as e:
        startup.exception("fatal startup error")
        show_fatal(f"{APP_NAME} failed to start:\n\n{e}\n\nSee log for details.")
        listener.stop()
        return 1

    # After App() — the keyboard listener thread exists by now, so both the
    # process class and the hook pump thread get bumped.
    raise_input_priority()

    exit_code = qapp.exec()
    startup.info("shutting down (exit=%d)", exit_code)
    listener.stop()
    return exit_code


if __name__ == "__main__":
    sys.exit(main())

