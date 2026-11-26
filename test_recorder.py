"""Offline checks for AudioRecorder's always-open mode (v2.7.5).

PortAudio is replaced with a fake whose streams are driven by hand, so these
run without a microphone: pre-roll seeding, the tap gate, held -> per-press
fallback without doubled audio, stall/default-mic-change recovery, and the
Settings round trip.

    python test_recorder.py
"""
import os
import sys
import tempfile
import time
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="tellur-recorder-test-")
os.environ["TELLUR_HOME"] = _TMP          # must be set before importing tellur
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import tellur as t  # noqa: E402

assert str(t.SETTINGS_FILE).startswith(_TMP), t.SETTINGS_FILE
PASS = FAIL = 0
SR = t.SAMPLE_RATE
BLOCK = 160   # 10 ms


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS", name, extra)
    else:
        FAIL += 1
        print("FAIL", name, extra)


class FakeStream:
    opened: list = []
    fail_open = False

    def __init__(self, samplerate, channels, dtype, callback, blocksize, device):
        if FakeStream.fail_open:
            raise RuntimeError("fake: device unavailable")
        self.callback = callback
        self.device = device
        self.active = False
        self.closed = False
        FakeStream.opened.append(self)

    def start(self):
        self.active = True

    def stop(self):
        self.active = False

    def close(self):
        self.closed = True
        self.active = False

    def feed(self, value: float, blocks: int = 1):
        for _ in range(blocks):
            self.callback(np.full((BLOCK, 1), value, dtype=np.float32), BLOCK, None, None)


class FakeSD:
    InputStream = FakeStream
    inits = 0

    @staticmethod
    def _terminate():
        pass

    @staticmethod
    def _initialize():
        FakeSD.inits += 1

    @staticmethod
    def query_devices():
        return [{"name": "Mic A", "max_input_channels": 1},
                {"name": "Mic B", "max_input_channels": 1}]

    class default:
        device = (0, 0)


t.sd = FakeSD
DEFAULT_ID = ["endpoint-A"]
t.default_capture_endpoint_id = lambda: DEFAULT_ID[0]
t.MIC_WATCHDOG_INTERVAL_SEC = 0.05
t.MIC_STALL_SECONDS = 0.3
t.MIC_REINIT_MIN_INTERVAL_SEC = 0.0


def live(rec):
    return [s for s in FakeStream.opened if s.active and s.callback == rec._held_callback]


def wait_for(cond, timeout=3.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


# --- per-press mode is unchanged ---------------------------------------------
rec = t.AudioRecorder()
rec.start()
cold = FakeStream.opened[-1]
cold.feed(0.1, 50)   # 0.5 s
audio = rec.stop()
check("per-press: records what the stream delivers", audio.size == 50 * BLOCK, audio.size)
check("per-press: stream closed on stop", cold.closed)

# --- always-open: held stream, pre-roll, instant start ------------------------
rec = t.AudioRecorder()
rec.set_always_open(True)
check("held stream opened by the holder thread", wait_for(lambda: len(live(rec)) == 1))
held = live(rec)[0]
held.feed(0.2, 100)  # 1 s of idle audio; only ~0.3 s should be kept
n_open = len(FakeStream.opened)
rec.start()
check("start() opens no new stream when held", len(FakeStream.opened) == n_open)
held.feed(0.5, 100)  # 1 s spoken
audio = rec.stop()
pre = int(t.MIC_PREROLL_SECONDS * SR)
check("recording = pre-roll + live audio", audio.size == pre + 100 * BLOCK, audio.size)
check("pre-roll comes first and is idle audio",
      np.allclose(audio[:pre], 0.2) and np.allclose(audio[pre:], 0.5))
check("held stream survives stop()", held.active and not held.closed)
check("monitor level comes from the held stream", rec.monitor_level > 0)

# --- a tap must not be padded past the too-short gate --------------------------
held.feed(0.2, 100)
rec.start()
held.feed(0.5, 3)    # 30 ms tap
audio = rec.stop()
check("tap drops the pre-roll",
      audio.size == 3 * BLOCK and audio.size < int(t.MIN_AUDIO_SECONDS * SR), audio.size)

# --- stall -> reopen with PortAudio re-init ------------------------------------
inits = FakeSD.inits
time.sleep(t.MIC_STALL_SECONDS + 2.2)   # past the open grace, no callbacks
check("stalled held stream is replaced",
      wait_for(lambda: held.closed and len(live(rec)) == 1))
check("stall triggers a PortAudio re-init", FakeSD.inits > inits)
held = live(rec)[0]

# --- held unhealthy at press -> per-press fallback, no doubled audio ----------
rec._last_held_cb = time.monotonic() - 10   # looks stalled right now
rec.start()
cold = FakeStream.opened[-1]
check("unhealthy held -> per-press stream", cold.callback == rec._callback)
held.feed(0.9, 20)   # held stream "recovers" mid-recording
cold.feed(0.4, 40)
audio = rec.stop()
check("no audio from the held stream leaks into a cold recording",
      audio.size == 40 * BLOCK and np.allclose(audio, 0.4), audio.size)
rec._last_held_cb = time.monotonic()

# --- Windows default mic changes -> re-init and reopen ------------------------
inits = FakeSD.inits
DEFAULT_ID[0] = "endpoint-B"
check("default mic change reopens the held stream",
      wait_for(lambda: held.closed and len(live(rec)) == 1))
check("default mic change re-initialises PortAudio", FakeSD.inits > inits)
held = live(rec)[0]

# --- re-init is skipped while passthrough owns device indexes ----------------
rec.reinit_allowed = lambda: False
inits = FakeSD.inits
DEFAULT_ID[0] = "endpoint-C"
wait_for(lambda: held.closed, timeout=1.0)
check("no re-init while passthrough is enabled", FakeSD.inits == inits)
rec.reinit_allowed = lambda: True
wait_for(lambda: len(live(rec)) == 1)
held = live(rec)[0]

# --- device change follows the configured mic --------------------------------
rec.set_device(1, name="Mic B")
check("set_device moves the held stream",
      wait_for(lambda: held.closed and len(live(rec)) == 1 and live(rec)[0].device == 1))
held = live(rec)[0]

# --- open failure: falls back per press, retries in the background ------------
FakeStream.fail_open = True
held.active = False   # device vanished
time.sleep(0.3)
check("held stream gone while the device is unavailable", len(live(rec)) == 0)
FakeStream.fail_open = False
check("held stream comes back once the device does", wait_for(lambda: len(live(rec)) == 1))
held = live(rec)[0]

# --- turning it off waits for an in-flight dictation --------------------------
held.feed(0.2, 40)
rec.start()
rec.set_always_open(False)
time.sleep(0.2)
check("switching off mid-dictation keeps the stream until release", not held.closed)
held.feed(0.5, 30)
audio = rec.stop()
check("that dictation still has its audio", audio.size >= 30 * BLOCK, audio.size)
check("held stream closed after the release", wait_for(lambda: held.closed))

# --- close() on quit ----------------------------------------------------------
rec.set_always_open(True)
wait_for(lambda: len(live(rec)) == 1)
held = live(rec)[0]
rec.close()
check("close() releases the held stream", held.closed and not rec.always_open)

# --- Settings round trip ------------------------------------------------------
sp = Path(_TMP) / "settings.json"
s = t.Settings(sp)
check("mic_always_open defaults off", s.mic_always_open is False)
s.mic_always_open = True
s.save_quiet() if hasattr(s, "save_quiet") else s.save()
check("mic_always_open persists", t.Settings(sp).mic_always_open is True)

# --- smart defaults: standalone pronoun "i" (v2.7.5) -------------------------
for raw, want in {
    "i want you to look, i think": "I want you to look, I think.",
    "so i'm done and i'll go": "So I'm done and I'll go.",
    "use i.e. this": "Use i.e. this.",
    "my iPad is it": "My iPad is it.",
    "the wifi is fine": "The wifi is fine.",
    "is it i?": "Is it I?",
    "pi is 3": "Pi is 3.",
}.items():
    got = t.apply_smart_defaults(raw)
    check(f"pronoun i: {raw!r}", got == want, repr(got))

print(f"\n{PASS} passed, {FAIL} failed")
sys.exit(1 if FAIL else 0)
