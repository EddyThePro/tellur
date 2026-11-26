"""Offline checks for the cloud/local transcription router.

Covers the ElevenLabs key pool (failover, hedged requests, per-clip deadline),
the auto-mode VRAM release rule, the press-time model preload, the Scribe
request fields and the settings migration. No network, no GPU decode and no
real API keys: the HTTP client and the Whisper model are replaced with fakes,
and settings live in a throwaway directory.

    python test_cloud_router.py
"""
import json
import os
import sys
import tempfile
import threading
import time
import types
import urllib.error
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="tellur-router-test-")
os.environ["TELLUR_HOME"] = _TMP          # must be set before importing tellur
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import tellur as t  # noqa: E402

assert str(t.SETTINGS_FILE).startswith(_TMP), t.SETTINGS_FILE
PASS = FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print("PASS", name, extra)
    else:
        FAIL += 1
        print("FAIL", name, extra)


SETTINGS_PATH = Path(_TMP) / "settings.json"


def fresh_settings():
    if SETTINGS_PATH.exists():
        SETTINGS_PATH.unlink()
    s = t.Settings(SETTINGS_PATH)
    s.stt_backend = "elevenlabs"
    s.eleven_api_key = "key-one-AAAA"
    s.eleven_api_key_backup = "key-two-BBBB"
    s.eleven_api_keys_extra = []
    s.eleven_exhausted = []
    s.eleven_exhausted_at = {}
    s.eleven_retry_after = {}
    s.eleven_credits_left = {}
    s.eleven_credits_at = {}
    return s


audio = (np.random.rand(16000 * 4).astype(np.float32) - 0.5) * 0.2   # 4 s
calls = []
behaviour = {}


def fake_transcribe_wav(self, wav):
    n = len(calls)
    calls.append((self.api_key, time.monotonic(), self.timeout))
    return behaviour["fn"](self, n)


real_transcribe_wav = t.ElevenLabsScribeClient.transcribe_wav
t.ElevenLabsScribeClient.transcribe_wav = fake_transcribe_wav

# --- time budget -----------------------------------------------------------
h, d = t.eleven_time_budget(10)
check("budget 10s clip", abs(h - 2.9) < 1e-6 and abs(d - 6.8) < 1e-6, (h, d))
h, d = t.eleven_time_budget(300)
check("budget 300s clip", abs(h - 14.5) < 1e-6 and abs(d - 30) < 1e-6, (h, d))

# --- fast path -------------------------------------------------------------
s = fresh_settings()
pool = t.ElevenLabsScribePool(s)
behaviour["fn"] = lambda self, n: "hello"
calls.clear()
t0 = time.monotonic()
r = pool.transcribe(audio)
check("fast path result", r == ("hello", "key-one-AAAA", 0), r)
check("fast path single request, no hedge",
      len(calls) == 1 and time.monotonic() - t0 < 0.5)
check("socket timeout = clip deadline",
      abs(calls[0][2] - t.eleven_time_budget(4)[1]) < 1e-6, calls[0][2])

# --- first request stalls, the hedge wins ----------------------------------
gate = threading.Event()


def stall_first(self, n):
    if n == 0:
        gate.wait(20)
        return "late"
    time.sleep(0.1)
    return "from-hedge"


behaviour["fn"] = stall_first
pool = t.ElevenLabsScribePool(s, timeout=1.0)     # hedge at 0.5, deadline 1.0
calls.clear()
t0 = time.monotonic()
r = pool.transcribe(audio)
el = time.monotonic() - t0
check("hedge wins", r[0] == "from-hedge", r)
check("hedge timing ~0.6s", 0.5 <= el < 0.9, round(el, 2))
check("two requests sent", len(calls) == 2)
gate.set()
time.sleep(0.2)                                   # late answer is discarded
check("late answer left key state alone", s.eleven_exhausted == [])

# --- both stall -> ElevenTimeout at the deadline ---------------------------
gate = threading.Event()
behaviour["fn"] = lambda self, n: (gate.wait(20), "late")[1]
calls.clear()
t0 = time.monotonic()
try:
    pool.transcribe(audio)
    ok = False
except t.ElevenTimeout:
    ok = True
el = time.monotonic() - t0
check("both stall -> ElevenTimeout", ok and 0.95 <= el < 1.4, round(el, 2))
check("ElevenTimeout is an ElevenAllKeysFailed",
      issubclass(t.ElevenTimeout, t.ElevenAllKeysFailed))
gate.set()
time.sleep(0.2)

# --- quota error on key 1 -> key 2 serves, key 1 benched -------------------
s = fresh_settings()
pool = t.ElevenLabsScribePool(s)


def quota_then_ok(self, n):
    if self.api_key.endswith("AAAA"):
        raise t.ElevenAuthError(
            401, "quota_exceeded",
            "This request exceeds your quota of 10000. "
            "You have 0 credits remaining")
    return "second key"


behaviour["fn"] = quota_then_ok
calls.clear()
r = pool.transcribe(audio)
check("failover to key 2", r == ("second key", "key-two-BBBB", 1), r)
check("key 1 marked exhausted", "key-one-AAAA" in s.eleven_exhausted)
calls.clear()
pool.transcribe(audio)
check("benched key skipped next time", [c[0] for c in calls] == ["key-two-BBBB"])

# --- network down -> fails fast, no other key tried ------------------------
s = fresh_settings()
pool = t.ElevenLabsScribePool(s)


def neterr(self, n):
    raise urllib.error.URLError("getaddrinfo failed")


behaviour["fn"] = neterr
calls.clear()
t0 = time.monotonic()
try:
    pool.transcribe(audio)
    kind = None
except t.ElevenTimeout:
    kind = "timeout"
except t.ElevenAllKeysFailed:
    kind = "allfailed"
check("network error fails fast",
      kind == "allfailed" and time.monotonic() - t0 < 0.5 and len(calls) == 1,
      (kind, len(calls)))


# --- first fails AFTER the hedge started; the hedge still answers ----------
def first_dies_late(self, n):
    if n == 0:
        time.sleep(0.6)
        raise urllib.error.URLError("connection reset")
    time.sleep(0.25)
    return "hedge-saved-it"


behaviour["fn"] = first_dies_late
pool = t.ElevenLabsScribePool(s, timeout=1.0)
calls.clear()
r = pool.transcribe(audio)
check("hedge survives the first request failing", r[0] == "hedge-saved-it", r)

# --- read timeout on one key -> next key in the same chain -----------------
s = fresh_settings()
pool = t.ElevenLabsScribePool(s)


def to_then_ok(self, n):
    if n == 0:
        raise TimeoutError("The read operation timed out")
    return "next key"


behaviour["fn"] = to_then_ok
calls.clear()
r = pool.transcribe(audio)
check("read timeout -> next key", r == ("next key", "key-two-BBBB", 1), r)

# --- no keys ---------------------------------------------------------------
s = fresh_settings()
s.eleven_api_key = ""
s.eleven_api_key_backup = ""
try:
    t.ElevenLabsScribePool(s).transcribe(audio)
    ok = False
except t.ElevenAllKeysFailed:
    ok = True
check("no keys -> ElevenAllKeysFailed", ok)


# --- engine: cloud timeout falls back to a local decode --------------------
class Seg:                      # faster-whisper segment stand-in
    def __init__(self, text):
        self.text = text


class FakeModel:
    def __init__(self):
        self.n = 0
        self.model = types.SimpleNamespace(unload_model=lambda: None)

    def transcribe(self, audio, **kw):
        self.n += 1
        return [Seg("local words here and there for you okay then yes sure thing")], None


class FakeMon:
    def __init__(self, busy, free):
        self.busy, self.free = busy, free

    def busy_reason(self, need_vram=True):
        return self.busy

    def free_vram_mb(self):
        return self.free


s = fresh_settings()
s.stt_backend = "auto"
eng = t.WhisperEngine("distil-large-v3.5", settings=s)
fm = FakeModel()
eng._model = fm
eng._eleven_pool = t.ElevenLabsScribePool(s, timeout=0.4)
gate = threading.Event()
behaviour["fn"] = lambda self, n: (gate.wait(20), "late")[1]
real_mon = t.GPU_MONITOR
t.GPU_MONITOR = FakeMon("GPU 80% busy", 24000.0)
eng._auto_busy_streak = 5
t0 = time.monotonic()
txt = eng.transcribe(audio)
el = time.monotonic() - t0
check("engine falls back to local after cloud timeout",
      txt.startswith("local words") and eng.last_backend == "local" and el < 1.0,
      (round(el, 2), eng.last_backend))
check("streak reset after timeout", eng._auto_busy_streak == 0)
gate.set()
time.sleep(0.1)

# --- auto mode: model release is gated on free VRAM ------------------------
behaviour["fn"] = lambda self, n: "cloud text"
eng._eleven_pool = t.ElevenLabsScribePool(s)
eng._auto_busy_streak = 0
for _ in range(5):
    eng.transcribe(audio)
check("model kept when VRAM is plentiful",
      eng.model_resident and eng.last_backend == "elevenlabs"
      and eng._auto_busy_streak == 5)
t.GPU_MONITOR = FakeMon("GPU 80% busy", 3000.0)
eng.transcribe(audio)
check("model released when VRAM is tight", not eng.model_resident)
eng._model = fm
eng._auto_busy_streak = 0
t.GPU_MONITOR = FakeMon("GPU 80% busy", None)
for _ in range(3):
    eng.transcribe(audio)
check("model released when NVML cannot report VRAM", not eng.model_resident)
t.GPU_MONITOR = real_mon

# --- preload on hotkey press -----------------------------------------------
loaded = []


class StubApp:
    _preload_local_model = t.App._preload_local_model


def make(backend, resident):
    a = StubApp()
    a.settings = types.SimpleNamespace(stt_backend=backend)
    a.engine = types.SimpleNamespace(
        model_resident=resident,
        load=lambda: (time.sleep(0.2), loaded.append(1)))
    return a


t.GPU_MONITOR = FakeMon(None, 24000.0)
a = make("auto", False)
a._preload_local_model()
a._preload_local_model()                 # second call while the first is in flight
a._preload_thread.join(2)
check("preload runs once while in flight", loaded == [1], loaded)
loaded.clear()
a = make("auto", True)
a._preload_local_model()
check("no preload when resident", not hasattr(a, "_preload_thread"))
a = make("elevenlabs", False)
a._preload_local_model()
check("no preload outside auto", not hasattr(a, "_preload_thread"))
t.GPU_MONITOR = FakeMon("GPU 90% busy", 24000.0)
a = make("auto", False)
a._preload_local_model()
check("no preload when GPU busy", not hasattr(a, "_preload_thread"))
t.GPU_MONITOR = real_mon

# --- mic probe (Windows + pycaw only) --------------------------------------
try:
    import comtypes
    comtypes.CoInitialize()
    ref = t.read_communications_mic_mute()
except Exception:
    ref = (None, "")
if ref[0] is None:
    print("SKIP mic probe (no readable communications microphone)")
else:
    probe = t.CommsMicProbe()
    first = probe.read()
    check("probe matches the one-shot read", first == ref, (first, ref))
    c0 = time.process_time()
    for _ in range(200):
        probe.read()
    cpu = (time.process_time() - c0) / 200 * 1000
    c0 = time.process_time()
    for _ in range(50):
        t.read_communications_mic_mute()
    old = (time.process_time() - c0) / 50 * 1000
    check("probe is much cheaper than one-shot", cpu < max(old / 5, 0.5),
          "probe %.3f ms vs one-shot %.2f ms cpu" % (cpu, old))
    probe._volume = object()             # broken interface -> unknown, then recovers
    r1 = probe.read()
    r2 = probe.read()
    check("probe recovers after a failure", r1[0] is None and r2 == ref, (r1, r2))
    probe = None

# --- start-on-login: only the task-level <Enabled> counts ------------------
xml_on = ("<Task><Triggers><LogonTrigger><Enabled>false</Enabled></LogonTrigger>"
          "</Triggers><Settings><Enabled>true</Enabled><IdleSettings/></Settings></Task>")
xml_off = ("<Task><Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger>"
           "</Triggers><Settings>\n <Enabled>false</Enabled>\n</Settings></Task>")
xml_none = ("<Task><Triggers><LogonTrigger><Enabled>true</Enabled></LogonTrigger>"
            "</Triggers><Settings><Hidden>false</Hidden></Settings></Task>")
if sys.platform == "win32":
    orig = t._tellur_task_xml
    t._tellur_task_xml = lambda: xml_on
    check("task enabled despite a disabled trigger", t.is_start_on_login_enabled() is True)
    t._tellur_task_xml = lambda: xml_off
    check("task disabled via Settings", t.is_start_on_login_enabled() is False)
    t._tellur_task_xml = lambda: xml_none
    check("no Enabled element -> enabled", t.is_start_on_login_enabled() is True)
    t._tellur_task_xml = orig

# --- recorder: dropped-buffer accounting -----------------------------------
rec = t.AudioRecorder()
rec._recording, rec._rec_source = True, "cold"   # as start() leaves it
rec._callback(np.zeros((160, 1), dtype=np.float32), 160, None, None)
rec._callback(np.zeros((160, 1), dtype=np.float32), 160, None, 2)
check("overflow counted", rec._overflows == 1 and rec._sample_count == 320)

# --- Scribe request fields + settings migration (2.7.4) --------------------
c = t.ElevenLabsScribeClient("k")
check("default cloud model is scribe_v3",
      c.model_id == "scribe_v3" and t.ELEVEN_DEFAULT_MODEL == "scribe_v3")
check("v3 clean fields",
      c._fields() == {"model_id": "scribe_v3", "tag_audio_events": "false",
                      "no_verbatim": "true", "numbers_format": "digits"},
      c._fields())
c = t.ElevenLabsScribeClient("k", clean=False)
check("v3 verbatim fields",
      c._fields() == {"model_id": "scribe_v3", "tag_audio_events": "false",
                      "numbers_format": "digits"})
c = t.ElevenLabsScribeClient("k", model_id="scribe_v2")
check("v2 gets no v3-only field",
      "numbers_format" not in c._fields() and c._fields().get("no_verbatim") == "true")

SETTINGS_PATH.write_text(
    json.dumps({"eleven_model_id": "scribe_v2", "eleven_api_key": "x"}),
    encoding="utf-8")
s2 = t.Settings(SETTINGS_PATH)
check("saved scribe_v2 migrates to v3",
      s2.eleven_model_id == "scribe_v3" and s2.eleven_clean_transcript is True)
s2.eleven_clean_transcript = False
s2.save_quiet()
d2 = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
check("migration flag + clean flag persisted",
      d2.get("eleven_model_migrated_v3") is True
      and d2.get("eleven_clean_transcript") is False
      and d2.get("eleven_model_id") == "scribe_v3")
d2["eleven_model_id"] = "scribe_v2"
SETTINGS_PATH.write_text(json.dumps(d2), encoding="utf-8")
s3 = t.Settings(SETTINGS_PATH)
check("hand-pinned v2 respected after migration",
      s3.eleven_model_id == "scribe_v2" and s3.eleven_clean_transcript is False)

seen = {}


def capture(self, wav):
    seen["clean"] = self.clean
    seen["model"] = self.model_id
    return "ok"


t.ElevenLabsScribeClient.transcribe_wav = capture
s4 = fresh_settings()
s4.eleven_clean_transcript = False
t.ElevenLabsScribePool(s4).transcribe(audio)
check("pool passes clean flag + model",
      seen == {"clean": False, "model": "scribe_v3"}, seen)
t.ElevenLabsScribeClient.transcribe_wav = real_transcribe_wav

print("\n%d passed, %d failed" % (PASS, FAIL))
sys.exit(1 if FAIL else 0)
