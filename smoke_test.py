"""Smoke test for Tellur v1.5-1.9 internals.

Exercises pure-Python pieces (sinks, exports, settings, transcript log,
LLM client) against a temp directory so the user's real D:\\Tellur data is
never touched. Skips the Qt UI and audio mic — those are best tested
manually by the user.

Run with the project venv:  D:\\Tellur\\.venv\\Scripts\\python.exe smoke_test.py
"""

from __future__ import annotations
import json
import os
import sys
import tempfile
import time
import traceback
import urllib.request
import urllib.error
from pathlib import Path

# Run Qt in offscreen mode so importing tellur doesn't try to spawn windows.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# We need a QApplication for QObject-based classes (Settings, TranscriptLog).
from PyQt6.QtWidgets import QApplication  # noqa: E402
_app = QApplication.instance() or QApplication(sys.argv)

import tellur as t  # noqa: E402


# -------------------------------------------------------------------------
# tiny test harness
# -------------------------------------------------------------------------
PASS = "[PASS]"
FAIL = "[FAIL]"
SKIP = "[SKIP]"

results: list[tuple[str, str, str]] = []  # (name, status, detail)


def _safe_print(s: str) -> None:
    """Console-encoding-safe print. Windows default cp1252 will raise on
    characters like '->' (U+2192); we replace those rather than crash."""
    try:
        print(s)
    except UnicodeEncodeError:
        enc = sys.stdout.encoding or "ascii"
        print(s.encode(enc, errors="replace").decode(enc, errors="replace"))


def check(name: str, fn):
    try:
        detail = fn() or ""
        results.append((name, PASS, str(detail)))
        _safe_print(f"{PASS} {name}  {detail}")
    except AssertionError as e:
        results.append((name, FAIL, f"assertion: {e}"))
        _safe_print(f"{FAIL} {name}  {e}")
    except Exception as e:
        results.append((name, FAIL, f"{type(e).__name__}: {e}"))
        traceback.print_exc()


def skip(name: str, reason: str) -> None:
    results.append((name, SKIP, reason))
    print(f"{SKIP} {name}  {reason}")


# -------------------------------------------------------------------------
# tests
# -------------------------------------------------------------------------
TMPDIR = Path(tempfile.mkdtemp(prefix="tellur-smoke-"))
print(f"\n--- using temp dir: {TMPDIR}\n")


def test_settings_roundtrip():
    """v1.5-1.9: all the new settings fields persist correctly."""
    path = TMPDIR / "settings.json"
    s = t.Settings(path)
    # v1.5
    s.hotkey_mode = "toggle"
    # v1.6
    s.input_device_name = "Test Mic XYZ"
    s.input_gain = 1.75
    # v1.7
    s.llm_enabled = True
    s.llm_base_url = "http://localhost:9999/v1"
    s.llm_model = "test-model"
    s.llm_api_key = "secret"
    s.llm_default_prompt = "bullets"
    s.llm_auto_apply = True
    # v1.8
    s.send_after_paste = True
    s.markdown_save_enabled = True
    s.markdown_folder = str(TMPDIR / "md")
    s.webhook_enabled = True
    s.webhook_url = "http://localhost:9999/hook"
    s.webhook_template = '{"foo": "{text}"}'
    # v1.9
    s.save_history = False
    s.history_retention_days = 30
    s.save()
    assert path.exists(), "settings.json wasn't written"

    s2 = t.Settings(path)
    assert s2.hotkey_mode == "toggle"
    assert s2.input_device_name == "Test Mic XYZ"
    assert abs(s2.input_gain - 1.75) < 1e-6
    assert s2.llm_enabled is True
    assert s2.llm_base_url == "http://localhost:9999/v1"
    assert s2.llm_model == "test-model"
    assert s2.llm_api_key == "secret"
    assert s2.llm_default_prompt == "bullets"
    assert s2.llm_auto_apply is True
    assert s2.send_after_paste is True
    assert s2.markdown_save_enabled is True
    assert s2.markdown_folder == str(TMPDIR / "md")
    assert s2.webhook_enabled is True
    assert s2.webhook_url == "http://localhost:9999/hook"
    assert s2.webhook_template == '{"foo": "{text}"}'
    assert s2.save_history is False
    assert s2.history_retention_days == 30
    return f"({len(json.loads(path.read_text())) } fields persisted)"


def test_settings_clamps_input_gain():
    path = TMPDIR / "settings-clamp.json"
    # Write a settings file with out-of-range gain manually.
    path.write_text(json.dumps({"input_gain": 50.0, "history_retention_days": -5}),
                    encoding="utf-8")
    s = t.Settings(path)
    assert s.input_gain == 10.0, f"expected clamp to 10.0, got {s.input_gain}"
    assert s.history_retention_days == 0, f"negative days should clamp to 0, got {s.history_retention_days}"
    return "gain clamped to [0.1, 10.0]; retention clamped to >= 0"


def test_transcript_log_purge():
    """v1.9: purge_older_than drops stale entries."""
    path = TMPDIR / "history-purge.json"
    log = t.TranscriptLog(path)
    now = time.time()
    # Synthetic entries: 0 days, 5 days, 20 days, 100 days old
    log._entries = [
        {"text": "fresh", "ts": now},
        {"text": "5 days", "ts": now - 5 * 86400},
        {"text": "20 days", "ts": now - 20 * 86400},
        {"text": "100 days", "ts": now - 100 * 86400},
    ]
    removed = log.purge_older_than(7 * 86400)  # 7 day cutoff
    assert removed == 2, f"expected 2 removed, got {removed}"
    remaining = [e["text"] for e in log.entries()]
    assert remaining == ["fresh", "5 days"], f"got {remaining}"
    # 0 / negative cutoff = no-op
    n2 = log.purge_older_than(0)
    assert n2 == 0
    return f"purged {removed}/4 entries; 0-cutoff is no-op"


def test_export_history_formats():
    """v1.9: export_history writes valid txt/md/json/csv."""
    entries = [
        {"ts": 1700000000, "text": "First transcript.", "raw": "first transcript"},
        {"ts": 1700001000, "text": "Second one, with comma."},
        {"ts": 1700002000, "text": "Third\nwith newline.", "edited": True},
    ]
    out_dir = TMPDIR / "exports"
    formats = ["txt", "md", "json", "csv"]
    counts = []
    for fmt in formats:
        p = out_dir / f"history.{fmt}"
        n = t.export_history(entries, p, fmt)
        assert p.exists() and p.stat().st_size > 0, f"{fmt} not written"
        counts.append((fmt, n, p.stat().st_size))
    # Sanity-check the JSON round-trip
    parsed = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
    assert parsed == entries, "json export didn't round-trip"
    # CSV should have 4 records (1 header + 3 entries). Use csv.reader so
    # quoted multi-line fields are counted as a single row (raw splitlines()
    # would over-count because one entry contains an embedded newline).
    import csv as csvmod
    with (out_dir / "history.csv").open(encoding="utf-8") as f:
        rows = list(csvmod.reader(f))
    assert len(rows) == 4, f"expected 4 csv records, got {len(rows)}"
    assert rows[0][0] == "timestamp_iso", f"unexpected header row: {rows[0]}"
    return "txt/md/json/csv all written; JSON round-trips; CSV record count OK"


def test_markdown_daily_sink():
    """v1.8: MarkdownDailySink appends one YYYY-MM-DD.md per local-time day."""
    folder = TMPDIR / "md-sink"
    sink = t.MarkdownDailySink(folder)
    now = time.time()
    sink.write("Hello world.", ts=now)
    sink.write("Second entry, same day.", ts=now + 60)
    # One file should exist for today
    files = list(folder.glob("*.md"))
    assert len(files) == 1, f"expected 1 file, got {len(files)}"
    body = files[0].read_text(encoding="utf-8")
    assert "Hello world." in body
    assert "Second entry, same day." in body
    assert body.startswith("# Tellur transcripts"), "missing H1 header"
    assert body.count("## ") == 2, "expected 2 timestamp H2 headings"
    return f"{files[0].name} contains both entries with timestamp headings"


def test_webhook_sink_json_vs_text():
    """v1.8: WebhookSink picks the right Content-Type based on template."""
    # We don't actually POST — we just verify the body builds correctly and
    # the json-detection logic flips. We mock urlopen.
    captured: dict = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["data"] = req.data.decode("utf-8")
        captured["headers"] = dict(req.headers.items())

        class _Resp:
            status = 200
            def __enter__(self_): return self_
            def __exit__(self_, *a): pass
            def read(self_): return b""
        return _Resp()

    real_urlopen = urllib.request.urlopen
    try:
        urllib.request.urlopen = fake_urlopen
        # JSON template path
        sink = t.WebhookSink("https://example.com/hook",
                             '{"text": "{text}", "ts": {ts}}')
        sink.post('Hello "world"', ts=1700000000.0, raw="hello world")
        assert "json" in captured["headers"]["Content-type"].lower(), \
            f"expected JSON content-type, got {captured['headers']}"
        parsed = json.loads(captured["data"])
        assert parsed["text"] == 'Hello "world"', f"got {parsed}"
        assert parsed["ts"] == 1700000000

        # Plain-text template path
        captured.clear()
        sink2 = t.WebhookSink("https://example.com/hook", "raw text body: {text}")
        sink2.post("hi", ts=0)
        assert "plain" in captured["headers"]["Content-type"].lower(), \
            f"expected plain content-type, got {captured['headers']}"
        assert captured["data"] == "raw text body: hi"
    finally:
        urllib.request.urlopen = real_urlopen
    return "JSON template -> application/json; plain template -> text/plain"


def test_smart_defaults_and_voice_commands():
    """v1.4 polish: capitalize, terminal punct. v1.4 voice cmds: comma/period/etc."""
    out = t.apply_smart_defaults("hello world")
    assert out == "Hello world.", f"got {out!r}"

    out = t.apply_smart_defaults("hello world,")
    assert out == "Hello world.", f"trailing weak comma should be replaced, got {out!r}"

    out = t.apply_smart_defaults("Already good!")
    assert out == "Already good!", f"got {out!r}"

    vc = t.VoiceCommandProcessor()
    out = vc.process("hello comma world period new line second sentence")
    assert "," in out and "." in out and "\n" in out, f"missing punctuation: {out!r}"
    return f"smart_defaults + voice cmds produce: {out!r}"


def test_llm_prompt_registry():
    """v1.7: built-in prompt registry has all 7 expected entries."""
    ids = {p["id"] for p in t.BUILTIN_LLM_PROMPTS}
    expected = {"cleanup", "paragraph", "formal", "bullets", "email", "slack", "summarize"}
    assert ids == expected, f"got {ids}, expected {expected}"
    # llm_prompt_by_id finds them all and returns None for unknown
    for pid in expected:
        assert t.llm_prompt_by_id(pid) is not None
    assert t.llm_prompt_by_id("nonexistent") is None
    return f"{len(ids)} prompts registered; lookup works"


def test_llm_client_against_lm_studio():
    """v1.7: live test against the user's local LM Studio.

    Per memory: LM Studio is running at http://localhost:1234/v1 with
    Qwen3-Coder-30B-A3B loaded. We do (1) a reachability ping, then (2)
    a single tiny chat completion with one of the built-in prompts.
    Skipped if the endpoint isn't reachable."""
    base = "http://localhost:1234/v1"
    client = t.LLMClient(base_url=base, model="", api_key="", timeout=30.0)

    if not client.ping():
        return None  # caller will mark SKIP

    # Use the "cleanup" prompt on a realistic noisy transcript.
    prompt = t.llm_prompt_by_id("cleanup")
    user = "um so like I was thinking maybe we should you know just go ahead and ship it"
    t0 = time.monotonic()
    result = client.chat(prompt["system"], user)
    dt = time.monotonic() - t0

    assert result, "empty response"
    assert len(result) >= 5, f"response too short: {result!r}"
    # Sanity: result should NOT contain "um" / "like" / "you know"
    lower = result.lower()
    fillers = ["um ", " um,", "you know", " like "]
    has_filler = any(f in lower for f in fillers)
    # We don't hard-assert this — some models keep some filler. Just report.
    return f"chat reply in {dt:.1f}s ({len(result)} chars){'; filler present' if has_filler else '; filler removed'}: {result[:120]!r}"


def test_hotkey_watcher_modes():
    """v1.5: HotkeyWatcher supports hold + toggle. We test set_mode
    transitions without actually starting the timer / hooks."""
    hw = t.HotkeyWatcher()
    # set_mode shouldn't emit if already in that mode
    fired = []
    hw.released.connect(lambda: fired.append("released"))
    hw.set_mode("hold")  # already hold; no-op
    assert hw._mode == "hold"
    # Simulate recording state, then switch — should emit released
    hw._recording = True
    hw.set_mode("toggle")
    assert hw._mode == "toggle"
    assert fired == ["released"], f"expected one released emit, got {fired}"
    assert hw._recording is False, "mode switch should force-release"
    # Invalid mode → fallback to hold
    hw.set_mode("nonsense")
    assert hw._mode == "hold"
    # Cleanup: unhook the global keyboard hooks the constructor registered
    try:
        import keyboard
        keyboard.unhook_all()
    except Exception:
        pass
    return "hold/toggle/invalid all behave correctly"


def test_audio_recorder_set_methods():
    """v1.6: AudioRecorder set_device / set_gain don't crash without a mic."""
    rec = t.AudioRecorder()
    rec.set_gain(2.5)
    assert rec._gain == 2.5
    rec.set_gain(99.0)  # clamp
    assert rec._gain == 10.0
    rec.set_gain(0.01)  # clamp
    assert rec._gain == 0.1
    rec.set_device(None)
    assert rec._device is None
    rec.set_device(0)
    assert rec._device == 0
    # list_input_devices should return a list (possibly empty if no mics)
    devs = t.list_input_devices()
    assert isinstance(devs, list)
    return f"set_device/set_gain OK; {len(devs)} input devices enumerated"


# -------------------------------------------------------------------------
# run
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Tellur smoke test — version {t.__version__}\n")
    check("settings: roundtrip all fields",       test_settings_roundtrip)
    check("settings: input_gain & retention clamp", test_settings_clamps_input_gain)
    check("transcript log: purge_older_than",     test_transcript_log_purge)
    check("export_history: all 4 formats",        test_export_history_formats)
    check("MarkdownDailySink: daily file format", test_markdown_daily_sink)
    check("WebhookSink: JSON vs plain detection", test_webhook_sink_json_vs_text)
    check("smart defaults + voice commands",      test_smart_defaults_and_voice_commands)
    check("LLM prompt registry",                  test_llm_prompt_registry)
    check("HotkeyWatcher: hold/toggle modes",     test_hotkey_watcher_modes)
    check("AudioRecorder: set_device/set_gain",   test_audio_recorder_set_methods)

    # LLM live test — checks reachability first; reports as SKIP otherwise.
    print()
    print("--- live LLM check against http://localhost:1234/v1 ---")
    try:
        live_result = test_llm_client_against_lm_studio()
        if live_result is None:
            skip("LLM live: chat completion",
                 "LM Studio at http://localhost:1234/v1 not reachable")
        else:
            results.append(("LLM live: chat completion", PASS, live_result))
            print(f"{PASS} LLM live: chat completion  {live_result}")
    except Exception as e:
        results.append(("LLM live: chat completion", FAIL, f"{type(e).__name__}: {e}"))
        print(f"{FAIL} LLM live: chat completion  {type(e).__name__}: {e}")
        traceback.print_exc()

    # Summary
    print()
    print("=" * 70)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped — total {len(results)}")
    if n_fail:
        print()
        print("FAILURES:")
        for name, status, detail in results:
            if status == FAIL:
                print(f"  - {name}: {detail}")
    sys.exit(0 if n_fail == 0 else 1)
