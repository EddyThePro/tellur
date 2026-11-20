"""One-off verification for v2.3.0 terminal-aware paste. Not part of the
smoke suite (it pops a real conhost window briefly). Run with the venv:
D:\\Tellur\\.venv\\Scripts\\python.exe verify_terminal_paste.py
"""
from __future__ import annotations
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from PyQt6.QtWidgets import QApplication  # noqa: E402
_app = QApplication.instance() or QApplication(sys.argv)

import tellur as t  # noqa: E402

failures: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}  {detail}")
    if not cond:
        failures.append(name)


# --- 1. dispatch: "type" strategy must never touch clipboard, must sanitize ---
captured: dict = {}
orig_write, orig_send = t.keyboard.write, t.keyboard.send
orig_copy, orig_paste_clip = t.pyperclip.copy, t.pyperclip.paste
orig_pick = t.pick_paste_strategy
try:
    t.keyboard.write = lambda text, delay=0, **kw: captured.setdefault("typed", text)
    t.keyboard.send = lambda chord: captured.setdefault("sent", chord)
    t.pyperclip.copy = lambda s: captured.setdefault("clipboard", s)
    t.pyperclip.paste = lambda: "ORIGINAL-CLIPBOARD"
    t.pick_paste_strategy = lambda: "type"

    inj = t.TextInjector()  # no settings → terminal-aware defaults ON
    inj.paste("line one\nline two\ttabbed")
    check("type path: text typed with \\n/\\t flattened",
          captured.get("typed") == "line one line two tabbed",
          repr(captured.get("typed")))
    check("type path: clipboard untouched", "clipboard" not in captured)
    check("type path: no chord sent", "sent" not in captured)

    # press_enter must still send enter after typing
    captured.clear()
    inj.paste("submit me", press_enter=True)
    check("type path: press_enter sends enter", captured.get("sent") == "enter",
          repr(captured.get("sent")))

    # --- 2. dispatch: chord strategy uses clipboard + that chord ---
    captured.clear()
    t.pick_paste_strategy = lambda: "shift+insert"
    inj2 = t.TextInjector()
    inj2.paste("hello ide terminal")
    check("chord path: clipboard set", captured.get("clipboard") == "hello ide terminal",
          repr(captured.get("clipboard")))
    check("chord path: shift+insert sent", captured.get("sent") == "shift+insert",
          repr(captured.get("sent")))
    check("chord path: nothing typed", "typed" not in captured)

    # --- 3. setting off → always ctrl+v even when detector says "type" ---
    captured.clear()
    t.pick_paste_strategy = lambda: "type"

    class _FakeSettings:
        terminal_aware_paste = False

    inj3 = t.TextInjector(settings=_FakeSettings())
    inj3.paste("legacy behavior")
    check("setting off: plain ctrl+v", captured.get("sent") == "ctrl+v",
          repr(captured.get("sent")))
    check("setting off: clipboard set", captured.get("clipboard") == "legacy behavior")
finally:
    t.keyboard.write, t.keyboard.send = orig_write, orig_send
    t.pyperclip.copy, t.pyperclip.paste = orig_copy, orig_paste_clip
    t.pick_paste_strategy = orig_pick

# Give the chord paths' clipboard-restore threads time to run against the
# mocks (CLIPBOARD_RESTORE_DELAY_SEC) before they fall through to the real
# pyperclip — they restore "ORIGINAL-CLIPBOARD" into the mock, harmless.
time.sleep(t.CLIPBOARD_RESTORE_DELAY_SEC + 0.3)

# --- 4. real window class: a live conhost window has CONSOLE_WINDOW_CLASS ---
# (Windows foreground-lock stops a background-spawned window from taking
# focus, so find the spawned console's window by walking top-level windows
# rather than asserting on GetForegroundWindow.)
proc = subprocess.Popen(["conhost.exe", "cmd.exe", "/k", "echo Tellur paste probe"])
try:
    time.sleep(1.5)  # let the window exist
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    found: list[str] = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def _enum(hwnd, _lparam):
        buf = ctypes.create_unicode_buffer(256)
        if user32.GetClassNameW(hwnd, buf, len(buf)) and \
                buf.value == t.CONSOLE_WINDOW_CLASS:
            found.append(buf.value)
        return True

    user32.EnumWindows(_enum, 0)
    check("live conhost window has CONSOLE_WINDOW_CLASS",
          bool(found), f"{len(found)} console-class window(s) found")
finally:
    proc.terminate()

# --- 5. strategy mapping per (class, process), detector mocked ---
orig_info = t.get_foreground_window_info
try:
    cases = [
        (("ConsoleWindowClass", "cmd"), "type"),
        (("ConsoleWindowClass", "powershell"), "type"),
        (("CASCADIA_HOSTING_WINDOW_CLASS", "windowsterminal"), "ctrl+v"),
        # VS Code family stays on ctrl+v — bound in editor, chat panels, AND
        # the integrated terminal; a shift+insert override broke Cursor's
        # chat panel in practice (v2.3.1).
        (("Chrome_WidgetWin_1", "cursor"), "ctrl+v"),
        (("Chrome_WidgetWin_1", "code"), "ctrl+v"),
        ((None, "mintty"), "shift+insert"),
        ((None, "alacritty"), "ctrl+shift+v"),
        ((None, None), "ctrl+v"),
        (("Chrome_WidgetWin_1", "msedge"), "ctrl+v"),
    ]
    for info, expected in cases:
        t.get_foreground_window_info = lambda _i=info: _i
        got = t.pick_paste_strategy()
        check(f"strategy {info} -> {expected}", got == expected, f"got {got!r}")
finally:
    t.get_foreground_window_info = orig_info

print()
if failures:
    print(f"{len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("all verification checks passed")
