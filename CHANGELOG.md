# Changelog

All notable changes to Tellur are documented here. This project follows [Semantic Versioning](https://semver.org/).

## [1.6.0] — 2026-05-17

**Phase 4 of the roadmap — audio control.** You can now pick a specific microphone, see your input level live, and boost a quiet mic with software gain. Before this release Tellur always followed the Windows default input device and ran every sample at unit gain.

### Microphone device picker

- **Settings → Audio input → Microphone device** dropdown lists every input device PortAudio sees, with the system default tagged. Pick **System default** to keep tracking Windows's choice (the previous behavior), or pin a specific mic so headset-vs-laptop switches don't change Tellur's input.
- The selection is persisted by **device name**, not index, so plugging/unplugging USB peripherals doesn't shuffle which device Tellur picks next time.
- If a previously-pinned device is gone at startup, Tellur logs a warning and falls back to the system default rather than refusing to record.

### Live input level meter

- A live RMS bar appears in Settings while the window is open. It runs a passive monitor stream so you can speak into your mic and see the meter respond *before* you start a real dictation — handy for verifying the right device is selected and that gain is set sensibly.
- The monitor stream auto-stops when Settings is hidden (so we don't keep a mic open in the background) and yields to push-to-talk: starting a recording closes the monitor; releasing the hotkey brings it back if Settings is still open.

### Input gain slider

- **0.10× → 4.00×** software gain, applied to every sample before Whisper sees it (and also to the meter, so the bar shows what Whisper will hear).
- Quiet mic? Push it up. Clipping on loud bursts? Push it down. Clamping at ±1.0 prevents wraparound distortion at high gain.
- Live-applies on slider drag; saves debounced at 300ms.

### Under the hood

- `AudioRecorder` learned `set_device()`, `set_gain()`, and a separate `start_monitor()` / `stop_monitor()` lifecycle for the Settings meter. The monitor and the recording use the same configured device + gain so the meter accurately previews what Whisper would receive.
- New `list_input_devices()` helper wraps `sd.query_devices()` with input-only filtering and default-device tagging, returning a clean list of `{index, name, default, channels}` dicts.
- `Settings` persists two new fields: `input_device_name` (string or null) and `input_gain` (float, 0.1–10.0, clamped on load).
- `MainPanel` gained `showEvent` / `hideEvent` overrides to start/stop the monitor stream — no background mic activity once you close Settings.

---

## [1.5.0] — 2026-05-17

**Phase 3 of the roadmap — modes & hotkey flexibility.** Choose between hold and toggle push-to-talk styles, abandon a recording mid-flight with Esc, and re-paste your last transcript into a fresh window with one keystroke.

### Push-to-talk mode picker

- **Settings → Push-to-talk mode** dropdown, with two options:
  - **Hold** (default, unchanged behavior) — speak while Ctrl+Win is held, paste on release.
  - **Toggle** — tap Ctrl+Win once to start recording, tap again to stop. Lets you dictate longer passages without finger fatigue and frees your hands to keep typing or scrolling between thoughts.
- The choice is live-applied — switching mode while idle just changes how the next dictation starts; switching mid-recording cleanly ends the in-flight session.
- Persisted to `settings.json` as `hotkey_mode` (`"hold"` or `"toggle"`).

### Cancel recording mid-flight

- **Esc while recording → abandon the recording without transcribing.** Works in both modes. Mic stops, overlay returns to idle, no clipboard pollution, no history entry. Esc outside of an active recording does nothing Tellur-specific (the key still propagates to your focused app).
- Useful for: trailed-off sentences, accidental hold-presses, "wait, never mind".

### Re-paste last transcript

- **Ctrl+Win+V → paste the most recent transcript into whatever window is focused right now.** Identical to right-click tray → "Copy last transcription" + Ctrl+V, but as a single global hotkey.
- Made for the "Tellur auto-pasted into the wrong window" recovery flow and "I want to say that same thing again in a different app" use case.

### Hotkey reference in Settings

Settings tab now includes a clear table of every global hotkey Tellur listens for, so you don't have to remember them or check the README.

| Hotkey | Action |
|---|---|
| **Ctrl+Win** | Push-to-talk (hold or toggle, per setting) |
| **Esc** | Cancel current recording |
| **Ctrl+Win+V** | Re-paste last transcription |
| **Ctrl+Win+Q** | Quit Tellur |

### Under the hood

- `HotkeyWatcher` rewritten with explicit mode state and a `force_release()` cancel path. Each press in toggle mode flips a logical `_recording` flag; key releases are ignored. In hold mode, recording follows the key state 1:1 (unchanged).
- New signals: `cancel_requested`, `repaste_requested`. New global hotkey registrations for `esc` and `ctrl+win+v`.
- App tracks a per-recording `_cancel_current_recording` flag so the release path can skip the transcribe stage cleanly without racing the audio buffer.

---

## [1.4.0] — 2026-05-17

**Phase 2 of the roadmap.** Two changes to how transcripts get post-processed: a tiny always-on polish pass that everyone benefits from, plus an opt-in "voice commands" mode for users who want to dictate punctuation explicitly.

### Always-on smart defaults

Runs on every transcription regardless of any setting:

- **Capitalize the first letter** of the transcript.
- **Strip trailing weak punctuation** — an orphan trailing comma / semicolon / colon (which Whisper sometimes leaves on incomplete thoughts) gets dropped.
- **Ensure terminal punctuation** — if the transcript doesn't end in `.` / `!` / `?`, a period is appended.

Whisper-v3 already handles tonality- and pause-based punctuation in the base transcription (commas at natural pauses, periods at sentence ends, question marks for interrogative intonation). This is just the polish layer on top.

### Voice commands (opt-in)

- **Settings → Voice commands & punctuation dictation** checkbox (default **off**, opt-in). When enabled, Whisper's output runs through a new `VoiceCommandProcessor` *after* the user dictionary. Capabilities:

  **Spoken punctuation** — dictated words become actual punctuation marks:
  - `comma` → `,` · `period` / `full stop` → `.`
  - `question mark` → `?` · `exclamation point` / `exclamation mark` → `!`
  - `colon` → `:` · `semicolon` → `;` · `ellipsis` → `…`
  - `open quote` / `close quote` → `"` (directional — attach correctly to neighbouring word)
  - `open parenthesis` / `close parenthesis` (and `paren`, `bracket`, `brace` variants)
  - `dash` / `hyphen` → `-` (attached) · `em dash` → `—` (kept with spaces)

  **Structural commands** — produce real whitespace:
  - `new line` → newline · `new paragraph` → double-newline · `tab` → tab

  **Editing commands** — modify the current transcription:
  - `scratch that` / `delete that` / `scratch it` — drops everything from the most recent sentence boundary back through the trigger phrase

  **Auto-capitalization** — the first letter of each sentence is uppercased automatically (start of utterance + any letter following `.` / `!` / `?` + whitespace).

  Direction-aware whitespace handling: "trailing" punctuation (comma, period, close-paren, …) eats the space before it so it attaches cleanly to the previous word; "leading" punctuation (open-paren, open-quote, …) eats the space after it.

  **Plays nicely with Whisper's own auto-punctuation.** Whisper-v3 already adds commas and periods based on speech tonality — saying "test one comma two period" might come out of Whisper as `Test 1, 2, period.` (Whisper adds its commas + transcribes "period" literally) — so the processor post-processes to dedupe adjacent punctuation (`,..` → `.` — strongest wins) and strips orphan commas at the start of new lines. End result: `Test 1, 2.` instead of `Test 1, 2,..`.

### Default OFF

Existing users get auto-updated to v1.4.0 with voice commands disabled — toggle on via Settings if you want them. Saying "comma" in normal speech with the toggle off still produces the literal word.

---

## [1.3.1] — 2026-05-17

UI overhaul for the v1.3 history panel — both visual fixes and a simplification of the per-row actions.

### Changed

- **Per-row actions: just a copy button now.** The pin and delete icons are gone. Pinning added clutter for marginal value, and per-row delete is a misclick hazard.
- **"Delete selected" moved to the top-left toolbar**, next to the search box. Only enabled when a row is selected. Always confirms with a Yes/No dialog before deleting — same path used by the Delete key and the right-click → Delete… menu entry. The Yes/No dialog includes a preview of the transcript so you know what you're about to lose.
- **History sort is just newest-first now.** Pinning machinery removed (the `pinned` field is silently ignored on load, so v1.3.0 history files keep working).
- **The copy button is inset 14 px from the right edge**, not flush against it.

### Fixed

- **Long transcript text no longer hard-clips** under the meta column. Added an `ElidedLabel` (QLabel subclass) that re-ellipsizes via `QFontMetrics.elidedText()` on every resize.
- **The copy icon now renders reliably.** The previous emoji-based icon (📋) depended on the system emoji font and was effectively invisible on Windows's default text font. Replaced with a custom `IconButton` (`QToolButton` + `autoRaise=True`) that paints its own clipboard glyph via `QPainter`.
- **Row widgets now fit the QListWidget viewport.** Added a `HistoryList` subclass that re-fits every row to viewport width on `resizeEvent` and `refit_rows()` after every refresh. Without this, custom row widgets rendered at their natural sizeHint (huge for long transcripts), pushing the meta column and buttons past the right edge — invisible behind a horizontal scrollbar.
- **Single row separator** between transcripts. The previous double-line was a leftover item-level border plus a new row-level border; cleaned up to one row-level border-bottom rendered through `WA_StyledBackground`.
- **Selection highlight fills the row.** The leftover item-level padding from v1.3.0 was creating dead space above and below the highlight; removed.
- **Vertical centering** — explicit `AlignVCenter` on text + meta labels so they line up with the fixed-size copy button.

### Removed

- `TranscriptLog.toggle_pin` and the `pin_toggled` signal on `HistoryRow` — no consumers after the UI simplification.
- `Ctrl+P` keyboard shortcut for pin (since pinning is gone).

---

## [1.3.0] — 2026-05-17

History panel overhaul. Every row now has inline actions; the "select then click the bottom button" friction is gone.

### Added

- **Per-row buttons** on every History entry, always visible:
  - **★ / ☆** — pin / unpin (pinned items float to the top of the list)
  - **📋** — copy that transcription to the clipboard
  - **✕** — delete that transcription
- **Search/filter box** at the top of the History tab — instant case-insensitive search across every saved transcript. Ctrl+F focuses it.
- **Pinned section** — starred entries always show at the top, separated from chronological entries. They also survive the 500-entry rolling cap.
- **Right-click context menu** on any row:
  - Copy
  - Copy with timestamp
  - Copy as markdown quote
  - Pin / Unpin
  - Edit text…
  - Delete
- **Inline edit** — double-click a row's text to fix typos. The edited text replaces the original (and is flagged with `· edited` in the row's metadata).
- **Word count** per row (in addition to the time-ago badge).
- **Keyboard shortcuts** when the History list is focused:
  - `Enter` — copy the selected row
  - `Delete` — delete the selected row
  - `Ctrl+P` — toggle pin on the selected row
  - `Ctrl+F` — focus the search box

### Changed

- `TranscriptLog` grows three new methods (`toggle_pin`, `delete_entry`, `update_text`) and three new Qt signals (`entry_changed`, `entry_removed`, plus the existing `entry_added` / `cleared`). All persist through the async save thread.

---

## [1.2.2] — 2026-05-17

Stability release — no new features, but a wide sweep of real and latent bugs caught during a deep audit pass.

### Fixed — concurrency

- **Engine: transcribe vs model-swap thread safety.** `transcribe()` now holds an engine state lock for the duration of the decode, so a concurrent `switch_to()` can't tear down the model mid-call. The model build still happens outside any lock, so a multi-minute model download never blocks dictation.
- **Engine: concurrent builds on startup.** Switching models before warm-up finished could launch two CUDA model builds in parallel and OOM the GPU. A dedicated build-lock now serializes builds while leaving in-flight transcribes unblocked.
- **Generation counter race.** The `_latest_gen` counter that prevents stale state emits is now protected by its own lock, so cross-thread reads can't see torn writes. Adds an early-stale-check in `_process_audio` to also drop work before it hits the GPU.
- **TranscriptLog: concurrent saves.** Replaced per-add save threads (which raced on a shared `.tmp` filename) with a single dedicated saver thread fed by a coalescing queue. Bursty dictations now write at most one final snapshot.

### Fixed — correctness

- **BOM tolerance** on `settings.json`, `history.json`, and `replacements.json`. A Notepad-saved file with a UTF-8 BOM no longer silently wipes user data; reads use `utf-8-sig`, writes still use `utf-8` (no BOM).
- **`pythonw` stderr/stdout fix moved to top of file** — runs before third-party imports, so any future library that writes to stderr at import time can't crash a silent boot.
- **Audio buffer cap.** Holding the hotkey for hours no longer accumulates unbounded RAM. Capped at `MAX_RECORDING_SECONDS = 300` (5 min); further audio is silently dropped.
- **`Settings.model_name` validation.** Hand-edit to a bogus model name no longer causes an unrecoverable startup error — falls back to default with a warning in the log.
- **Pre-release version comparison.** Update check now uses `packaging.version` so `1.2.0-rc1` correctly sorts below `1.2.0` (the previous naive parse would have flagged a pre-release as "newer" than the matching release).
- **`_warm_up` error recovery.** A failed model load no longer leaves the overlay stuck in the orange "error" state forever; it auto-resets to idle after 3 seconds.
- **TextInjector stuck-state fix.** If clipboard copy or keystroke send fails after the original clipboard was captured, the saved state is now rolled back instead of leaking forever (which would cause a future paste's restore thread to clobber a much-newer clipboard).
- **Update helper handles wrapper-directory archives.** If someone downloads GitHub's "Source code (zip)" instead of the named release asset (which wraps everything in `tellur-X.Y.Z/`), the helper now detects the wrapper and treats its contents as the source root.
- **Update download size cap.** Refuse any release zip larger than 250 MB before downloading; sanity guard against a misconfigured release asset filling the temp drive.
- **`_pid_alive` uses `WaitForSingleObject` on Windows** instead of `GetExitCodeProcess`. The previous code would have treated a parent that exited with code 259 as still alive (a known Windows footgun).
- **`_PRESERVE_FILES` match by full relative path** rather than basename — future-proofing in case we ever ship a same-named file in a subdirectory.
- **Multi-monitor.** The overlay now repositions itself if the user changes their primary monitor while the app is running.
- **`keyboard.unhook_all()` on quit** — releases the global Win32 keyboard hook cleanly instead of leaking on abnormal exits.

### Fixed — tooling

- **`teach.py` writes atomically** (temp + rename), so a power loss or kill mid-write can't corrupt `replacements.json`.
- **`teach.bat` and `tail-log.bat`** now read `TELLUR_HOME` from `HKCU\Environment` if it isn't set in the process env, matching `run.bat`. Previously they'd fall back to `%LOCALAPPDATA%\Tellur` even when `setx TELLUR_HOME D:\Tellur` had been run.
- **`run.bat`** now probes `py -3.11 --version` separately so the user gets a clear "Python 3.11 specifically is required" message if they have e.g. only 3.10 or 3.12 installed.
- **README** corrected: `MODEL_NAME` → `DEFAULT_MODEL` (the actual constant name).

---

## [1.2.1] — 2026-05-17

### Changed

- **Real Tellur branding** everywhere a Python feather used to show up:
  - Main window title bar
  - Windows taskbar (using an explicit `AppUserModelID` so Windows groups the app under "Tellur" instead of "pythonw.exe")
  - Alt+Tab switcher
  - QMessageBox popups

  The tray icon is intentionally unchanged — the simple painted red-dot fits better at 16×16 in the system tray than a downscaled logo.

---

## [1.2.0] — 2026-05-17

### Added

- **Model picker.** Settings → Transcription model now has a dropdown with five Whisper variants covering the full speed/accuracy/size range:
  - **Tiny** (75 MB) — fastest, low accuracy. For old or slow PCs.
  - **Small** (466 MB) — balanced for modest hardware.
  - **Distil-Large v3** (1.5 GB) — fast, English only.
  - **Large v3 turbo** (1.6 GB) — default. Recommended for most users.
  - **Large v3** (3.0 GB) — highest accuracy, slower on weaker GPUs.

  Selecting a model that isn't on disk yet triggers a background download (via faster-whisper / HuggingFace Hub). Your choice persists across restarts.

- **Cache-aware switching.** Tellur checks whether a model is already on disk before claiming to "download" it. Cached models switch instantly with a "Switching to X…" message; only first-time switches show the download status.

- **Live, byte-accurate download progress** with three distinct phases so there's never a silent stretch:
  1. *Connecting…* — animated indeterminate bar during HF Hub's API + ETag + connection setup.
  2. *Downloading X… 42 / 75 MB* — determinate bar that updates as bytes arrive (hooks directly into HuggingFace Hub's per-chunk progress; no dir-polling).
  3. *Loading X into memory…* — animated indeterminate bar during the final VRAM-load step.

  After success the bar hides and the status reads `Active: X`.

- **Dropdown markers.** Each entry in the model dropdown is annotated with `✓ downloaded` if the model is already cached locally, so you can tell at a glance which selections won't require a download.

---

## [1.1.0] — 2026-05-17

### Added

- **In-app auto-update.** Tellur now checks GitHub on startup and surfaces new versions both in the **Settings → Software updates** section and in the system-tray right-click menu. Click **Install** and Tellur downloads the new release, swaps the files cleanly, and restarts itself — no manual download, no website visit. Your `replacements.json` and all data in `%LOCALAPPDATA%\Tellur\` are preserved across upgrades.

### Notes for upgrading from 1.0.x

The auto-updater lives inside `tellur.py`, so the **first time** you move from 1.0.x to 1.1.0+ you still have to download the zip manually (because the old version doesn't know how to update itself). From 1.1.0 onward, every future upgrade is one click.

---

## [1.0.1] — 2026-05-17

### Fixed

- **Whisper hallucinating dictionary vocabulary.** On unclear or paused audio, transcriptions would sometimes come out as fragments of the custom dictionary (`HTML1, HTML2, HTML3, ...` or `.s, .toml, .x, .md, ...`) instead of real transcribed speech.
- **Empty transcriptions on real audio.** Short clips would sometimes return an empty string with suspiciously fast transcribe times (~15–30 ms), as if the model had decided the audio was "already covered" and skipped decoding.

**Root cause:** The decoder's `initial_prompt` was being stuffed with the entire dictionary as a comma-separated vocabulary list. Whisper treats `initial_prompt` as "text that came BEFORE this audio," and a long vocabulary list there poisoned the decoder — sometimes it hallucinated those tokens into the output, sometimes it short-circuited and returned nothing.

**Fix:** Dictionary vocabulary is no longer placed in `initial_prompt`. It continues to steer Whisper via the `hotwords` parameter (attention bias, the safe mechanism). Recent transcripts continue to be fed as legitimate prompt context.

### Upgrading

Download `Tellur-1.0.1.zip` from the [latest release](https://github.com/EddyThePro/tellur/releases/latest) and replace your existing install folder. Your history, settings, and dictionary persist across upgrades — they live under `%LOCALAPPDATA%\Tellur\` (or `TELLUR_HOME` if you set it).

---

## [1.0.0] — 2026-05-17

Initial public release.

- Push-to-talk voice dictation: hold **Ctrl+Win**, speak, release.
- Tiny live mic-meter overlay at the bottom of the screen.
- System tray icon with right-click **Copy last transcription**.
- Main window with full transcription history and basic settings.
- Custom dictionary (`replacements.json`) for personal vocabulary fixes.
- Runs entirely on-device — no API calls, no accounts.
- MIT licensed.
