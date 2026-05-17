# Changelog

All notable changes to Tellur are documented here. This project follows [Semantic Versioning](https://semver.org/).

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
