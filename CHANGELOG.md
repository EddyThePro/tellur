# Changelog

All notable changes to Tellur are documented here. This project follows [Semantic Versioning](https://semver.org/).

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
