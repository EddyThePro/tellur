# Changelog

All notable changes to Tellur are documented here. This project follows [Semantic Versioning](https://semver.org/).

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
