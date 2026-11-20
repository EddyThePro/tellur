# Tellur — Free Speech-to-Text Dictation for Windows

[![Latest release](https://img.shields.io/github/v/release/EddyThePro/tellur)](https://github.com/EddyThePro/tellur/releases/latest)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Platform: Windows](https://img.shields.io/badge/platform-windows-blue.svg)](#)

**Tellur is a free, open-source, fully-offline voice-to-text dictation tool for Windows.** Hold **Ctrl+Win**, speak, release — your transcription is pasted straight into whatever window has focus. Discord, browser, code editor, Slack, email, Notion — anything you can type into, you can talk into.

Powered by OpenAI's **Whisper** model running locally on your machine via [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper). No accounts, no cloud, no API keys, no telemetry, no subscription. Free forever, MIT licensed.

**[⬇ Download the latest release](https://github.com/EddyThePro/tellur/releases/latest)** — unzip, double-click `run.bat`. First launch installs everything for you.

## Features

- 🎙️ **Push-to-talk voice typing** — hold `Ctrl+Win`, speak, release. Words appear instantly in the focused window.
- ⚡ **Fast on-device transcription** — ~80× realtime on a modern NVIDIA GPU. CPU fallback included.
- 🔒 **100% offline and private** — runs entirely on your computer. Nothing is sent anywhere, ever.
- 💯 **Free, open source, no limits** — MIT licensed. No subscription, no usage caps, no telemetry.
- 🪟 **Tiny live mic-meter overlay** at the bottom of your screen so you can see your voice being captured in real time.
- 📋 **System tray icon** with right-click **Copy last transcription** — perfect when you typed into the wrong window.
- 💻 **Terminal-aware paste** — dictation lands correctly in CMD/PowerShell windows and Claude Code CLI, which can't accept a simulated Ctrl+V while a console TUI is running.
- 🔊 **Mic passthrough (hear-through)** — mix your mic into your headphones at a level you set, so you can hear the room without taking closed IEMs out. The level is independent of your Windows mic volume, so turning it down doesn't make you quieter to anyone else.
- 📜 **Transcript history panel** — every dictation is saved (locally). Re-copy past transcriptions with one click.
- 🧠 **Personal vocabulary dictionary** — teach Tellur the words and phrases Whisper consistently mishears. Live-reloaded, no restart needed.
- 🎯 **Zero setup ceremony** — install Python 3.11 once, then double-click `run.bat`. The launcher handles the venv, dependencies, and model download for you.

## Quick start

1. Install **Python 3.11+** from [python.org](https://www.python.org/downloads/). **Check "Add to PATH"** during install.
2. Download `Tellur-X.Y.Z.zip` from the [latest release](https://github.com/EddyThePro/tellur/releases/latest).
3. Unzip anywhere.
4. Double-click **`run.bat`**.

First launch downloads the Python dependencies (~2.5 GB) and the Whisper model (~1.5 GB). Subsequent launches start silently in your system tray in under 2 seconds.

## Hotkeys

| Hotkey | Action |
|---|---|
| **Ctrl+Win** | Push-to-talk dictation — hold to talk, or use toggle mode (tap on / tap off) |
| **Esc** | Cancel the current recording without transcribing |
| **Ctrl+Win+B** | Re-paste the most recent transcription into the focused window ("B" for "bring back" — Ctrl+Win+V is reserved by Windows 11) |
| **Ctrl+Win+L** | Apply the default AI prompt to the last transcript and paste the result |
| **Ctrl+Win+Q** | Quit Tellur |

Push-to-talk style is configurable in **Settings → Push-to-talk mode** — pick **Hold** (classic) or **Toggle** (tap to start, tap to stop) per your taste.

## AI post-processing (optional, local)

Point Tellur at any OpenAI-compatible endpoint (LM Studio, Ollama, llama.cpp server, vLLM) and pick a default prompt from the list — **Clean up filler words**, **Make it a clean paragraph**, **Make it formal**, **Convert to bullet points**, **Convert to email**, **Convert to Slack message**, **Summarize**. Press **Ctrl+Win+L** to apply it to your last transcript, or enable **Auto-apply** to run it on every transcript before paste. Stays 100% local if your endpoint is local. Configure in **Settings → AI post-processing**.

## What you'll see

The bottom-center overlay also shows a small, persistent **Windows communications microphone** indicator:

- **Small red crossed-out microphone:** the endpoint is actually muted.
- **Small green microphone:** the endpoint is unmuted (this does not mean Tellur is recording).
- **Amber microphone:** the endpoint's state could not be read; no mute state is assumed.

The indicator is just the icon, with no text or background.

It reads Windows Core Audio every 0.4 seconds on a background thread, independently of IEM Control's button events. It follows changes to the default communications input and leaves Tellur's recording meter, engine badge, and input-device settings intact. An app using a different input device will not be covered by this indicator. The overlay remains click-through and does not take keyboard focus.

- **A tiny indicator** at the bottom-center of your primary monitor:
  - dim gray dot = idle
  - red bars = recording (height tracks your live mic level)
  - blue pulse = transcribing
  - green flash = done
  - orange dot = error (check the log)
- **A tray icon** in the Windows system tray. Right-click for:
  - **Open Tellur** — opens the main window.
  - **Copy last transcription** — drops the most recent transcript on the clipboard.
  - **Quit** — same as `Ctrl+Win+Q`.

  Left-click or double-click the tray icon to open the main window.

### Main window

Two tabs:

- **History** — every transcription, newest first. Click an entry to view full text; buttons for *Copy selected*, *Copy most recent*, *Clear history*. Double-click to copy. Persists to disk (last 500 entries).
- **Settings** — *Auto-paste* toggle and *mic-meter sensitivity* slider (live-updates).

Closing the main window only hides it — only **Quit** actually exits the app.

## Personalizing transcription

Four layers steer Whisper toward your vocabulary, all with effectively zero performance cost:

1. **Custom dictionary** — `replacements.json` next to `tellur.py`. A flat `{"wrong": "right"}` map. Matching is case-insensitive, whitespace-tolerant, and word-boundary aware (so `pi` won't replace inside `pickle`). Live-reloaded on every transcription.
2. **Per-app vocabulary overlays** (v2.0) — drop additional JSON files into `<TELLUR_HOME>\replacements.d\` named after an app's executable (`code.json`, `slack.json`, etc.). When that app is the foreground window, its rules merge on top of the base dictionary (per-app wins on conflict). Manage via Settings → Per-app vocabulary.
3. **Whisper `initial_prompt`** — your most recent transcript plus your dictionary values are fed back as context, so the model gets steered toward your vocabulary over a session.
4. **Hotwords** — the same vocab is passed as faster-whisper's `hotwords` attention bias.

### Editing the dictionary from the CLI

```
teach pi .py                        add or update a rule
teach "java script" "JavaScript"    quote multi-word keys
teach --list                        show all rules
teach --remove pi                   delete a rule
```

Or just edit `replacements.json` directly with any text editor.

## Why use Tellur instead of cloud dictation?

| | Tellur | Cloud dictation (Otter, Google, etc.) |
|---|---|---|
| **Cost** | Free forever | Subscription or per-minute |
| **Privacy** | Audio never leaves your machine | Audio uploaded to a third party |
| **Offline** | Works on a plane | Needs internet |
| **Latency** | ~60 ms on GPU | Network round-trip |
| **Vocabulary tuning** | Live-editable JSON dictionary | Usually limited or paid |
| **Source code** | MIT, hackable | Closed |

## Optional cloud engine (ElevenLabs Scribe)

Tellur is local by default and stays that way unless you opt in. **Settings → Transcription engine** offers three modes:

| Mode | What it does |
|---|---|
| **Local Whisper** (default) | Everything on your machine. No network. |
| **ElevenLabs Scribe** | Sends each clip to ElevenLabs' Speech-to-Text API (Scribe v3). Needs an API key and an internet connection; your audio leaves the machine. |
| **Auto** | Local when the GPU is free, cloud when it is busy (a game, a render, a large LLM), so dictation never competes for the GPU. |

A few things worth knowing if you use the cloud modes:

- **Several keys, automatic failover.** Add a primary, a backup and any number of extra keys; a key that runs out of credits or is rate-limited is benched and retried later, and the status line under the engine picker shows which key is live.
- **It can't hang.** Each clip gets a time budget scaled to its length. A slow request is raced by a second one, and past the deadline the clip is decoded locally instead.
- **Clean-up.** *Clean up cloud transcripts* (on by default) leaves out "um", "uh" and restarted words, matching how local Whisper writes. Turn it off for word-for-word transcripts.
- **You can see which engine ran.** When a clip you expected to go to the cloud was decoded locally, the overlay shows a small amber `WHISPER` badge.

## System requirements

- **OS:** Windows 10 or Windows 11
- **Python:** 3.11 (any patch)
- **Disk:** ~5 GB free for the venv + Whisper model
- **GPU:** Optional. NVIDIA GPU with recent drivers gives the fastest experience; CPU fallback works too, just slower.

## Where things live

By default everything is stored under `%LOCALAPPDATA%\Tellur\`:

```
%LOCALAPPDATA%\Tellur\
├── .venv\               Python venv with dependencies
├── hf-cache\            Whisper model (~1.5 GB)
├── logs\tellur.log      rotating log file, 2 MB × 3
├── history.json         transcript log
└── settings.json        user preferences
```

Override paths via env vars before running:

| Variable | Default | Purpose |
|---|---|---|
| `TELLUR_HOME` | `%LOCALAPPDATA%\Tellur` | All app data lives under here. |
| `TELLUR_VENV` | `%TELLUR_HOME%\.venv` | Python venv location. |
| `TELLUR_LOG_DIR` | `%TELLUR_HOME%\logs` | Log file directory. |

To install everything on a different drive (e.g., D:), run this once and re-launch:

```cmd
setx TELLUR_HOME D:\Tellur
```

## Debugging

`tail-log.bat` follows the log in real time. Every dictation logs a line like:

```
INFO  engine  transcribed audio=2.13s in 58ms text='Test one two three'
INFO  app     rewrite 'Test one two three' -> 'Test 1 2 3'
```

so you can see latency, what Whisper heard, and what got rewritten by your dictionary. Uncaught crashes (main thread or worker threads) are captured automatically and logged at `CRITICAL` with a full traceback. Fatal startup errors pop a Qt message box.

Logging is fully asynchronous — every log call enqueues onto a thread-safe queue (microseconds); formatting and file I/O happen on a dedicated listener thread, so logging never blocks transcription or paste.

### CLI flags

```
tellur.py --debug        mirror DEBUG level to console (use python.exe, not pythonw)
tellur.py --version
```

## Hotkey caveats

- **Elevated apps can swallow hotkey events.** If the foreground window runs as Administrator (admin terminal, some games), the Ctrl+Win hold may not register. Run `run.bat` as Administrator to match privilege.
- **Start menu stays away** — holding Ctrl while pressing Win prevents the Start menu from opening on release, so the normal hold→release flow doesn't pop Start.
- **Heavy games used to kill dictation (fixed in 2.3.4).** A AAA game launch can saturate every core long enough that Windows silently revokes low-level keyboard hooks — push-to-talk would go dead until Tellur restarted. Push-to-talk now polls key state directly (no hook to lose), and a watchdog reinstalls the hook the secondary chords (quit / cancel / re-paste) rely on, logging `keyboard hook dead … installing replacement` when it fires. If you game a lot, also consider raising the hook deadline: set `HKCU\Control Panel\Desktop\LowLevelHooksTimeout` (DWORD) to `1000` and sign out/in once.

## Pasting into terminals (CMD, Claude Code, IDE terminals)

Tellur normally pastes by simulating **Ctrl+V** — which only works if the focused window translates that chord into a paste. Classic console windows often can't: a bare `cmd.exe` / PowerShell window disables its Ctrl+V and Shift+Insert shortcuts entirely while a TUI app like **Claude Code** holds the console in raw mode, and a few terminal emulators reserve a different paste chord altogether.

With **Settings → Integration & automation → Terminal-aware paste** (default on), Tellur picks the right strategy per target:

| Focused window | Strategy |
|---|---|
| Classic console window (`cmd.exe`, PowerShell, Claude Code inside one) | Transcript is **typed** as unicode key events; newlines become spaces so dictation can never press Enter in your shell |
| mintty | Clipboard + **Shift+Insert** |
| Alacritty / WezTerm | Clipboard + **Ctrl+Shift+V** |
| Windows Terminal, VS Code-family editors/terminals, and everything else | Clipboard + **Ctrl+V** (unchanged) |

Tip: for Claude Code, **Windows Terminal is the best host** — its Ctrl+V delivers a proper bracketed paste, which is the input form Claude Code is designed for. The typing path exists so plain-CMD-hosted sessions work at all.

## Mic passthrough (hear-through)

**Settings → Mic passthrough.** Mixes your microphone into your headphones so the room stays audible through closed IEMs.

| Control | What it does |
| --- | --- |
| **Hear-through** | Turns the monitor on. Two audio streams open; nothing runs when it's off. |
| **Passthrough level** | `0%` is silent, `100%` is your mic at full level. Squared, so most of the travel sits at the quiet end where it's useful. |
| **Play into** | Which output the mic is echoed to. Devices are labelled with their host API. |
| **Mute while dictating** | Silences the monitor while you hold the hotkey, so you aren't hearing yourself as you dictate. On by default. |

### Why not just use Windows' "Listen to this device"?

Windows can already route a mic to a playback device, and it needs no running process at all. But its only volume control is the mic's **capture** level, which is shared with every other app — dropping the passthrough to 20% would also make you 20% quieter to everyone in Discord. Tellur mixes it separately, so the level you hear is independent of the level others hear.

### Latency, EQ, and feedback

- **Pick a WASAPI device if you hear yourself echoing.** MME buffers roughly 80 ms; WASAPI is closer to ~15 ms. The picker shows the host API next to each device for exactly this reason.
- **Any system EQ on the output applies to the passthrough too.** If Equalizer APO (or similar) is boosting that device, the mic gets boosted with it — start the level low.
- **On open-back headphones or speakers, a high level can feed back.** Closed IEMs leak very little, which is the case this is built for.

## Tuning

Constants near the top of `tellur.py`:

- `DEFAULT_MODEL` — `distil-large-v3.5` is the sweet spot for English dictation (best short-form accuracy in the Whisper family, ~1.5× faster than turbo). Try `large-v3-turbo` for multilingual or long clips, or `medium` for lower VRAM use. You can also switch models live from **Settings → Transcription model**.
- `LANGUAGE` — `"en"` by default; set to `None` for auto-detect across languages.
- `BAR_LEVEL_SCALE` — default mic-meter sensitivity (also adjustable from Settings).
- `OVERLAY_WIDTH` / `OVERLAY_HEIGHT` / `OVERLAY_BOTTOM_MARGIN` — overlay size & position.
- `BEAM_SIZE` / `NO_REPEAT_NGRAM_SIZE` / `REPETITION_PENALTY` — decoder robustness knobs.
- `PASSTHROUGH_BLOCK` / `PASSTHROUGH_QUEUE_BLOCKS` — hear-through delay. The block size is what you hear yourself at; the queue length caps how far behind the monitor can drift before old audio is dropped. Both are small on purpose: past roughly 20 ms, hearing your own voice delayed actively disrupts speaking.

## How it works (technical)

Tellur is a single-file Python desktop app built on:

- **[faster-whisper](https://github.com/SYSTRAN/faster-whisper)** running `distil-large-v3.5` (distilled Whisper large-v3) through CTranslate2 (fp16 on CUDA, int8 on CPU)
- **PyQt6** for the overlay, system tray icon, and main panel
- **sounddevice** + numpy for low-latency mic capture
- **`keyboard`** (Windows low-level hooks) for push-to-talk detection
- **`pyperclip`** + synthesized Ctrl+V for text injection

With the default local engine, all audio capture and Whisper inference happens on the local machine: no network calls during normal use (a model already on disk is loaded straight from the cache). The one exception is the update check against GitHub releases at start-up. The optional cloud engine is described above.

## File layout

```
tellur/
├── tellur.py            main app
├── replacements.json    custom dictionary
├── requirements.txt
├── run.bat              launcher (creates venv on first run, then pythonw)
├── run-hidden.vbs       runs run.bat with no console window (used by the start-on-login task)
├── tail-log.bat         live log tail
├── teach.bat / teach.py CLI for editing the dictionary
├── smoke_test.py        offline self-test of settings, history, hotkeys, paste strategy
├── test_cloud_router.py offline tests for the cloud/local router
├── CHANGELOG.md
├── README.md
└── LICENSE              MIT
```

## License

MIT. See [LICENSE](LICENSE). Use it, fork it, ship it, no strings attached.

## Keywords

Free speech-to-text · voice-to-text · dictation software · Windows dictation · push-to-talk dictation · offline transcription · local Whisper · faster-whisper · open-source voice typing · private speech recognition · free Otter.ai alternative · free Dragon NaturallySpeaking alternative · free WhisperFlow alternative · free Wispr Flow alternative · local AI dictation
