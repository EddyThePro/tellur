# Tellur

Push-to-talk voice dictation that runs entirely on your machine. Hold **Ctrl+Win**, speak, release — the transcription is pasted into whatever window has focus.

No accounts, no cloud, no API keys. Whisper runs locally on your GPU (or CPU as a fallback). Free and MIT-licensed.

## Why "Tellur"

It's named after **H₂Te (hydrogen telluride)** — the closest real compound to "H₂T," which decodes as Hold-to-Type (Hold = H, to = 2, Type = T). Same kind of chemistry-pun naming as a dictation-app brand should have.

## Requirements

- Windows 10 / 11
- Python 3.11 (any patch). Get it from [python.org](https://www.python.org/downloads/) — tick "Add to PATH" during install.
- ~3 GB free disk for the venv + Whisper model
- Optional: NVIDIA GPU with recent drivers. CPU works too, just slower.

## Install + run

1. Clone or download this repo.
2. Double-click **`run.bat`**.

On first run, `run.bat` creates a venv at `%LOCALAPPDATA%\Tellur\.venv`, installs the Python dependencies, and launches the app. On subsequent runs it just launches silently (no console window).

The first transcription downloads the Whisper model (~1.5 GB) into `%LOCALAPPDATA%\Tellur\hf-cache`. You'll see a short delay the first time; afterwards it's cached.

## What you'll see

- **A tiny indicator** at the bottom-center of your primary monitor.
  - dim gray = idle
  - red bars = recording (height tracks your live mic level)
  - blue pulse = transcribing
  - green flash = done
  - orange = error (check the log)
- **A tray icon** in the Windows system tray. Right-click for:
  - **Open Tellur** — opens the main window.
  - **Copy last transcription** — drops the most recent transcript on the clipboard (silent — no toast).
  - **Quit** — same as `Ctrl+Win+Q`.

  Left-click or double-click also opens the main window.

### Main window

Two tabs:

- **History** — every transcription this session and prior sessions, newest first. Click an entry to see its full text; buttons for *Copy selected*, *Copy most recent*, *Clear history*. Double-click an entry to copy it. Persists to `history.json` (last 500 entries).
- **Settings** — *Auto-paste* toggle (turn off if you'd rather grab text from history) and *mic-meter sensitivity* slider (live-updates the bar response).

Closing the main window hides it — only **Quit** actually exits the app.

## Personalizing transcription

Three layers, all near-zero cost:

1. **Custom dictionary** — `replacements.json` next to `tellur.py`. Flat `{"wrong": "right"}` map. Matching is case-insensitive, whitespace-tolerant, word-boundary aware (so `pi` won't replace inside `pickle`). Live-reloaded on every transcription.
2. **Whisper `initial_prompt`** — your last 3 transcripts plus your dictionary values are fed back as Whisper context, so the model gets steered toward your vocabulary over the course of a session.
3. **Hotwords** — same vocab is passed as faster-whisper's `hotwords` attention bias.

### Editing the dictionary from the CLI

```
teach pi .py                        add or update a rule
teach "java script" "JavaScript"    quote multi-word keys
teach --list                        show all rules
teach --remove pi                   delete a rule
```

Or just edit `replacements.json` directly with any editor.

## Where things live

By default everything lives under `%LOCALAPPDATA%\Tellur\`:

```
%LOCALAPPDATA%\Tellur\
├── .venv\               Python venv with dependencies
├── hf-cache\            Whisper model (~1.5 GB)
├── logs\tellur.log      rotating, 2 MB × 3
├── history.json         transcript log
└── settings.json        user preferences
```

Override paths via env vars before running:

| Variable | Default | Purpose |
|---|---|---|
| `TELLUR_HOME` | `%LOCALAPPDATA%\Tellur` | All app data lives under here. |
| `TELLUR_VENV` | `%TELLUR_HOME%\.venv` | Python venv location. |
| `TELLUR_LOG_DIR` | `%TELLUR_HOME%\logs` | Log file directory. |

To put everything on a different drive (e.g., D:), run:

```cmd
setx TELLUR_HOME D:\Tellur
```

…then start a new terminal and run `run.bat`.

## Debugging

`tail-log.bat` follows the log in real time. Every dictation logs a line like:

```
INFO  engine  transcribed audio=2.13s in 58ms text='Test one two three'
INFO  app     rewrite 'Test one two three' -> 'Test 1 2 3'
```

so you can see latency, what Whisper heard, what got rewritten. Uncaught crashes (main thread OR worker thread) are captured by `sys.excepthook` / `threading.excepthook` and hit the log at `CRITICAL` with a full traceback. Fatal startup errors also pop a Qt message box.

Logging is **asynchronous** — every log call enqueues onto a thread-safe queue (microseconds); formatting + file I/O happen on a dedicated listener thread, so logging never blocks transcription or paste.

### CLI flags

```
tellur.py --debug        mirror DEBUG level to console (use python.exe, not pythonw)
tellur.py --version
```

## Hotkey caveats

- **Elevated apps** can swallow hotkey events. If the foreground window runs as Administrator (an admin terminal, some games), our hold-detection may not fire. Run `run.bat` as Administrator to match privilege.
- **Start menu**: holding Ctrl while pressing Win prevents the Start menu from opening on release, so the normal hold→release flow works cleanly.

## Tuning

Constants near the top of `tellur.py`:

- `MODEL_NAME` — `large-v3-turbo` is the sweet spot. `distil-large-v3` is faster (English only). `medium` is smaller if VRAM matters elsewhere.
- `LANGUAGE` — `"en"` by default; set to `None` for auto-detect.
- `BAR_LEVEL_SCALE` — default mic-meter sensitivity (also adjustable from Settings).
- `OVERLAY_WIDTH` / `OVERLAY_HEIGHT` / `OVERLAY_BOTTOM_MARGIN` — pill size & position.

## File layout

```
tellur/
├── tellur.py            main app
├── replacements.json    custom dictionary
├── requirements.txt
├── run.bat              launcher (creates venv on first run, then pythonw)
├── tail-log.bat         live log tail
├── teach.bat / teach.py CLI for editing the dictionary
├── README.md
└── LICENSE              MIT
```

## License

MIT. See `LICENSE`.
