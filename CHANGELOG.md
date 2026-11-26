# Changelog

All notable changes to Tellur are documented here. This project follows [Semantic Versioning](https://semver.org/).

## [2.7.5] — 2026-11-25

### Recording starts the instant you press the hotkey

**Symptom:** pressing Ctrl+Win took a noticeable moment to start listening, and dictations lost their first words ("I want you to look into Tellur" arrived as "To look into Teller").

**Cause:** the microphone was opened on every press and closed on every release. On a wired or USB mic that costs ~20 ms. On a Bluetooth headset (measured on a Shokz OpenComm2) Windows first has to switch the headset into its hands-free profile: opening the mic took ~600 ms and the first audio arrived ~1.4 s after the press. Windows switches back after 6-12 s idle, so nearly every press paid that again. The log made it plain: press-to-record had been ~30 ms median for a month and jumped to ~600 ms the moment the default mic became the headset, and a 26.5 s hold captured only 25.0 s of audio.

**Fix: Settings → Audio input → Keep microphone open** (off by default).
- The capture stream is opened once and kept running. A press now starts recording in ~0.1 ms, and the 0.3 s before the press is kept too, so a word spoken together with the keypress isn't clipped.
- A short tap is still treated as a tap: the pre-roll is dropped when less than 0.25 s was recorded after the press, so an accidental press never reaches the cloud.
- A background watchdog keeps the held stream healthy. It reopens the stream when it stops delivering audio (headset switched off, USB unplugged), when you pick another mic, and when the Windows default mic changes. PortAudio only notices a new default after re-initialising, so the watchdog does that too. It never re-initialises mid-dictation, or while mic passthrough is enabled (its streams hold device indexes of their own).
- If the held stream isn't ready at press time, the press opens the mic itself, as before. So the setting can only make things faster.
- Trade-offs, also stated in Settings: Windows shows the mic as in use, and a Bluetooth headset stays in call-quality audio while Tellur runs.
- Without the setting, a slow mic is now logged: `mic took N ms to deliver audio after the press`.

### Smaller fixes
- Local Whisper transcripts get a capital "I" ("i want" → "I want", "i'm" → "I'm"). Standalone pronoun only: "i.e.", "iPad" and other words are untouched. About one local clip in five had this.
- Device enumeration and PortAudio re-initialisation are serialised (`PA_LOCK`), so the Settings device list can't race a re-init.
- New offline test: `test_recorder.py` (fake PortAudio, 32 checks).

## [2.7.4] — 2026-11-19

A refresh of everything underneath the app: the cloud model, the inference libraries, and how a cached model is loaded.

### Whisper models: already current

Checked rather than assumed: OpenAI has published no Whisper checkpoint since `large-v3-turbo` (October 2024), Distil-Whisper nothing since `distil-large-v3.5` (March 2025), and faster-whisper 1.2.1 is still the newest release. Tellur's default and its model picker were already on the latest of each, so nothing changes there.

### Cloud engine: ElevenLabs Scribe v3

ElevenLabs released Scribe v3 on 2026-11-10 (27% fewer word errors than v2 by their FLEURS numbers). It uses the same endpoint and request. Measured on this app's own recordings before switching: the same credit cost per clip (read from the quote in a quota rejection, so the comparison cost nothing) and the same latency (0.5-0.7 s for a 10 s clip, 1.2-1.5 s for 30 s).

- `scribe_v3` is the default cloud model. A saved `scribe_v2` is moved over once; an id you set by hand afterwards is respected.
- **Clean-up is on by default.** Out of the box v3 is far more literal than v2 about stumbles — *"I almost just want you to l- to as-- act like an overseer … and just, uh, you know, if there's …"*. The request now sends `no_verbatim=true`, which drops filler words and false starts the way local Whisper already does, plus `numbers_format=digits`, because clean-up otherwise spells numbers out ("preview version two"). Neither changes what a request costs. **Settings → Transcription engine → Clean up cloud transcripts** turns it off for word-for-word output.
- Tried and not adopted, because they made no measurable difference: raw-PCM upload (`file_format=pcm_s16le_16`), `timestamps_granularity=none`, and pinning `language_code`.

### A cached model no longer asks the internet for permission

Every model load — at start-up, and after each VRAM release in auto mode — first asked huggingface.co whether the repository had changed: 50-250 ms when the site answered, its full network timeout when it didn't. For an app that advertises working offline, that was wrong. A model that is already on disk is now opened with `local_files_only=True` (zero connection attempts, verified with the network blocked); only a missing or incomplete cache entry goes online.

### Libraries

Tested together in a fresh environment before touching the live one — all 20 recordings in the rolling debug buffer (579 s of audio) decode to **identical** text on CUDA at the same speed:

| | was | now |
|---|---|---|
| CTranslate2 | 4.7.1 | 4.9.0 |
| cuDNN (`nvidia-cudnn-cu12`) | 9.22 | 9.28 |
| NumPy | 2.2.6 | 2.4.6 |
| PyQt6 / Qt | 6.11.0 / 6.11.1 | 6.12.1 / 6.12.0 |
| onnxruntime (Silero VAD) | 1.26.0 | 1.32.0 |
| huggingface_hub | 1.15.0 | 2.4.0 |
| PyAV | 17.0.1 | 18.1.0 |
| pycaw / comtypes | 20251023 / 1.4.16 | 20260927 / 1.5.0 |
| sounddevice, tokenizers, nvidia-ml-py | | latest |

`requirements.txt`: `faster-whisper>=1.2.1` (1.2.0 is the first release that knows the default model — the old `>=1.0.3` floor allowed installs that could not load it) and the NumPy cap moves from `<2.3` to `<3`.

huggingface_hub 2.x logs through `httpx2`/`httpcore2`, which narrate every request phase at DEBUG (34 lines per model load); they join the silenced-logger list.

### Updates now bring their dependencies

The in-app updater replaces the program files but has never installed packages, so a release that added or raised a dependency started against a stale environment. `run.bat` now compares `requirements.txt` with the copy stamped at the last successful install and runs `pip install -r requirements.txt` when they differ — after the old instance has been stopped (pip cannot replace DLLs in use), and without blocking the launch if the install fails offline.

### Tests

`test_cloud_router.py` is new and ships with the app: 42 offline checks for the key pool (failover, hedged requests, the per-clip deadline), the VRAM-gated model release, the press-time preload, the Scribe request fields and the settings migration. No network, GPU or API key needed.

## [2.7.3] — 2026-11-12

"It's taking a very long time." Diagnosed from three months of `tellur.log` (3,400 dictations) before anything was changed.

### A stalled cloud request could hold a dictation for a full minute

Twice on 2026-11-12 a short clip sent to ElevenLabs simply did not come back: a 4 s clip took **52 s**, an 18 s clip took **61 s** — the second one only because the flat 60-second timeout finally expired and the next key answered in under a second. The pill sat on "transcribing" the whole time, and the text eventually pasted into whichever window had focus a minute later.

Scribe normally answers in about 0.6 s + 0.022 s per second of audio (1,127 cloud clips; the slowest healthy one was 9.5 s for a five-minute recording), so a 60 s timeout was protecting nothing. Each clip now gets a time budget scaled to its length (`eleven_time_budget`):

- **Hedge** — if there's no answer by 2.5 s + 0.04 s/s (about 1% of healthy requests get that far), a second request is raced against the first on a fresh connection and whichever answers first wins.
- **Deadline** — if neither has answered by 6 s + 0.08 s/s, the cloud is abandoned for that clip and local Whisper decodes it instead (`ElevenTimeout`). Nothing is benched; the next clip tries the cloud again.

A 10-second clip therefore hedges at 2.9 s and can never wait past 6.8 s on the network. A request that answers after it was given up on is discarded without touching key state. Checked against the logged latencies: 10 of 1,127 past cloud clips would have triggered the hedge, and 4 would have crossed the deadline — the two stalls, plus two requests that took 6-7 s (which the hedge would most likely have answered first).

### The local model was being unloaded for no reason, then reloaded on your time

Auto mode released the local model's VRAM after three "GPU busy" clips in a row. An idle model costs no GPU time, only memory — and on this machine those releases were happening with 20+ GB of VRAM free. The next local clip then waited out a reload: **100 clips at a 2.06 s median to paste instead of 0.35 s.**

- The release now only happens when free VRAM is actually low (`AUTO_RELEASE_BELOW_FREE_VRAM_MB = 6000`), or when NVML can't report it.
- When the model does need loading (after a real release, or a warm-up deferred at login), it now starts loading **when you press the hotkey**, while you're still talking, instead of at release. `engine.load()` is serialized, so the decode just joins the load in flight.

### Mic-mute indicator: same behaviour, a fraction of the CPU

The overlay's microphone status icon (added after 2.7.2, see below) re-walked the Windows device enumerator and re-opened the endpoint's property store every 0.4 s — about 10 ms of CPU per poll, forever, which was most of Tellur's idle CPU and a 10 ms GIL hold next to the 20 ms hotkey poll. `CommsMicProbe` keeps the endpoint-volume interface and just calls `GetMute()` (microseconds), re-resolving the default device every 3 s or after any error. The GUI is only signalled when the state changes.

### Smaller fixes

- **Dropped audio is now visible.** PortAudio's status flags (input overflow = the driver dropped samples) were ignored; they're counted during a recording and logged once at stop, so "it missed a word" can be told apart from a decode problem.
- `is_start_on_login_enabled()` read every `<Enabled>` element in the scheduled task's XML, including each trigger's. A task with one disabled trigger read as "off" permanently, and every settings save re-ran `schtasks` trying to fix it. It now reads only the task-level switch under `<Settings>`.
- Cloud log lines distinguish *slow* (`ElevenLabs slow: no answer 2.7s into a 4.0s clip — racing a second request`) from *out of keys*, and say which request won.

### Documented late: microphone mute indicator

Shipped in the working tree on 2026-09-13 without a changelog entry: a small microphone glyph under the overlay pill shows the mute state of the Windows default *communications* capture endpoint — red and crossed out when muted, green when live, amber when the state can't be read. It reads the real endpoint state (never inferred from silence) and follows device changes.

### Looked at, deliberately not changed

- **Compressing the upload.** Measured on a 20 s clip: WAV 622 KB, FLAC 248 KB, Opus 71 KB — and the same ~1.0 s round trip for all three. The time is ElevenLabs' processing, not the upload.
- **Connection keep-alive.** DNS + TCP + TLS to the API is ~70 ms of a ~1 s request.
- **Auto-mode routing itself.** With the model resident, a local decode measured 110 ms at 81% GPU load against 657 ms for the cloud on the same clip. The ≥50% rule still routes those clips to the cloud; that's a policy choice (protecting a game's frame time), left as is.

## [2.7.2] — 2026-08-23

A log-driven audit of the whole app after 2.7.1 (three months of `tellur.log`, plus a code review of every subsystem). Everything below was confirmed against the logs or reproduced before it was fixed.

### Hotkeys are now entirely hook-free

The quit / re-paste / LLM / note chords and the Esc cancel were still `keyboard.add_hotkey` bindings on the library's low-level Windows hook. Two real problems came with that:

- **Chords re-fired on keyboard auto-repeat.** Holding Ctrl+Win+Z for a second toggled note mode ~12 times (69 toggles in one minute on 2026-08-08, each one cancelling the live dictation); holding Ctrl+Win+B re-pasted repeatedly.
- **Windows silently revokes the hook under load**, and the watchdog's answer — a replacement message-pump thread — ran roughly hourly (57 replacements in one 8-day session), each one a leaked thread.

Chords now use the same GetAsyncKeyState edge detection as the Ctrl+Win push-to-talk (`parse_chord`, 20 ms poll): one fire per physical press, order-independent, no hook, no watchdog, no thread churn. The hook path survives only as a fallback for a combo string the parser can't map (then the watchdog arms as before).

### Pasting no longer cuts the next recording

`keyboard.send("ctrl+v")` presses and releases Ctrl, and the push-to-talk poll reads the same key state. When a transcript finished while you were already holding Ctrl+Win for the next one, the synthetic release ended that recording after ~0.3 s — 14 cases in the logs, some landing in Windows shell windows. `paste()` now waits (bounded) for Ctrl+Win to be released, pastes are serialized so two finished transcripts go out in order, and re-paste / LLM-apply cancel the in-flight dictation their own chord started (as note-toggle already did) and run off the GUI thread.

### Transcript fixes

- **Scribe audio-event tags** (`[pause]`, `[clicking]`, `[silence]`…) were pasted as text ~40 times. They're now stripped (and `tag_audio_events=false` is sent to the API); a clip that was only tags counts as empty.
- **Dictionary rules no longer rewrite each other's output** — the shipped `"dot html": ".html"` + `"html": "HTML"` pair turned `dashboard.html` into `dashboard.HTML` (three real history entries). The dictionary is now one single-pass alternation, a key glued to a preceding dot (`package.json`) is left alone, and `.py`-style values attach to the previous word (`main dot pi` → `main.py`).
- A backslash in a dictionary value (a Windows path) raised inside `re.sub` and **killed the transcribe thread** — overlay stuck on "transcribing", no paste. Replacement is a lookup callback now, and the whole polish step is wrapped so a bad entry costs the polish, never the transcript.
- `apply_smart_defaults` no longer adds a second period after a closing quote/bracket (`He said "hello."`, `(see above.)`, `Wait…`).
- The per-app dictionary context is captured at key release; it used to read `_current_context` after the next recording had already overwritten it.

### Cloud / router

- Auto mode released the local model's VRAM on the 3rd consecutive busy-routed clip **before** knowing the cloud would serve it; when every key was benched the same clip then reloaded the model on the busy GPU it was avoiding. The release now happens only after the cloud actually served, and the streak resets when it doesn't.
- A connection-level failure (DNS/TLS/timeout) or a 4xx bad-request no longer fans out across every key — it used to re-upload the clip to each of 4 keys with a 60 s timeout apiece.
- `GpuLoadMonitor` re-initialises NVML after a query failure instead of keeping a stale handle (which reported "busy" for the rest of the session).
- "Test key" now reports *out of credits* / *rate-limited* — the client raises `ElevenAuthError` for 401/402/429, which the button's `HTTPError` branch never saw.
- ElevenLabs key cooldowns/balances are persisted with `Settings.save_quiet()`: the old path ran the full `save()` on every rejected request — two `schtasks` subprocesses, re-registering every hotkey and re-resolving audio devices, mid-paste, 263 times across the logs.

### Lifecycle

- **Single-instance guard** (`Local\Tellur-single-instance` mutex): a second launch by any path other than `run.bat` used to mean two pastes per dictation.
- `AudioRecorder` no longer loses the Settings meter after a failed mic start, and no longer reopens an orphan monitor stream when the panel was hidden during a recording; a failed start closes its half-opened stream.
- The updater's install thread is a daemon with a per-read timeout and an in-flight size cap (a stalled download could pin a tray-less process open after Quit), temp files are cleaned on failure, and the post-install restart goes through `App.quit`.
- `Overlay` connects `primaryScreenChanged` to a bound method; the old lambda outlived "Refresh UI" and fired on the deleted widget at the next monitor change.
- History/notes savers are flushed at quit, so a dictation finished right before Ctrl+Win+Q is written.

## [2.7.1] — 2026-08-23

Fixes local transcriptions that came back as a fragment of what was said — `"Also if you can't just answer the question i I'm like is the website a and."` for a 14-second clip, or function-word salad like `"I we , the to do and a , you know, it, I is, I can't,"`. About one in ten local clips over 5 s did this since 2.5.0 switched the local model to `distil-large-v3.5` (and the 2.6.0 auto-router started sending most clips local).

### Root cause

The local decode fed two things into Whisper's `<|startofprev|>` context on every clip: the previous transcript as `initial_prompt`, and the dictionary as `hotwords` (faster-whisper puts both in the same place). Distil-Whisper keeps only 2 of large-v3's 32 decoder layers, and that shallow decoder cannot carry a ~200-token prompt of unrelated text: it emits end-of-text early or drifts into salad. The `no_repeat_ngram_size` / `repetition_penalty` guards were band-aids for loops the prompt itself induced.

Reproduced on 8 SAPI renders of real past dictations plus 5 real mic recordings: the shipped config lost >70% of the words on 4 clips and truncated others; the same model with nothing but the audio was near-verbatim on all 13.

### What changed

- **Prompt-fragile models decode plain.** For any model whose name starts with `distil` (`PROMPT_FRAGILE_MODEL_PREFIXES`), the local decode sends audio only — no `initial_prompt`, no `hotwords`, no repetition penalties. Other local models (turbo, large-v3) keep the previous behaviour. The log line gains a `(plain)` marker.
- **Truncation guard.** A local result under 1.2 words/s on a clip ≥5 s is re-decoded once with the plain config and no VAD; the decode with more words wins. Logged as `local decode looks truncated … re-decoded plain/no-VAD`.
- **Rolling raw-recording buffer** (`keep_recordings`, default 0 = off): keeps the newest N dictation recordings as 16-bit WAV in `<data dir>/debug_recordings/` so a mis-transcribed clip can be replayed through the decoder instead of guessed at. File names carry sample count and RMS to match the `recording stopped` log line.

## [2.7.0] — 2026-08-20

Adds **mic passthrough** ("hear-through"): your mic is mixed into your headphones at a level you control, so the room stays audible without taking closed IEMs out.

### Why here rather than in Windows

Windows can already do this with "Listen to this device", and it needs no running process at all — the audio engine does the mixing. The catch is the volume: the only control is the mic's *capture* level, which is shared with everything else. Turning the passthrough down to 20% would also make you 20% quieter to everyone in Discord. Mixing it inside Tellur costs a resident process, which Tellur already is, and buys a level that is genuinely independent of what other apps hear.

### What changed

- **`PassthroughMonitor`** — a mic-to-speakers monitor built from two independent PortAudio streams joined by a bounded deque. It is not a duplex stream because input and output are usually different devices with free-running clocks, which PortAudio will not open as a pair. The deque's `maxlen` is the latency ceiling: if the output side falls behind, the oldest blocks are dropped rather than letting the delay grow without bound. Runs at 48 kHz, not `SAMPLE_RATE` — 16 kHz is a Whisper *input* rate and sounds muffled played back.
- **Settings → Mic passthrough**: enable, level (0–100%), output device, and "mute while dictating" (on by default).
- **Gain and mute are read live inside the audio callbacks**, so dragging the level slider or ducking for a dictation never restarts a stream.
- **Auto-mute while recording.** Both the dictation and the note paths silence the monitor before opening the recorder. If the mic start still fails, the monitor is torn down entirely and the start is retried once — some drivers refuse a second reader on the same device.
- **Playback devices are chosen by `"<API>::<name>"`,** with an occurrence ordinal on duplicates. Every host API lists the same hardware under nearly the same name, so a bare name always resolved to whichever came first and the WASAPI entries could never be picked; two NVIDIA HDMI outputs then still collided on name *and* API. Keys written before this still resolve by name.
- The picker labels each device with its host API. This is not cosmetic: MME buffers roughly 80 ms against WASAPI's ~15 ms, which for a monitor you hear yourself through is the difference between "present" and "echo".

### Notes

Passthrough is played into the chosen output, so any system EQ on that device (Equalizer APO and the like) applies to it too — start the level low. On open-back headphones or speakers a high level can feed back.

## [2.6.1] — 2026-08-16

Fixes a hard crash (0xc0000409 process abort) the first time 2.6.0's sustained-busy model release fired mid-game.

### Why

`release_local_model()` freed VRAM by dropping the Python reference, which runs CTranslate2's C++ destructor. Destructors are noexcept: if the CUDA context has died — exactly what happens when a game pins the GPU at ~97% and the driver does a WDDM reset — the CUDA error inside the destructor escalates to `std::terminate` and kills the whole process, uncatchably. So the release path crashed in precisely the situation it exists for.

### What changed

- **`_retire_model()`**: device memory is now freed with `ctranslate2`'s `unload_model()` — a normal bound method whose CUDA errors surface as catchable Python exceptions. If it fails, the context was already reset and the driver reclaimed the VRAM; we just log it.
- **Model objects are never garbage-collected mid-session.** Retired models are parked in a graveyard list (a few MB of host RAM per model per session) so GC can never trigger the fatal destructor.
- `switch_to()` had the same latent hazard on model swap; it now retires the old model the same way.

## [2.6.0] — 2026-08-15

Adds an "Auto" STT backend that routes each clip on live GPU load: local Whisper when the GPU is free, ElevenLabs when it's busy.

### Why

The fixed backend choice was wrong in both directions. With ElevenLabs primary, an idle-desktop dictation still paid the network round-trip (~1s) and ~1.11 credits/sec for a clip the local model decodes in ~0.1s for free — and the keys drain in days (see 2.4.0). With local primary, dictating mid-game contends with the game for the GPU: the decode is slow and the game can hitch. The right backend is a function of what the GPU is doing *right now*, so the app should decide per clip.

### What's new

- **`stt_backend: "auto"`** (new Settings option, now first in the Engine picker). Each clip samples NVML (the nvidia-smi library, new `nvidia-ml-py` dep): utilization ≥ 50% or free VRAM below 2.6 GB routes to ElevenLabs, otherwise the clip decodes locally. Verdicts are cached 2 s so the record-start badge and the decode agree. All existing failovers still apply — cloud keys exhausted falls back to local even with a busy GPU, so dictation never fully breaks.
- **The VRAM floor only guards fresh loads.** An already-resident local model needs no new allocation, so `busy_reason(need_vram=False)` ignores the floor for it — the model that's serving you doesn't get evicted just because LM Studio grabbed memory. Both verdicts come from the same cached sample.
- **Sustained load releases the local model.** Three consecutive busy-routed clips drop the loaded model (`release_local_model()`, under the same `_state_lock` discipline as `switch_to`), handing its ~1.6 GB back to the game. It reloads transparently (~1–2 s) on the next local decode.
- **GPU-aware warm-up.** At startup, auto mode warms the local model only if the GPU is free; logging in mid-game defers it exactly like the cloud backends do.
- No NVIDIA GPU / NVML missing → auto simply prefers the cloud (on such a machine local Whisper is CPU-bound, which is the slow path auto exists to avoid).

Verified with an 11-check sandboxed test (real decode on the free-GPU path, simulated 99%-busy for routing/streak/release, `active_stt_engine` badge on live GPU state).

## [2.5.0] — 2026-08-15

Upgrades the local Whisper fallback to `distil-large-v3.5`.

### Why

The local fallback matters exactly when ElevenLabs keys drain (see 2.4.0), and the machine had been set to `distil-large-v3` — not even the app's own recommended `large-v3-turbo`. Distil-Whisper's v3.5 release is a strict upgrade for Tellur's workload: same 756M parameters and footprint as distil-large-v3 but trained on 4× more data (98k hours, 80 epochs), it posts 7.08% short-form OOD WER vs 7.37% for distil-large-v3 and ~7.30% for large-v3-turbo — while running ~1.5× faster than turbo. Dictation clips are short-form by definition, so the one place v3.5 loses to turbo (long-form, by ~1%) doesn't apply. English-only, which Tellur already pins via `LANGUAGE = "en"`. faster-whisper ≥1.2 resolves `distil-large-v3.5` natively (repo `distil-whisper/distil-large-v3.5-ct2`).

Measured on this machine's real debug recordings (RTX 5090, fp16, beam 5): an 8.1s clip decoded in 0.14s vs 0.97s on distil-large-v3, with equal-or-cleaner text.

### What's new

- `distil-large-v3.5` added to the model picker and made `DEFAULT_MODEL`; `large-v3-turbo` stays available and is now summarized as the multilingual/long-clip choice.
- Hard-coded repo fallback map gains the `distil-whisper/distil-large-v3.5-ct2` entry so the download path works even if `faster_whisper.utils._MODELS` import fails.

## [2.4.0] — 2026-07-26

Makes the fallback to local Whisper visible instead of silent, and stops a healthy key from being benched for six hours.

### Why

Dictation had been running on local Whisper for most of the previous day without saying so. Both ElevenLabs keys were genuinely out of credits — key #1 at 0 of 10,000, key #2 at 20 of 10,000 — so `ElevenAllKeysFailed` fired on every clip and the engine quietly fell through to the local decode. Nothing in the UI changed. The only symptom was transcripts that read slightly worse: proper nouns degraded ("Tellur" → "teller", "ElevenLabs" → "11 lab") and long clips took ~10 s instead of ~1 s. The overlay showed the same red bars either way, so there was no way to tell which engine had produced a given transcript.

Two adjacent bugs made the outage stickier than it needed to be. A `429` (back-pressure) was treated identically to a spent monthly quota and cost the key a full six-hour cooldown, so one burst of rapid dictation could take a working key offline for the evening. And a key reporting a *partial* balance — key #2's 20 credits, enough for roughly 18 s of audio — was written off wholesale rather than kept for short clips. Meanwhile "Test key" probes with a ~1 s tone costing 1 credit, so it returned `Primary ✓` on a key that could not cover any real dictation, and cleared the exhausted flag on the way out.

### What's new

- **Engine badge on the overlay.** When local Whisper is serving a clip, the pill grows upward and paints `WHISPER` in amber directly under the level bars. The bottom edge stays pinned, so the badge appears without the pill drifting. Emitted twice per dictation: predictively at record start via `Settings.active_stt_engine()`, so it is up while you are still speaking, then again from `WhisperEngine.last_backend` once the decode returns — which is what catches a failover that happened mid-request.
- **Live engine status in Settings → Transcription engine.** One line naming the backend that will actually serve the next dictation, plus per-key state: `ready`, `rate-limited, retries in 45s`, or `out of credits, retries in 5h 59m · 20 credits left`. Refreshes on a 5 s timer. Red when the cloud engine is selected but local is what's running; dim when local was chosen deliberately.
- **Failure-aware cooldowns.** `mark_eleven_exhausted()` takes a per-key `retry_after_sec`, persisted as `eleven_retry_after`. A `429` gets `ELEVEN_RATELIMIT_RETRY_SEC` (60 s); a spent quota keeps the 6 h default. State written by ≤2.3.4 has no `eleven_retry_after`, and is still read correctly via the old `eleven_exhausted_at` + flat-constant path.
- **Partial balances are used, not discarded.** The balance in a quota rejection is parsed out (`_eleven_credits_remaining`) and stored with a 30-minute TTL. A key with credits left gets a zero-length cooldown and stays eligible; the pool skips it only for clips its known balance cannot cover, estimated at `ELEVEN_CREDITS_PER_SEC = 1.11` (measured: 133 credits for a 120 s clip). A rejected request costs nothing, so a wrong estimate is self-correcting.

- **The key pool is unbounded.** `eleven_api_keys_extra` holds keys 3..N, tried in order after the primary and backup, with a per-line field in Settings and "Test key" covering the whole pool. Two fixed slots was the real ceiling: each ElevenLabs key carries its own monthly allowance, and at ~1.11 credits/sec a 10,000-credit key is only ~150 minutes of dictation a month. Stacking keys is the only way to raise that short of a larger plan. Verified end-to-end with a deliberately-failing cascade — invalid key → drained key → working key, succeeding on #3 with both failures correctly benched.

### Fixed

- **The engine badge is confirmatory, never predictive.** It was raised at record start from `active_stt_engine()`, which is a guess about key state. Now record start emits `"pending"` (clearing it) and the badge is only raised once the transcribe has returned and reported that local Whisper actually ran, so it can never claim local while a key is in fact working. It persists afterwards as a notification until the next hotkey press. Live before-you-talk engine state lives in Settings instead, where being briefly wrong costs nothing.
- **`Settings.save()` no longer mutates global OS state from a non-default settings path.** It unconditionally called `apply_start_on_login()`, which edits the machine-wide `Tellur` scheduled task and is not scoped to `self.path`. Any `Settings` built on another path — a test, a scratch `TELLUR_HOME`, a one-off script — defaults `start_on_login` to `False`, so its first `save()` silently ran `schtasks /change /disable` and killed the user's autostart. This cost a real autostart outage during development, twice, and was misdiagnosed as pre-existing before the cause was found. The reconcile is now gated on `self.path == SETTINGS_FILE`.
- `comtypes` interface-release logging silenced to WARNING. It was emitting a DEBUG line per COM pointer the audio ducker touched — measured at **71% of all log lines**, which cut the history retained in a 2 MB rotation to roughly a third of its intended span.

### Known issue

- `is_start_on_login_enabled()` treats **any** `<Enabled>false</Enabled>` anywhere in the task XML as "autostart off", but scheduled-task XML nests one under `<Settings>` and another under each trigger. A task whose trigger is disabled while the task itself is enabled would read as off. Not currently firing — this machine's task XML carries no `<Enabled>` element at all — but the query is too loose and should be scoped to the `<Settings>` block.

## [2.3.3] — 2026-07-12

Fixes the two ways dictation was silently dying — permanently-benched ElevenLabs keys and quiet recordings decoding to empty text — and retires the per-login UAC prompt via an elevated scheduled task.

### Why

Both ElevenLabs keys hit their monthly quota (key #2 on 2026-07-01, HTTP 401 `quota_exceeded`) and stayed marked exhausted forever: nothing ever re-probed them, so every dictation since logged an ERROR and fell back to local Whisper — even after the quota reset made the keys good again. The local fallback then had its own failure mode: recordings whose peak sat well below full scale (headset mic drifted off-axis) intermittently decoded to `''`, so dictation appeared dead (29 of the last 129 transcriptions were empty). Separately, `run.bat`'s self-elevation triggered a UAC prompt on every launch and login — friction that made "just keep it running elevated" unattractive, and unelevated Tellur can't hotkey or paste into admin terminals at all (Windows UIPI).

### What's new

- **Exhausted ElevenLabs keys heal themselves.** Each exhausted mark now carries a timestamp (`eleven_exhausted_at`, persisted); after a 6-hour cooldown (`ELEVEN_EXHAUSTED_RETRY_SEC`) the failover pool probes the key again on the next dictation. Success re-enables it (with an INFO log); failure re-arms the full cooldown. A monthly quota reset now brings Scribe back on its own within one dictation instead of never.
- **Quiet recordings get rescued before the local decode.** If the buffer's peak is under 0.30, it's linearly scaled so the peak hits 0.70 before hitting Whisper — fixing the empty-text class without touching the Scribe path (Scribe never needed it). Near-silent buffers (peak < 0.003 — a muted or dead mic, not quiet speech) are left alone and logged as a WARNING naming the likely cause, instead of hallucinating text from boosted noise.
- **Local VAD is less trigger-happy.** `vad_filter` threshold lowered from Silero's 0.5 default to 0.35, so quiet-but-real speech stops being classified as silence and dropped wholesale.
- **Start-on-login prefers the elevated "Tellur" scheduled task.** When the task exists, `apply_start_on_login()` / `is_start_on_login_enabled()` toggle and read *it* (`schtasks /change /enable|/disable`) instead of the Run key, and the legacy Run-key entry is deleted so login can't double-launch. Result: silent elevated start at login, zero UAC prompts, admin windows reachable. Installs without the task keep the v2.2.2 Run-key behavior unchanged.
- **Startup logs elevation state** (`running elevated: yes` / `NO — hotkey and paste won't reach admin windows (UIPI)`), so "dictation dead in terminal X" is diagnosable from the log alone.

### Under the hood

- New `Settings.eleven_exhausted_at: dict[str, float]` (round-trips through settings.json; keys listed in `eleven_exhausted` without a timestamp count as due-for-retry immediately, which is exactly what a pre-2.3.3 settings file contains). New `Settings.eleven_exhausted_active()` computes the still-benched subset.
- `mark_eleven_exhausted()` now refreshes the timestamp even for already-listed keys, so a failed probe re-arms the cooldown rather than leaving the key eternally due.
- New module constants `QUIET_PEAK_SILENCE` / `QUIET_PEAK_NORMALIZE` / `QUIET_PEAK_TARGET`; normalization lives at the top of the local path in `WhisperEngine.transcribe()`.
- New `TELLUR_TASK_NAME`, `_schtasks()` (always `CREATE_NO_WINDOW` — pythonw must never flash a console), `_tellur_task_xml()`, `_delete_legacy_run_key()`. Task enabled-state is read from the task XML (`<Enabled>`), which is locale-independent where `schtasks /query`'s status column is not.
- Settings dialog hint for start-on-login rewritten to describe both mechanisms.

---

## [2.3.2] — 2026-07-08

Fixes dictation into the **integrated terminal of Electron editors** (Cursor / VS Code / Windsurf), which had been silently failing since the v2.3.1 revert.

### Why

v2.3.0 tried routing the whole VS Code family to Shift+Insert to make the integrated terminal work; v2.3.1 reverted that because Shift+Insert broke the chat panel — and the editor, chat panel, and terminal all share one opaque top-level window class (`Chrome_WidgetWin_1`), so a per-*window* rule can't tell them apart. That left the terminal back on plain Ctrl+V, which xterm.js (and a raw-mode TUI like Claude Code running in it) doesn't reliably accept — so dictation into the terminal logged "pasted" while nothing appeared.

### What's new

- **UIA-based focused-element detection.** `pick_paste_strategy()` now inspects the *focused control's* class via UI Automation (`get_focused_element_class()`), not just the window. This finally distinguishes the surfaces inside an Electron editor: the integrated terminal focuses an `xterm-helper-textarea`; the chat box and editor do not.
- When focus is on an xterm.js terminal inside a known Electron editor (`cursor`/`code`/`windsurf`/`codium`/`vscodium`), Tellur **types** the transcript as unicode key events — the one injection path that lands in a raw-mode terminal. Everything else in those apps (chat panel included) stays on Ctrl+V, preserving the v2.3.1 behavior that fixed chat paste.
- Bare `cmd`/PowerShell (`ConsoleWindowClass`), mintty, Alacritty/WezTerm, Windows Terminal, and all non-editor apps are unchanged.

### Under the hood

- New `get_focused_element_class()` builds and caches a `CUIAutomation` COM object lazily on first use; a build failure disables it permanently (falls back to the old window/process heuristics), and a transient failure (no focused element) just returns None. Never raises — it's on the per-paste hot path.
- New `XTERM_FOCUS_CLASS_MARKER` (substring match, robust to xterm.js markup changes) and `ELECTRON_EDITOR_PROCS` set. Paste debug line now includes the focused element class (`paste target class=... proc=... el=... → strategy=...`).
- No new dependency — `comtypes` was already required (audio session ducking).

---

## [2.3.1] — 2026-06-10

Hotfix on top of v2.3.0, minutes after testing it live.

- **Reverted the VS Code-family Shift+Insert override.** v2.3.0 routed VS Code / Cursor / Windsurf to clipboard + Shift+Insert (following Wispr Flow's integrated-terminal workaround) — but that chord doesn't land in Cursor's chat panel, which broke the most common dictation target. The VS Code family binds Ctrl+V everywhere that matters on Windows (editor, chat panels, *and* the integrated terminal, where `workbench.action.terminal.paste` is on Ctrl+V by default), so they're back on the plain Ctrl+V path. Shift+Insert remains only for mintty; Ctrl+Shift+V for Alacritty/WezTerm; the console typing path is unchanged.

---

## [2.3.0] — 2026-06-10

Adds **terminal-aware paste** so dictation lands correctly in console windows — classic CMD/PowerShell windows, Claude Code CLI, and IDE integrated terminals.

### Why

Tellur pastes by putting the transcript on the clipboard and simulating **Ctrl+V** — which only works if the focused window translates that chord into a paste. Terminals frequently don't:

- **Classic console windows** (`ConsoleWindowClass` — a bare `cmd.exe` / PowerShell window): when a TUI app like Claude Code holds the console in raw mode, conhost's built-in Ctrl+V *and* Shift+Insert paste shortcuts are disabled outright (both are gated on `ENABLE_PROCESSED_INPUT`, which raw mode clears). No paste chord can work; Tellur would log "pasted" while nothing appeared.
- **Claude Code** specifically has **no Ctrl+V text-paste handler of its own** on Windows — text paste only works when the terminal host converts Ctrl+V into a bracketed paste (Windows Terminal does; conhost in raw mode can't).
- **VS Code-family integrated terminals** (VS Code, Cursor, Windsurf) don't accept a synthetic Ctrl+V reliably; Shift+Insert is the chord dictation tools use there (it's what Wispr Flow ships for the same reason).

### What's new

- New **Settings → Integration & automation → Terminal-aware paste** checkbox (default **on**). Per-paste, Tellur inspects the foreground window's class + process and picks a strategy:
  - **Classic console windows** → the transcript is **typed** as synthetic unicode key events (layout-independent, accepted by conhost in both cooked and raw input modes). Newlines/tabs are flattened to spaces so dictation can never press Enter into a shell or submit a TUI prompt mid-transcript. The clipboard is left completely untouched on this path.
  - **VS Code / Cursor / Windsurf / VSCodium / mintty** → clipboard + **Shift+Insert**.
  - **Alacritty / WezTerm** → clipboard + **Ctrl+Shift+V** (their reserved paste chord).
  - **Windows Terminal and everything else** → clipboard + **Ctrl+V**, unchanged.
- Turning the checkbox off restores the old always-Ctrl+V behavior.
- Re-paste (**Ctrl+Win+B**) and AI-prompt paste (**Ctrl+Win+L**) go through the same strategy selection automatically.

### Under the hood

- New `get_foreground_window_info()` returns `(window_class, process_basename)`; `get_foreground_process_basename()` (per-context vocabulary) now delegates to it.
- New module-level `pick_paste_strategy()` + `PASTE_CHORD_BY_PROCESS` table; `TextInjector` takes the live `Settings` and routes between the chord path and the new console-typing path (`keyboard.write` with a 2 ms per-char delay, `TERMINAL_TYPE_KEY_DELAY_SEC`).
- Paste log lines now include the strategy (`pasted 298 chars (ctrl+v)`, `typed 298 chars into console window`) and a debug line with the detected target (`paste target class=... proc=... → strategy=...`).
- Tip for Claude Code users: Windows Terminal remains the best host — there Ctrl+V arrives as a bracketed paste, which is the input form Claude Code is designed for. The typing path exists so plain-CMD-hosted sessions work at all.

---

## [2.2.2] — 2026-05-21

Adds **start Tellur on Windows login** as an opt-in Settings toggle.

### Start on login

- New **Settings → General → Start Tellur when I log into Windows** checkbox.
- Backed by an `HKCU\Software\Microsoft\Windows\CurrentVersion\Run\Tellur` entry pointing at this install's `run.bat`. Per-user — no admin needed to enable.
- `App.start()` reconciles the registry against the saved setting on every boot, so a moved install or an externally-cleared registry entry re-registers itself.
- Each `Settings.save()` calls `apply_start_on_login()` to keep the registry consistent with what's persisted in `settings.json`.
- Triggers one UAC prompt at login (because `run.bat` self-elevates). Skip by setting `TELLUR_NO_ELEVATE=1` before launch.

### Under the hood

- New module-level helpers: `is_start_on_login_enabled()`, `apply_start_on_login()`, `_start_on_login_command()`. All `winreg`-only, Windows-guarded, fail-soft.
- New `Settings.start_on_login: bool` field; round-trips through `_load` / `save`.

---

## [2.2.1] — 2026-05-20

Two narrowly-scoped additions on top of v2.2.0. An earlier `v2.2.1 → v2.2.3` fix-pack was reverted because part of it correlated with empty-transcribe bugs we couldn't fully isolate; this release re-introduces only the two pieces that are clearly safe and useful.

### Orange flash on empty transcribes

When Whisper returns empty text (too-short press, silent audio, VAD-rejected) the overlay now flashes an orange dot before resetting to idle, instead of silently returning to idle. This makes "we tried, got nothing" visually distinct from "pasted successfully into a window you're not looking at". Pure render-state change — no audio-pipeline side effects.

### `run.bat` self-elevates by default

`run.bat` checks for Administrator privilege at launch and re-spawns itself elevated via PowerShell's `Start-Process -Verb RunAs` if it isn't. Triggers one UAC prompt per launch. This makes Tellur a privilege peer to elevated foreground apps, so paste keystrokes aren't dropped by Windows UIPI when targeting admin terminals / certain games.

Opt out by setting `TELLUR_NO_ELEVATE=1` before running the script — useful if elevation correlates with a problem and you need to bisect.

### Not re-introduced from the reverted commit

These were also in the original 2.2.1–2.2.3 work but stayed out of this release pending further investigation: paste-target elevation mismatch warning, mic-device name re-resolution on every press, LLM "Test connection" exception-scope fix, personalized dictionary, debug logging.

---

## [2.2.0] — 2026-05-18

Three things: **audio ducking** (lower other apps while you talk), **customizable secondary hotkeys**, and the notes default moved off `Ctrl+Win+N`.

### Audio ducking

When you start recording, every other app's volume drops to a low level so background videos / music / Discord don't compete with your voice. Snaps back the moment you release the key.

- New section in **Settings → Lower other apps while recording**. Toggle + slider for the duck level (default 5%, range 0–50%).
- Per-app session ducking via the Windows Core Audio API (pycaw / `ISimpleAudioVolume`). Tellur's own process is excluded so any future Tellur sound isn't ducked under itself.
- Sessions already below the target level are left untouched — we wouldn't be lowering them, only confusing the user by pushing them back up on release.
- Restore runs at every recording boundary that matters: hotkey release, Esc cancel, note end, Refresh UI, and `quit()`. If a recording somehow ends without going through any of those paths, the next press will detect we're already ducked and noop.
- Off by default — opt-in via Settings.

### Customizable secondary hotkeys

Four hotkeys are now user-editable from **Settings → Hotkeys**: **Notes**, **Re-paste last**, **Apply AI prompt**, **Quit Tellur**. Each is a Qt key-sequence field — click it, press your new combo, it saves and rebinds live (no restart). A per-row **Reset** button restores the default.

- `Ctrl+Win` push-to-talk and `Esc` cancel remain fixed — the former uses key polling rather than a registered hotkey (so changing it would be a different and bigger change), and the latter is universally Esc by convention.
- Combo strings are translated between Qt's format (`Ctrl+Meta+Z`) and the `keyboard` library's format (`ctrl+windows+z`) via two small helpers.
- `HotkeyWatcher` now tracks `add_hotkey` handles per slot so a Settings edit cleanly removes the old combo before installing the new one — no leaking bindings across edits.

### Default note hotkey: Ctrl+Win+N → Ctrl+Win+Z

`Ctrl+Win+N` overlapped with users muscle-memory for Windows 11's Notification Center (which uses bare `Win+N`). Moving to `Z` clears the ambiguity. Existing users get the new default automatically since v2.1.0 didn't persist a `hotkey_note` field; if you'd already remapped it, your choice is preserved.

### Under the hood

- New `AudioDucker` class — pycaw-backed, idempotent, thread-safe, with deferred imports so a missing pycaw doesn't break startup (Settings UI greys out the toggle in that case).
- New Settings fields: `duck_enabled`, `duck_level_pct`, `hotkey_quit`, `hotkey_repaste`, `hotkey_llm`, `hotkey_note`. All round-trip through the existing `_load` / `save` flow.
- `HotkeyWatcher.__init__` now accepts a `settings=` argument and exposes `apply_settings()` so `App` can re-bind hotkeys whenever Settings emits `changed`.
- New module-level helpers `keyseq_to_combo()` / `combo_to_keyseq()` for translating between Qt key sequences and the `keyboard` library's combo strings.
- `requirements.txt` gains `pycaw>=20240210; sys_platform == "win32"`. On non-Windows it's not pulled in, and on Windows the missing-pycaw path is handled gracefully (ducking UI is disabled with a hint).

---

## [2.1.0] — 2026-05-18

Two new features: **voice-captured notes** and a **Refresh UI** recovery affordance.

### Notes (Ctrl+Win+N)

A second dictation lane that doesn't paste — it saves the transcription to a separate Notes log for later review.

- **New hotkey: `Ctrl+Win+N`.** Press to start a note recording, press again to stop. Toggle behavior regardless of the global Push-to-talk mode setting — notes are typically longer than dictations, so a hold-style key isn't ergonomic for them.
- **Auto-flagged by foreground app.** The basename of the app that had focus when you started the note (e.g. `code.exe`, `chrome.exe`) is captured into the entry as an `app` field. Lets you find notes from a specific session/context later.
- **Manual flags.** Right-click any note → **Flag as** → **Critical / Important / Followup / Random / None**. Flag renders as a colored chip on the row; the default is no flag.
- **New Notes tab** in the main window, between History and Settings. Filter by flag (combo box) or by free-text search (matches note body or app name). Same row interactions as History — copy, delete-with-confirm, right-click context menu.
- **Notes are never pasted.** This is the whole point: silent capture for later. Auto-paste / LLM auto-apply / markdown daily-file / webhook integrations are all bypassed for notes. They land in `<TELLUR_HOME>\notes.json`.
- Dictation+note ergonomics: pressing `Ctrl+Win+N` while a dictation is in flight cancels the dictation (no phantom paste) and starts the note cleanly. While a note is recording, the Ctrl+Win polling stands down so a still-held modifier doesn't retrigger dictation underneath.

### Refresh UI

A recovery action for when the visible UI gets wedged — overlay vanished, "recording" state stuck, tray icon dropped after an Explorer restart.

- **Tray right-click menu → Refresh UI** — the fastest path when the panel itself isn't reachable.
- **Settings → Troubleshooting → Refresh UI** — same action, from inside the main window.
- What it does: stops the live-level timer, abandons any in-flight recording (audio dropped, mic stream closed), tears down and re-creates the overlay widget, re-shows the tray icon, rebuilds the History + Notes lists. The Whisper model stays loaded, so a refresh is fast — no model re-warmup, no settings reload.

### Under the hood

- New `NotesLog` class — same JSON-on-disk + async saver pattern as `TranscriptLog`, with an extra `set_flag()` method and an `app` field on every entry. Cap: 2000 notes.
- New `NOTES_FILE` constant (`<TELLUR_HOME>\notes.json`) and `NOTE_FLAGS` list.
- New `HotkeyWatcher.note_toggle_requested` signal and `set_note_in_progress()` helper so the Ctrl+Win poll can stand down while a note is being captured.
- New `App.refresh_ui()` method, exposed via `TrayIcon.refresh_ui_requested` and `MainPanel.refresh_ui_requested`.
- New `NoteRow` widget with two chips (app + flag); flag color is themed via QSS `[flag="..."]` attribute selectors so a future theme just needs to provide palette colors.

---

## [2.0.1] — 2026-05-17

UI fix: clicking **Check for updates** (or any button in the lower Settings sections) no longer shifts the entire panel to the right with no way to scroll back.

### What happened

Qt's `QScrollArea` auto-scrolls a freshly-focused widget into view. With the horizontal scrollbar policy set to `ScrollBarAlwaysOff`, the panel would silently scroll horizontally to fit the button — but the scrollbar that would have let you scroll back was suppressed. Net effect: clicking the button cut off the left edge of every form row.

### Fix

- **Horizontal scrollbar policy: `AlwaysOff` → `AsNeeded`.** Now hidden when content fits the viewport (the normal case) and visible when the user shrinks the window below the content's natural width. Either way, no more silent shifting with no way back.
- **Default window size bumped from 640×560 → 680×580.** Gives the Settings content enough breathing room that the focus-follow can't push anything off the edge at default size. The new dimensions are also applied to the first-open positioning fallback in `_show_panel` so re-centering uses the same numbers.

No feature changes. Patched on top of v2.0.0.

---

## [2.0.0] — 2026-05-17

**Phase 8 — the v2.0 finale.** Two quality-of-life features close out the roadmap: a light/dark theme toggle, and per-app vocabulary overlays.

### Light theme

- **Settings → Appearance → Theme** dropdown: pick **Dark** (default) or **Light**. Applies instantly, persists across restarts, restyles the entire main window (list, buttons, combos, sliders, tabs, form labels, hints, section headers — everything).
- Under the hood, both themes are rendered from a single QSS template fed by a `THEME_PALETTES` color dictionary, so adding a new theme is a one-entry change.
- All inline label colors were refactored to `objectName`-driven QSS rules (`#sectionHeader`, `#sectionHint`, `#formLabel`, `#valueDim`, `#footerLabel`, `#dangerButton`) so a single `setStyleSheet` swap covers every label without iterating through the widget tree.
- The LLM "Test connection" success/fail color now reads from the active palette so it stays readable in both themes.

### Per-app vocabulary overlays

- Drop a JSON file into `<TELLUR_HOME>\replacements.d\` named after an app's executable basename (`code.json`, `slack.json`, `discord.json`, `chrome.json`, etc.). Whenever that app is the foreground window, its entries merge into your base `replacements.json` — per-app entries win on conflict, base entries are the fallback.
- The foreground app is captured on the hotkey **press** (via Win32 `GetForegroundWindow` + `QueryFullProcessImageNameW` through ctypes), not on release, so per-context vocab matches what you're actually dictating *into* — even if you tab away while Whisper finishes.
- Same JSON format as the main dictionary; mtime-cached so repeated dictations don't re-read or re-compile the overlay.
- Per-app vocab also drives Whisper's `hotwords` attention bias for that session, so the model is biased toward the right vocabulary during decode — not just patched after the fact.
- **Settings → Per-app vocabulary** gives you:
  - **Open per-app folder** — drops you in `replacements.d\` in Explorer
  - **Show current app** — reports the foreground process's basename so you know exactly what to name the file
  - Inline docs explaining the merge order

### Under the hood

- New `THEME_PALETTES` dict + `build_panel_qss(theme)` function — single source of truth for theme colors.
- New `Replacements.per_context_dir` parameter, internal context-cache (mtime-keyed), and `apply(text, context=...)` signature. The `vocab` property tracks the last context too, so hotwords reflect per-app rules.
- New module-level helper `get_foreground_process_basename()` — returns lowercase executable basename without extension, `None` on any failure. Pure-stdlib (`ctypes`), Windows-only.
- New `PER_CONTEXT_DICT_DIR = <DATA_DIR>/replacements.d`.
- One new setting field: `theme` (`"dark"` | `"light"`, default `"dark"`).

### Roadmap closeout

This ships the last planned milestone. Tellur is now feature-complete against its v1.0 roadmap: push-to-talk dictation, model picker, in-app updates, voice commands, modes & hotkey flexibility, audio control, AI post-processing, integration & automation, privacy & data management, and now theme + per-app vocab.

---

## [1.9.0] — 2026-05-17

**Phase 7 of the roadmap — privacy & data management.** A new "Privacy & data" section in Settings gives you full control over what gets written to disk, how long it sticks around, how to get it out, and how to wipe it.

### Don't-save-history toggle

- **Settings → Privacy & data → "Save transcript history"**. When off, transcripts are NOT written to `history.json`. The in-memory list still tracks recent entries during the session so hotkeys like **Ctrl+Win+B** (re-paste last) and **Ctrl+Win+L** (AI apply) keep working — but nothing persists across restarts.

### Auto-delete after N days

- Slider from **forever** through **365 days**. Purge runs at app startup (so a long downtime doesn't leave very old entries hanging around) and after each new transcription. 0 means "keep forever" (the previous behavior).
- Friendly labels: 1 day, 7 days, 1 month, ~3 months, 1 year, etc.

### Export history (txt / md / json / csv)

- **Export history…** button opens a Save dialog with format picker. Pick:
  - **Markdown** — H2 timestamp blocks, perfect for Obsidian / Logseq
  - **Plain text** — `[YYYY-MM-DD HH:MM:SS] transcript text` one per line
  - **JSON** — exact structure of `history.json`, preserves `raw` and `edited` flags
  - **CSV** — `timestamp_iso, ts, text, raw, edited` columns, opens cleanly in Excel
- Filename defaults to `tellur-history-YYYYMMDD-HHMMSS.<ext>`.

### Import dictionary (team-share vocab)

- **Import dictionary…** button picks a `replacements.json` from disk and **merges** it into your local one. Imported entries win on conflict; existing-but-not-imported entries are kept. Useful for sharing project-specific vocabulary between machines or teammates.
- Reports: "Added N new, overwrote M, total now X" so you can see exactly what changed.

### Clear all data

- **Clear all data…** button wipes `history.json` and `replacements.json` after a Yes/No confirmation. Your `settings.json` is intentionally preserved — your preferences survive the data wipe.

### Open data folder

- **Open data folder** button opens `%TELLUR_HOME%` in Explorer so you can poke around manually (logs, hf-cache, history, etc.).

### Under the hood

- New `TranscriptLog.purge_older_than(seconds)` method drops entries older than the cutoff, emits `cleared` on changes for listener refresh.
- New module-level `export_history(entries, path, fmt)` helper covers all four output formats.
- Two new settings fields: `save_history` (bool, default True), `history_retention_days` (int, default 0 = keep forever).
- LLM apply and normal dictation now both gate `log.add(...)` behind `save_history` so the privacy toggle is honored end-to-end.

---

## [1.8.0] — 2026-05-17

**Phase 6 of the roadmap — integration & automation.** Tellur becomes a hub: every transcript can be persisted to a markdown daybook, POSTed to a webhook, and/or auto-submitted in chat apps. All three are opt-in, all three are best-effort (failures log and move on without disturbing dictation).

### Send & Enter mode

- **Settings → Integration & automation → "Press Enter after paste (chat-app mode)"**. When enabled, Tellur synthesizes an Enter keystroke right after pasting your transcript — perfect for Slack / Discord / Teams / iMessage where messages need a final Enter to actually send.
- Skip it in code editors and docs where Enter would insert an unwanted newline.
- Honored by all three paste paths: normal dictation, Ctrl+Win+B re-paste-last, and Ctrl+Win+L AI-apply.

### Daily markdown transcripts

- **Settings → Integration & automation → "Save each transcript to a daily markdown file"**. Each transcript appended to `YYYY-MM-DD.md` under a configurable folder (defaults to `<TELLUR_HOME>\transcripts`). One file per local-time day.
- Each day's file is created with a `# Tellur transcripts — YYYY-MM-DD` H1; each transcript is recorded as `## HH:MM:SS` followed by the body. Clean enough to read directly, structured enough to grep / parse / sync to Obsidian / Logseq.
- An **Open** button next to the folder field opens it in Explorer (Windows).

### Webhook target

- **Settings → Integration & automation → "POST every transcript to a webhook URL"**. Tellur POSTs each accepted transcript to your URL using a configurable body template.
- **Placeholders**: `{text}` (final transcript), `{raw}` (Whisper's raw output), `{ts}` (Unix timestamp). All string placeholders are JSON-escaped before substitution, so quotes/newlines in the transcript don't break a JSON template.
- **Content-Type auto-detection**: if the rendered body parses as JSON, sent as `application/json`; otherwise as `text/plain`. The default template is JSON.
- **Best-effort**: failures (network down, 5xx response, etc.) are logged and swallowed. The webhook never blocks dictation, and runs on a worker thread.

### Under the hood

- New `MarkdownDailySink` and `WebhookSink` classes encapsulate the I/O for each destination so each transcript can be fanned out to N sinks without bloating `_process_audio`.
- New `App._dispatch_integrations()` snapshots the relevant settings up front and runs both sinks on a single worker thread, so a slow webhook doesn't delay the markdown write or hold up the next dictation.
- `TextInjector.paste()` gained a `press_enter=True` parameter that synthesizes the Enter keystroke after a short delay so single-line chat inputs don't race the paste.
- Six new settings fields: `send_after_paste`, `markdown_save_enabled`, `markdown_folder`, `webhook_enabled`, `webhook_url`, `webhook_template`.

---

## [1.7.0] — 2026-05-17

**Phase 5 of the roadmap — AI post-processing.** The headline feature: optionally route your transcripts through a local (or remote) OpenAI-compatible LLM to clean them up, rewrite as bullets/email/Slack, or summarize. Built for local-first stacks (LM Studio, Ollama, llama.cpp server, vLLM) so your audio AND your text both stay on your machine.

### Built-in prompts

Toggleable from **Settings → AI post-processing → Default prompt**:

- **Clean up (remove filler words)** — light cleanup, preserves voice and meaning. Filler words like "um", "uh", "like", "you know" get dropped; dictation errors get fixed.
- **Make it a clean paragraph** — collapses run-on speech into one well-punctuated paragraph.
- **Make it formal** — rewrites in professional written English (email/report tone).
- **Convert to bullet points** — restructures content as a markdown bulleted list.
- **Convert to email** — formats as an email body in a polite professional tone.
- **Convert to Slack message** — makes it conversational and concise.
- **Summarize** — produces a 2–4-sentence summary capturing key points.

All prompts are tuned to output ONLY the result with no preamble, commentary, or quote wrapping.

### Two ways to apply

1. **Hotkey: Ctrl+Win+L** — applies the configured default prompt to your most recent transcript and pastes the result into the focused window. The original raw transcript is kept; the LLM result is added as a new history entry. The overlay shows the blue "thinking" pulse while the LLM works.
2. **Auto-apply checkbox** — when enabled, every transcript gets routed through the default prompt before paste. The history entry stores both: `raw` keeps Whisper's original output, `text` becomes the LLM-cleaned version.

### Endpoint configuration

- **Endpoint URL** — defaults to `http://localhost:1234/v1` (LM Studio's default). Override for Ollama (`http://localhost:11434/v1`), llama.cpp server, vLLM, or any OpenAI-compatible service.
- **Model** — free-text model name as your endpoint exposes it. LM Studio often accepts empty here; Ollama wants the model name (e.g. `llama3.1`).
- **API key** — masked password field, optional. Local endpoints almost always leave this blank.
- **Test connection** button — pings `/models` then issues a tiny chat completion so 401s, missing-model errors, and unreachable endpoints all surface clearly before you start dictating.

### Under the hood

- New `LLMClient` class implements OpenAI-compatible chat completions over `urllib.request` (stdlib only — no new dependencies). Synchronous, but all callers wrap in worker threads so the Qt event loop never blocks on a slow LLM.
- New `BUILTIN_LLM_PROMPTS` registry and `llm_prompt_by_id()` helper.
- New `HotkeyWatcher.llm_apply_requested` signal driven by the global `ctrl+win+l` hook.
- New settings fields: `llm_enabled`, `llm_base_url`, `llm_model`, `llm_api_key`, `llm_default_prompt`, `llm_auto_apply`.
- Both the auto-apply path and the Ctrl+Win+L hotkey reuse the same client instantiation, so endpoint changes apply immediately to both flows.

### Privacy guarantee

If your endpoint URL is `localhost` (the default), nothing leaves your machine — audio is transcribed locally by Whisper, then the text is sent to your local LLM, then the result is pasted. The Tellur tradition of "no network calls during normal use" is preserved.

### Hotkey rebind: Ctrl+Win+V → Ctrl+Win+B

- Re-paste-last has moved to **Ctrl+Win+B** ("B" for "bring back"). Windows 11 reserves Ctrl+Win+V for its audio-output device picker, which was hijacking Tellur's binding. README, in-app hotkey table, and footer all updated.

### Settings & main-window UI polish

- **Settings tab is now scrollable** — wrapped in a `QScrollArea` so a long settings list no longer forces the entire window taller than your screen. Vertical scrollbar only, never horizontal.
- **Sections redesigned** — bold headers with a thin underline, consistent 10px in-section spacing, 20px between sections. No more text clumping or controls touching.
- **Form-row alignment** — every label/control pair (Microphone, Endpoint, Model, API key, Default prompt, Input gain, Meter sensitivity) uses a fixed-width label so all controls align in a tidy column.
- **Wheel-event hijack fixed** — scrolling through Settings no longer accidentally changes combo selections or slider values when the cursor passes over them. `_NoWheelComboBox` and `_NoWheelSlider` subclasses `ignore()` the wheel so it bubbles up to the scroll area.
- **Window opens centered** — first open per session positions the panel horizontally centered, ~8% above vertical center (eye-level rather than dead-center). Drag-to-relocate persists within the session. Height is capped to the work-area so the bottom never clips behind the taskbar.

---

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

- **Ctrl+Win+B → paste the most recent transcript into whatever window is focused right now.** Identical to right-click tray → "Copy last transcription" + Ctrl+V, but as a single global hotkey.
- Made for the "Tellur auto-pasted into the wrong window" recovery flow and "I want to say that same thing again in a different app" use case.

### Hotkey reference in Settings

Settings tab now includes a clear table of every global hotkey Tellur listens for, so you don't have to remember them or check the README.

| Hotkey | Action |
|---|---|
| **Ctrl+Win** | Push-to-talk (hold or toggle, per setting) |
| **Esc** | Cancel current recording |
| **Ctrl+Win+B** | Re-paste last transcription |
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
