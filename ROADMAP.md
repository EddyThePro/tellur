# Tellur roadmap

Planned phases for Tellur's continued development. Each version is a contained themed release. We ship them sequentially.

> **Out of scope across all phases:** cloud-required features, paid third-party APIs by default (LLM integration in v1.7 is user-provided endpoint), mobile, multi-user/team, speech synthesis.

---

## v1.3 — History UX overhaul

Make the **History** tab actually useful day-to-day. Every row gets inline actions; bottom-bar friction goes away.

- Per-row buttons: **Copy** · **Pin (★)** · **Delete (×)**
- Right-click context menu: Copy / Copy with timestamp / Copy as markdown quote / Pin-Unpin / Edit text / Delete
- **Inline edit** — double-click a row's text to fix typos before copying
- **Search/filter box** — instant case-insensitive text search
- **Pinned section** — starred entries float to the top and survive the 500-entry cap
- **Word + char count** per row
- **Keyboard shortcuts**: Enter = copy · Delete = delete · Ctrl+P = pin · Ctrl+F = focus search · Space = preview detail

## v1.4 — Voice commands & smart text

Less manual editing after dictation. The "feels professional" jump.

- **Spoken punctuation:** comma, period, question mark, exclamation point, colon, semicolon, open/close parenthesis, open/close quote, ellipsis, dash
- **Structural:** "new line" · "new paragraph" · "tab" · "indent"
- **Editing:** "scratch that" / "delete that" — drops the last sentence of the current dictation
- **Capitalization:** "all caps X" · "capitalize next" · "lowercase X"
- **Smart formatting** (toggle each in Settings):
  - Auto-capitalize sentence starts
  - Spelled numbers → digits (`twenty five` → `25`)
  - Spelled dates → formatted (`October fifth` → `October 5`)
  - URLs (`github dot com slash foo` → `github.com/foo`)
- **Per-dictation override:** "spell it" to force literal mode for a phrase

## v1.5 — Modes & hotkey flexibility

Different work patterns deserve different hotkeys.

- **Customizable hotkey** with a record-a-keystroke picker in Settings
- **Toggle mode** alternative to hold mode (tap once to start, tap again to stop)
- **Multiple hotkeys for different modes:**
  - Default — current dictionary + behaviors
  - Code — different dictionary, smart code formatting, auto-capitalize off
  - Email — auto-capitalize on, auto-punctuate
- **`Re-paste last`** hotkey
- **`Re-transcribe last clip`** hotkey
- **`Cancel current recording`** hotkey (Esc by default)
- **Mouse-button as hotkey** support

## v1.6 — Audio control

Get the mic right.

- **Mic device selection** in Settings
- **Live input level meter** in Settings
- **Per-device profiles** — laptop vs headset
- **Adjustable input gain** slider
- **VAD silence-threshold** slider
- **Mic test recording** — record/playback/transcript preview
- **Optional: save audio clips** to disk with retention
- **First-launch mic check** wizard

## v1.7 — AI post-processing ⭐ *the killer feature*

Speech-to-prose, not just speech-to-text. The biggest competitive differentiator vs. paid alternatives.

- **Optional integration with a local LLM** (Ollama / LM Studio / any OpenAI-compatible endpoint — no cloud requirement)
- **Built-in prompts (toggleable):**
  - Clean up filler words (um, like, you know)
  - Make it a clean paragraph
  - Make it formal
  - Convert to bullet points
  - Convert to email
  - Convert to Slack message
  - Summarize
- **Custom user prompts**
- **Per-app default prompt**
- **Hotkey to apply post-processing to last transcript**
- **Streaming output** to the overlay as the LLM types

## v1.8 — Integration & automation

Tellur becomes a hub, not an island.

- **Save transcripts as markdown** (one per day, configurable folder)
- **Webhook target** — POST every transcript to a URL with a template
- **"Send & enter"** mode for chat apps (auto-press Enter after paste)
- **Per-app paste configuration** — different paste rules per target window class
- **"New email" voice command** — opens default mail with transcript as body
- **Templates** — `to:` / `subject:` auto-formatting
- **Shell command via voice** with confirmation dialog
- **One-click push to GitHub gist**

## v1.9 — Privacy & data management

For users who care about retention and portability.

- **"Don't save history"** toggle
- **Auto-delete history older than N days**
- **Encrypted history** option with password
- **Export history:** txt / markdown / json / csv
- **Import dictionary** from file (team-share vocab)
- **"Clear all data"** button
- **Local audit log** of every paste destination
- **Optional encrypted cloud sync** via rclone / WebDAV / S3 (user-provided storage)

## v2.0 — Power user / extensibility

Bigger architectural moves.

- **Per-context vocabularies** — different `replacements.json` per active window
- **Plugin system** — Python plugins for custom voice commands (spotify-control, browser-tab-switch, vscode-snippet-insert, etc.)
- **Local HTTP API** at `http://localhost:7842/transcribe` — external tools can trigger dictations
- **Light theme** alternative
- **Streaming preview** while talking
- **Wake-word mode** — "Hey Tellur, …" with no hotkey
- **Real-time language switching** via Whisper multilingual
- **LSP integration** — IDE-aware vocabulary

---

## Status

| Version | Theme | Status |
|---|---|---|
| 1.0.0 | Initial release | ✅ shipped |
| 1.0.1 | Transcription bug fix | ✅ shipped |
| 1.1.0 | In-app auto-update | ✅ shipped |
| 1.2.0 | Model picker | ✅ shipped |
| 1.2.1 | Branded windows + taskbar | ✅ shipped |
| 1.2.2 | Stability audit pass | ✅ shipped |
| 1.3.0 | History UX overhaul | ✅ shipped |
| 1.3.1 | History UI polish | ✅ shipped |
| 1.4.0 | Voice commands & smart text | ✅ shipped |
| **1.5.0** | **Modes & hotkey flexibility** | ✅ shipped |
| 1.6 | Audio control | 🚧 next |
| 1.7 | AI post-processing | planned |
| 1.8 | Integration & automation | planned |
| 1.9 | Privacy & data management | planned |
| 2.0 | Power user / extensibility | planned |
