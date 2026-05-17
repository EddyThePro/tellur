"""Add or remove a replacement entry in replacements.json.

  teach <wrong> <right>     # add or update
  teach --remove <wrong>    # delete an entry
  teach --list              # print current dictionary

Tellur auto-picks up changes on the next transcription — no restart needed.
"""

import json
import sys
from pathlib import Path

REPL = Path(__file__).resolve().parent / "replacements.json"


def load() -> dict[str, str]:
    if not REPL.exists():
        return {}
    try:
        return json.loads(REPL.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"replacements.json is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)


def save(d: dict[str, str]) -> None:
    # Write atomically via temp + rename so a power loss or kill mid-write
    # can't corrupt replacements.json.
    tmp = REPL.with_suffix(REPL.suffix + ".tmp")
    tmp.write_text(
        json.dumps(d, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    tmp.replace(REPL)


def main() -> int:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    data = load()

    if args[0] == "--list":
        if not data:
            print("(empty)")
            return 0
        width = max(len(k) for k in data) + 2
        for k, v in sorted(data.items()):
            print(f"  {k.ljust(width)} -> {v}")
        print(f"\n{len(data)} entries in {REPL}")
        return 0

    if args[0] == "--remove":
        if len(args) != 2:
            print("usage: teach --remove <wrong>", file=sys.stderr)
            return 1
        key = args[1]
        if key not in data:
            print(f"not found: {key!r}", file=sys.stderr)
            return 1
        del data[key]
        save(data)
        print(f"removed: {key!r}  ({len(data)} entries remain)")
        return 0

    if len(args) != 2:
        print("usage: teach <wrong> <right>   (use quotes for phrases)", file=sys.stderr)
        return 1

    wrong, right = args[0], args[1]
    existed = wrong in data
    data[wrong] = right
    save(data)
    verb = "updated" if existed else "added"
    print(f"{verb}: {wrong!r} -> {right!r}  ({len(data)} entries total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
