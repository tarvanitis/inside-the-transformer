#!/usr/bin/env python3
"""Project policy wrapper around the prose-linter scanner.

SINGLE SOURCE OF TRUTH: every prose rule (B9 em-dash budget, B18 clause-join
semicolons, B17b spelling dialect, list spacing, hyphen splits, word tells) is
implemented in the prose-linter skill's scripts/scan.py and its word list in
references/dialect-wordlist.txt, published at:

    https://github.com/tarvanitis/ai-prose-tools

Do not reimplement a check here. If a rule needs changing, change it in the skill
so every project that uses the skill gets the same behaviour. This file only
supplies project policy: which files, what budget, and the exemption list.

The scanner is an optional development dependency and is NOT bundled with this
repository. Without it the gate reports SKIPPED and the build continues, so that
cloning and building the book does not require installing a skill. See
CONTRIBUTING.md.
"""
import pathlib, subprocess, sys

HERE = pathlib.Path(__file__).parent
ALLOW = HERE / "prose-allow.txt"
EMDASH_PER = 500          # B9: one em-dash per N words

# Checked in order. The first that exists wins.
CANDIDATES = [
    HERE / "prose-linter" / "scripts" / "scan.py",                    # vendored
    pathlib.Path.home() / ".claude/skills/prose-linter/scripts/scan.py",
    # Legacy: prose-linter was called humanize-prose before it was published.
    pathlib.Path.home() / ".claude/skills/humanize-prose/scripts/scan.py",
]


def find_scanner():
    return next((p for p in CANDIDATES if p.exists()), None)


def main(files):
    scan = find_scanner()
    if scan is None:
        print("PROSE GATE SKIPPED: the prose-linter scanner is not installed.")
        print("  Looked in:")
        for c in CANDIDATES:
            print(f"    {c}")
        print("  Install prose-linter to enable the em-dash budget, clause-join")
        print("  semicolon, and spelling-dialect checks:")
        print("    git clone https://github.com/tarvanitis/ai-prose-tools")
        print("    ln -s $PWD/ai-prose-tools/prose-linter ~/.claude/skills/prose-linter")
        print("  The build continues without it. See CONTRIBUTING.md.")
        return 0
    cmd = [sys.executable, str(scan), *files, "--gate", "--quiet",
           "--emdash-per", str(EMDASH_PER), "--min-count", "2"]
    if ALLOW.exists():
        cmd += ["--allow", str(ALLOW)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = [l for l in r.stdout.split("\n") if l.startswith(("GATE", "FAIL")) or "[" in l]
    print("\n".join(tail) if r.returncode else
          f"prose gate OK across {len(files)} files (via prose-linter scan.py)")
    return r.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
