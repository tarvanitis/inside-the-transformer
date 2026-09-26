#!/usr/bin/env python3
"""Structural-integrity checker for a book-length Markdown manuscript.

Catches what breaks when content moves. Figure and table numbers, chapter
cross-references and dated claims are hard-coded literals in Markdown. Nothing
renumbers them automatically, so inserting a table into chapter 12 silently
leaves the rest of that chapter mis-numbered and every reference to it stale.

Two classes of finding:

  FAIL  Objectively broken. A reference points at a chapter that does not
        exist, numbering skips, a figure has no caption. Exits non-zero, so it
        gates a build.

  WARN  Inconsistent with the manuscript's own dominant convention. The tool
        infers conventions from your book rather than imposing its own, so
        these are advisory. Use --strict to make them fail too.

Configuration is optional. Drop a `book-audit.toml` beside the manuscript:

    chapter_pattern = "ch-(\\d+)-"   # how chapter numbers appear in filenames
    stale_days      = 90             # age at which an "as of" claim is flagged
    allow           = "allow.txt"    # files exempt from caption-count checks
    xref_styles     = ["Chapter N"]  # cross-reference forms you use on purpose

Declaring `xref_styles` replaces the tool's guess with a fact. Styles you list
are accepted; anything else still warns, so genuine drift is still caught. Leave
it out and the dominant style is inferred instead.

Anything absent is inferred or defaulted.
"""
import pathlib, re, sys, datetime, collections

DEFAULTS = {"stale_days": 90, "chapter_pattern": None, "allow": None,
            "xref_styles": None}

# Headings every chapter is supposed to repeat. Flagging these as duplicated
# content would fire on every well-structured book.
STRUCTURAL_HEADINGS = {
    "summary", "introduction", "conclusion", "overview", "recap", "in this chapter",
    "what you'll learn", "what you will learn", "key takeaways", "takeaways",
    "exercises", "further reading", "notes", "next steps", "the big picture",
    "what we covered", "where we are", "prerequisites", "references",
}

# Filename conventions tried when chapter_pattern is not configured, best match wins.
CHAPTER_PATTERNS = [
    r'ch-(\d+)-', r'chapter-(\d+)', r'ch(\d+)', r'^(\d+)[-_]', r'-(\d+)-',
]

# Cross-reference spellings. The dominant one in your manuscript becomes the norm.
XREF_STYLES = {
    "Chapter N": re.compile(r'\bChapter (\d+)\b'),
    "Ch. N":     re.compile(r'\bCh\. (\d+)\b'),
    "Ch N":      re.compile(r'\bCh (\d+)\b'),
    "§N":        re.compile(r'§\s?(\d+)\b'),
}


def load_config(paths):
    cfg = dict(DEFAULTS)
    # Deterministic order: the directories holding the manuscript first, in the
    # order given, then the working directory. This was a set, so which config
    # won was decided by hash order, which is not stable between runs. A gate
    # must not depend on that.
    roots = []
    for p in paths:
        d = pathlib.Path(p).resolve().parent
        if d not in roots:
            roots.append(d)
    cwd = pathlib.Path.cwd()
    if cwd not in roots:
        roots.append(cwd)
    for root in roots:
        for name in ("book-audit.toml", ".book-audit.toml"):
            f = root / name
            if f.exists():
                try:
                    import tomllib
                    cfg.update(tomllib.loads(f.read_text()))
                except Exception as e:
                    print(f"warning: could not read {f}: {e}", file=sys.stderr)
                return cfg, f
    return cfg, None


def infer_chapter_pattern(paths):
    """Pick the filename pattern that identifies the most chapters."""
    best, best_n = None, 0
    for pat in CHAPTER_PATTERNS:
        n = sum(1 for p in paths if re.search(pat, pathlib.Path(p).name))
        if n > best_n:
            best, best_n = pat, n
    return best, best_n


def body_lines(path):
    """Yield (lineno, text) for lines outside fenced code blocks."""
    infence = False
    for i, ln in enumerate(pathlib.Path(path).read_text().split("\n"), 1):
        if ln.startswith("```"):
            infence = not infence
            continue
        if not infence:
            yield i, ln


def load_allow(cfg, cfgpath):
    names = set()
    candidates = []
    if cfg.get("allow"):
        base = cfgpath.parent if cfgpath else pathlib.Path.cwd()
        candidates.append(base / cfg["allow"])
    candidates.append(pathlib.Path(__file__).parent / "structure-allow.txt")
    for f in candidates:
        if f.exists():
            for ln in f.read_text().split("\n"):
                ln = ln.split("#")[0].strip()
                if ln:
                    names.add(ln)
            break
    return names


def main(argv):
    strict = "--strict" in argv
    paths = [a for a in argv if not a.startswith("-")]
    if not paths:
        print("usage: check-structure.py [--strict] FILE.md [FILE.md ...]")
        return 2

    cfg, cfgpath = load_config(paths)
    skip = load_allow(cfg, cfgpath)
    stale_days = int(cfg.get("stale_days") or 90)

    pattern = cfg.get("chapter_pattern")
    inferred = False
    if not pattern:
        pattern, n = infer_chapter_pattern(paths)
        inferred = True
        if not pattern:
            pattern = r'ch-(\d+)-'

    def chapter_no(path):
        name = pathlib.Path(path).name
        m = re.search(pattern, name)
        if m:
            return str(int(m.group(1)))
        low = name.lower()
        m = re.search(r'appendix-([a-z])', low)
        if m:
            return m.group(1).upper()
        return None

    fails, warns, unresolved = [], [], []
    captions = {}                       # "Figure 5.1" -> (path, line)
    chapters = set()                    # chapter numbers that actually exist
    xref_counts = collections.Counter()
    xref_where = collections.defaultdict(collections.Counter)
    caption_forms = collections.Counter()
    heading_index = collections.defaultdict(list)

    for p in paths:
        c = chapter_no(p)
        if c:
            chapters.add(c)

    # A reference above the highest chapter present is ambiguous: either you were
    # handed a subset of the manuscript, or it is a typo. A reference to a gap
    # *inside* the range is unambiguously broken, because that chapter should be here.
    numeric = sorted(int(c) for c in chapters if c.isdigit())
    max_ch = numeric[-1] if numeric else 0

    # ---- per-file structure ----
    for p in paths:
        ch = chapter_no(p)
        seen = {"Figure": [], "Table": []}
        n_mermaid = n_tables = n_figblocks = n_bare_images = 0
        in_figblock = False
        infence = False
        prev_level = 0

        for i, ln in enumerate(pathlib.Path(p).read_text().split("\n"), 1):
            if ln.startswith("```mermaid"):
                n_mermaid += 1
            if ln.startswith("```"):
                infence = not infence
                continue
            if infence:
                continue

            stripped = ln.strip()
            if stripped.startswith("::: {.figure}"):
                n_figblocks += 1
                in_figblock = True
            elif stripped == ":::":
                in_figblock = False
            elif re.match(r'^!\[[^\]]*\]\(', stripped) and not in_figblock:
                n_bare_images += 1

            if re.match(r'^\|[\s:\-|]+\|\s*$', ln):
                n_tables += 1

            h = re.match(r'^(#{1,6})\s+(.*)', ln)
            if h:
                level, text = len(h.group(1)), h.group(2).strip()
                if prev_level and level > prev_level + 1:
                    warns.append(f"{p}:{i}  heading jumps from H{prev_level} to H{level}")
                prev_level = level
                if level == 2 and text.lower().strip() not in STRUCTURAL_HEADINGS:
                    heading_index[text.lower()].append(p)

            m = re.match(r'^\*\*(Figure|Table) ([A-Z\d]+)\.(\d+)\.?\*\*', stripped)
            if m:
                kind, c, k = m.group(1), m.group(2), int(m.group(3))
                seen[kind].append((c, k, i))
                captions[f"{kind} {c}.{k}"] = (p, i)
                caption_forms["**Kind N.k.**"] += 1
                if ch and c != ch:
                    fails.append(f"{p}:{i}  {kind} {c}.{k} is in chapter {ch}")
            elif re.match(r'^(Figure|Table) ([A-Z\d]+)[.:]', stripped):
                caption_forms["unbolded"] += 1

        for kind, got in seen.items():
            nums = [k for _, k, _ in got]
            if nums and nums != list(range(1, len(nums) + 1)):
                fails.append(f"{p}  {kind} numbering not sequential: {nums}")

        n_figs = n_mermaid + n_figblocks + n_bare_images
        if n_figs != len(seen["Figure"]) and pathlib.Path(p).name not in skip and p not in skip:
            fails.append(f"{p}  {n_figs} figure block(s) but {len(seen['Figure'])} Figure caption(s)")
        if n_tables != len(seen["Table"]) and pathlib.Path(p).name not in skip and p not in skip:
            fails.append(f"{p}  {n_tables} table(s) but {len(seen['Table'])} Table caption(s)")

    # ---- cross-references resolve, and use one style ----
    for p in paths:
        for i, ln in body_lines(p):
            for style, rx in XREF_STYLES.items():
                for m in rx.finditer(ln):
                    xref_counts[style] += 1
                    xref_where[style][p] += 1
                    if style != "§N" and chapters and m.group(1).lstrip("0") not in chapters:
                        n = int(m.group(1))
                        if n <= max_ch:
                            fails.append(f"{p}:{i}  reference to Chapter {n}, which is missing from the range present")
                        else:
                            unresolved.append(f"{p}:{i}  reference to Chapter {n} (above the highest chapter given)")
            for m in re.finditer(r'\b(Figure|Table) ([A-Z\d]+\.\d+)', ln):
                key = f"{m.group(1)} {m.group(2)}"
                if key not in captions and not ln.strip().startswith(f"**{key}"):
                    fails.append(f"{p}:{i}  reference to {key}, which has no caption")

    def _where(style, n):
        top, topn = xref_where[style].most_common(1)[0]
        return f"mostly {top}" if topn > n * 0.6 else f"across {len(xref_where[style])} files"

    declared = cfg.get("xref_styles")
    if declared:
        unknown = [d for d in declared if d not in XREF_STYLES]
        if unknown:
            warns.append(f"book-audit.toml declares unrecognised xref_styles {unknown}; "
                         f"known styles are {sorted(XREF_STYLES)}")
        stray = [(st, n) for st, n in xref_counts.most_common() if st not in declared]
        if stray:
            bits = [f"{st} x{n} ({_where(st, n)})" for st, n in stray]
            warns.append("cross-reference styles outside the declared set "
                         f"{declared}: " + "; ".join(bits))
    elif len(xref_counts) > 1:
        (dom, dn), *rest = xref_counts.most_common()
        bits = [f"{st} x{n} ({_where(st, n)})" for st, n in rest]
        warns.append(f'mixed cross-reference styles: dominant is "{dom}" x{dn}; also ' + "; ".join(bits))
    if len(caption_forms) > 1:
        (dom, dn), *rest = caption_forms.most_common()
        minority = ", ".join(f"{s} x{n}" for s, n in rest)
        warns.append(f"mixed caption formats: dominant is {dom} (x{dn}); also {minority}")

    # ---- a heading repeated across many chapters suggests a foundation taught twice ----
    for text, where in heading_index.items():
        files = sorted(set(where))
        if len(files) >= 3:
            warns.append(f'section "{text}" appears in {len(files)} chapters, check it is not taught more than once')

    # ---- dated claims ----
    today = datetime.date.today()
    for p in paths:
        for i, ln in body_lines(p):
            for m in re.finditer(r'as of (\d{4})-(\d{2})-(\d{2})', ln):
                age = (today - datetime.date(*map(int, m.groups()))).days
                if age > stale_days:
                    fails.append(f"{p}:{i}  dated claim is {age} days old, re-verify against a primary source")

    # ---- bibliography hard breaks ----
    for p in paths:
        if 'bibliograph' not in pathlib.Path(p).name.lower():
            continue
        for i, ln in enumerate(pathlib.Path(p).read_text().split("\n"), 1):
            if re.match(r'^\*\*[^*]+\*\*$', ln.rstrip()) and not ln.endswith("  "):
                fails.append(f"{p}:{i}  bibliography title lacks a hard line break (two trailing spaces)")

    # ---- report ----
    if cfgpath:
        print(f"(config: {cfgpath})")
    if inferred and chapters:
        print(f"(inferred chapter pattern {pattern!r}, found {len(chapters)} chapters)")
    if unresolved:
        tgt = sorted({int(u.rsplit("Chapter ", 1)[1].split()[0]) for u in unresolved})
        print(f"note: {len(unresolved)} reference(s) point above Chapter {max_ch}, the highest given "
              f"(to {', '.join(map(str, tgt[:8]))}{'...' if len(tgt) > 8 else ''}).")
        print("      Not treated as failures. Run against the whole manuscript to check these.")
    if warns and not fails:
        print("STRUCTURE WARNINGS")
        for w in warns[:25]:
            print("   ", w)
        if len(warns) > 25:
            print(f"    ... and {len(warns)-25} more")
    if fails:
        print("STRUCTURE FAIL")
        for f in fails[:25]:
            print("   ", f)
        if len(fails) > 25:
            print(f"    ... and {len(fails)-25} more")
        for w in warns[:10]:
            print("    (warn)", w)
        return 1
    if strict and warns:
        return 1
    if not warns:
        print(f"structure gate OK ({len(captions)} captions, cross-refs resolve, dates fresh)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
