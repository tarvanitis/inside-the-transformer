# Maintenance & Freshness

*Inside the Transformer* covers a fast-moving field. The mechanics (attention, embeddings, the
transformer block, the training objectives) are durable; the **named models, sizes, dates, and
tooling versions are perishable**. This file catalogues every perishable spot and gives the refresh
procedure. It is the "where staleness lives + how to verify it" playbook — run it on demand, or when
a durable trigger (see bottom) reminds you.

A skill file is inert: it does not watch vendor pages or run on its own. It only tells whoever
invokes it exactly what to check and how.

## Freshness catalogue (what goes stale, and how fast)

| Location | What churns | Cadence | How to verify |
|---|---|---|---|
| `> **Current model families (as of YYYY-MM-DD):**` callout blocks | Flagship names/versions: GPT-5.x, Claude (Fable/Opus/Sonnet/Haiku), Gemini 3.x, Grok, DeepSeek, Qwen, Llama, Mistral, Gemma, Kimi | **High** (weeks) | WebFetch each vendor's model/pricing page; confirm the current flagship + tier names; re-date the block |
| Appendix B — Modern LLM Specifications | Context-window sizes, parameter counts, MoE active/total params | **High** | Vendor model cards / docs; cross-check numbers, don't trust memory |
| Appendix C — Evolution 2017→YYYY | The timeline's last rows | **Yearly** | Add the year's notable shifts; keep older rows as history |
| Applications chapters (18–20) tooling | Ollama, Continue, Open WebUI, LangChain/LangGraph, PGVector versions; the sample hardware (Dell Latitude 7480); any "Guide Version / date" remnant | **High** | Project release pages / changelogs; update version numbers and note tested-on dates |
| Architecture "current trends" (Ch 9) | MoE-as-default, hybrid Mamba–transformer, FlashAttention-N, MLA, MCP — which are now standard vs. superseded | **Medium** (quarters) | Recent survey/vendor posts; adjust "current/emerging/standard" framing |
| Bibliography / Further Reading | Link rot; new seminal papers | **Medium** | Check links resolve; add landmark papers since last pass |
| Any `as of`, `latest`, `current`, `state-of-the-art`, `today` phrasing | Silent staleness | — | Grep for these; each is a claim with an implicit expiry |
| Worked examples pinned to a specific model (e.g. Llama 3 8B shapes `ℝ^{128000×4096}`) | The numbers can drift if the reference model is bumped | **Low** | If you re-peg to a newer model, re-verify the arithmetic in Python |

Fast grep to locate hotspots across the manuscript:

```bash
grep -rniE 'as of|current model families|guide version|latest|state-of-the-art|\b20[0-9]{2}\b' .
```

## Refresh procedure

1. Grep the canonical tree for the hotspots above; build a worklist of every dated/volatile claim.
2. For each, **web-verify from the vendor's own page** (WebFetch/WebSearch) — never from training
   data or memory (`CLAUDE.md`). Record source + date checked.
3. Update names, versions, sizes, and the `as of YYYY-MM-DD` dates. Preserve the "durable shape vs.
   dated snapshot" framing so the book keeps ageing gracefully — don't turn it into a changelog.
4. Re-verify (in Python) any worked example whose numbers you changed.
5. Normalise any LaTeX to plain Unicode; re-validate any touched Mermaid diagram (v11).
6. **Report a diff of proposed changes for the author to approve — do not silent-edit.** Only after
   approval, apply and re-run the production build (`references/production.md`).
7. Run the prose-linter scan on changed chapters.

Output of a refresh run is a short report: what was checked, what's current, what changed, and the
proposed edits — plus anything genuinely new worth a paragraph (a new model class, a new technique).

## Durable trigger (the part CronCreate can't do)

The in-session scheduler (`CronCreate` / `/loop`) is **session-only and expires in 7 days**, so it
cannot deliver a persistent monthly check across sessions. For a trigger that survives, use one of:

- **A `SessionStart` hook** (in `.claude/settings.json`, set up via the `update-config` skill): on
  opening a session in this project, a small script reads the newest `as of YYYY-MM-DD` date in the
  canonical tree and, if it's older than a threshold (e.g. 30 days), prints a reminder to run this
  refresh. Durable, no external dependencies; it *nudges*, the skill does the actual web-verified
  work. Recommended default.
- **An OS `crontab` entry** running `claude -p "<refresh prompt>"` on a real monthly schedule. Truly
  unattended, but requires the machine to be on at fire time, working non-interactive auth, and
  per-run cost. Use if you want the check to happen without you opening a session.

Either way, the refresh itself is this procedure; the trigger only decides *when* it runs.
