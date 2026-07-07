# Frozen workloads

All files here are **frozen inputs** — locked before any measurement so every published
number traces to identical prompts. Integrity is pinned by `MANIFEST.sha256`.

## Source text (doc-chat)
- **Work:** *Frankenstein; or, The Modern Prometheus* by Mary Wollstonecraft (Godwin) Shelley.
- **Origin:** Project Gutenberg eBook #84 — https://www.gutenberg.org/ebooks/84
  (plain-text UTF-8: https://www.gutenberg.org/cache/epub/84/pg84.txt), downloaded 2026-07-04.
- **License / rights:** The work is **public domain in the United States** (first published
  1818; author d. 1851). The Project Gutenberg header/footer boilerplate **and its license
  section were stripped** (kept only the text between the `*** START ... ***` / `*** END ... ***`
  markers), so `source/frankenstein_pg84.txt` carries no Project Gutenberg trademark or license
  text — it is the plain public-domain work. If redistributed, do not attach the PG trademark.
- **Stripped source:** `source/frankenstein_pg84.txt` — 419,331 chars, 75,042 words,
  **97,954 tokens** under this build's Llama-3.1-8B tokenizer (llama.cpp b9871 `/tokenize`).

## Doc-chat cuts (`docchat/`)
Token-**exact** prefixes of the source, cut with THIS build's tokenizer
(`harness/build_workloads.py`: tokenize full → first N ids → detokenize → write → re-tokenize
to confirm). Actual counts (`cuts_tokencount.json`) all matched targets exactly:

| file          | tokens | chars   |
|---------------|--------|---------|
| doc_2k.txt    | 2048   | 8,733   |
| doc_4k.txt    | 4096   | 17,612  |
| doc_8k.txt    | 8192   | 35,321  |
| doc_16k.txt   | 16384  | 71,331  |
| doc_32k.txt   | 32768  | 141,362 |
| doc_64k.txt   | 65536  | 281,436 |

(Regenerating the ladder reproduced the earlier cuts byte-identically.) **128K is SKIPPED for v1** — the source has only 97,954 tokens; a 128K rung
would require a disclosed two-text concatenation (future work).

- `questions.json` — fixed 5-question set + `prompt_template` (how a cut + question combine).

## Multi-turn (`multiturn/conversation_20turn.json`)
Scripted 20 user turns (assistant replies generated), greedy/deterministic. Context accumulates
(later turns depend on facts from earlier turns) to exercise cross-turn persistence.

## Restart (`restart/restart_scenario.json`)
Same 20-turn conversation with a deliberate server **kill after turn 10** → restart → restore →
resume turns 11–20, plus a cold-recompute control. Drives the reuse assertion + restore probe.

## Regenerating
`harness/build_workloads.py --port <p>` against a running Llama-3.1-8B server reproduces the
cuts byte-for-byte (deterministic). The source download URL + strip rule are above.
