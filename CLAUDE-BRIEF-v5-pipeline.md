# Claude brief: v5 data build and evaluation pipeline

**Hand this to Claude at the start of a session working on this repo.**
Self-contained — it does not assume Claude has seen the other documents.

Companion documents, if present in the repo: `HANDOFF-nigerian-english-asr.md`
(background), `nigerian-english-data-and-eval-cleanup.md` (full plan with the
reasoning behind every step).

**Revision 4** — 2026-08-18. Decisions marked **[settled]** were made by Jeff and
are not open for relitigation. If you think one is wrong, say so and stop; do not
work around it.

*Changed in rev 4: Stage A now stores paths relative to a corpus root rather than
absolute paths, and records that the inventory is a dated snapshot of a corpus
that is still growing.*

---

## 0. How to use this document — read this part yourself

For the team, not for Claude.

- **Paste or attach this document at the start of every Claude session.** Claude
  sessions do not share memory. A session that hasn't seen this file knows none of
  the rules below and will cheerfully reintroduce bugs this document exists to
  prevent.
- **Work one stage at a time.** Start a session, do Stage C, stop. Don't ask for
  the whole pipeline in one go — the point is that you can review what comes back.
- A good opening prompt is boring and specific:
  > "Read `CLAUDE-BRIEF-v5-pipeline.md`. We're doing Stage C only. Start by
  > writing `test_normalize.py` — just the tests, no implementation yet. Explain
  > what each test protects before you write it."
- **If you don't understand a line of code, ask what it does before keeping it.**
  That is not a delay; it is the exercise. Code you can't explain is code you
  can't debug at 11pm before a delivery.
- When a stage's verification output looks wrong, or a number looks surprisingly
  good, **stop and send it to Jeff** rather than adjusting until it looks right.

---

## 1. Your role

You are helping a small team rebuild the data build and evaluation path for a
Nigerian English ASR model (wav2vec2 XLS-R 300m, character CTC, greedy
decoding). The existing pipeline produces WER numbers that cannot be trusted.
This work fixes the measurement. **It does not change the model or the training
recipe.**

**The team is early in their software and ML careers.** That changes how you
should work, not what you should build:

- **Explain before you produce.** State what you're about to write and why,
  in two or three sentences, before writing it.
- **Small, readable diffs.** Prefer the clear implementation over the compact
  one. A `for` loop they can step through beats a nested comprehension.
- **One function at a time.** Do not generate a whole module unprompted.
- **Name the invariant each test protects.** Not "tests normalize" — "this
  catches the case where a curly apostrophe and a straight one produce different
  training text, which is what caused an 11.7-point WER discrepancy."
- **Flag anything you inferred rather than were told.** Especially file paths,
  directory layouts, and field names. Say "I assumed X — confirm before we rely
  on it."
- **Stop at stage boundaries.** Do not roll from Stage C into Stage D because
  the work seems obvious. Each stage ends with verification output that a human
  reads.
- If you are asked to do something that conflicts with §4 (Hard rules), say so
  and explain which rule and why, rather than silently complying or silently
  refusing.

---

## 2. Architecture — already decided, do not relitigate

**[settled] This repo is the build path** for Nigerian English ASR datasets
going forward. The Colab notebooks that produced v3 and v4 are retired as a
source of truth. Anything they did that still needs doing gets reimplemented
here as library code with tests.

**The pipeline is a plain importable Python library plus a CLI, living in this
repo. Any notebook is a three-line launcher.**

```bash
!git clone <repo> && pip install -e idem-ai-core
!python -m idem.normalize --check data/v5_inventory.jsonl
!python -m idem.score --checkpoint <path> --processor <path> --split v5_test.jsonl
```

Reasoning you should preserve in every decision: the central bug being fixed is
that training and scoring used *different* text cleaning. The fix is that both
`import` the same `normalize()`. Anything that copies logic — templating it into
a generated notebook cell, duplicating a regex "just for scoring," a second
charset check — recreates that bug in a new place. **If you find yourself
writing a second copy of a rule, stop and say so.**

Constraints that follow:

- `normalize()` and the split logic must run from a shell and be testable under
  pytest with no app, no Streamlit, no FastAPI, no Google Drive.
- Stages A–D are **CPU-only**. Do not add GPU or `torch` dependencies to them.
- Target environment is a persistent Linux VM, not Colab. Colab is rented for
  Stage E and for training.
- Never read audio from Google Drive in a loop. Copy or extract to local disk
  first.

---

## 3. Repo state — findings from a prior reconnaissance pass

Accurate as of commit `cfedae5`. **Verify before relying on any of it**; the
team may have moved since.

This repo is a *data-collection* app for Efik / Ibibio / Yoruba with Nigerian
English (`en_NG`) recently added as a fourth language profile. **It did not
build the existing `nigerian_train_v3.jsonl` or `v4` datasets** — those came from
standalone Colab notebooks with no connection to this code. You are making this
repo the build path for the first time.

What exists:

| Module | Relevance |
|---|---|
| `schemas/manifest_entry.py` | `ManifestEntry` dataclass — has `speaker_id`, `duration`, `split`, content hash. **No `source`, no license fields, no `text_raw`.** |
| `engine/validator.py` | A **gate**, not a normalizer. NFC + strip, then rejects text outside a character whitelist. |
| `config/language_profiles.py` | Per-language character whitelists including `en_NG`. |
| `engine/audio_pipeline.py` | Resample to 16 kHz, SNR/clipping, SHA-256 id, manifest append, dataset export. |
| `engine/vocab_manager.py` | **A second, mutable charset authority** that `validator.py` does not use, wired into the LLM tool router. Being retired — see §3a. |
| `agent/aziz_orchestrator.py` | "Aziz" — a Claude Haiku tool-calling router (`MODEL = "claude-haiku-4-5-20251001"`). Real code in this repo, not an external service. It imports `VOCAB_EXECUTOR`/`VOCAB_TOOLS` at line 28, which is how an LLM call can currently mutate a training charset. |
| `engine/media_processor.py` | TTS and bulk Whisper STT. Both write to the manifest. See §5. |
| `tests/` | 59 tests, passing. Covers `validator`, `script_generator`, `manifest_entry` only. |

What does not exist anywhere: a normalizer, split logic, a scorer, a speaker-ID
parser, license fields, `wav2vec_scripts/`, or any chunked decoder.

**There is no local dev environment** — no venv, no pinned deps. Setting one up
is a reasonable first step; ask before adding dependencies to `requirements.txt`.

---

## 3a. Working agreements — settled at the plan review

**[settled] Character sets require human review.** A training character set
determines the CTC output layer; one stray character shifts the index mapping
and silently scrambles the model's output with no error raised. It is not
runtime state.

What that means concretely:

- The effective character set for a language lives in **version control**
  (`config/language_profiles.py` or a committed data file), and changes to it go
  through a pull request like any other code change.
- **Remove the `manage_vocabulary` tool from the Aziz tool list.** An LLM tool
  call must not be able to widen or narrow a training charset. Removing it also
  collapses the two competing charset authorities into one, which is hard rule 4.
- Retire the runtime-writable `data/vocab/<lang>_custom.json` path. If any such
  files already exist on a team machine, **surface their contents before deleting
  anything** — they may encode real linguistic decisions that belong in the
  committed profile.
- The `export_reference()` Markdown output is worth keeping: a character
  inventory a linguist can sign off on is exactly the review artifact this needs.

**[settled] All changes land via pull request, reviewed by Jeff.** Do not push
to `main`. Work on a branch, keep the diff small enough to review in one sitting,
and make the PR description say what invariant the change protects.

**[settled] Add a `.gitignore` early — before Stage A writes anything.** There
isn't one today, so the first manifest, vocab JSON, or stray `.wav` lands in a
commit. At minimum: `data/`, `*.wav`, `*.jsonl`, `__pycache__/`, `.venv/`,
`*.egg-info/`, credentials and token files.

**No audio, no transcripts, and no dataset files go into git.** Commit the
scripts, the manifest summaries, and the provenance table — never the corpus.
This is not only repo hygiene: SLR70 carries a deployment scope, and git history
is the one place a mistake cannot be quietly undone.

**[settled] Scoring stays separate from Cobalt's central eval infrastructure for
now.** The consequence you must handle: this creates a second WER convention, and
someone will eventually compare a number from here with a number from there. So
`score.py` must **document its convention explicitly** — which normalizer version,
which decode strategy, which metric implementation and version (`jiwer` vs
`evaluate`, and their exact behavior on empty references) — and record all of it
in the results file, not just in a docstring. Reconciliation should later be a
mechanical exercise, not an archaeological one.

---

## 4. Hard rules

1. **Scripts, not notebook cells**, for anything producing an artifact another
   step depends on.
2. **Fail loudly.** Raise on anything unparseable. Never skip a record and
   continue. No `except Exception: pass`, no silent defaults for missing data.
3. **Never delete data silently.** Every filter reports how many records it
   dropped and why. **A filter dropping over 1% stops for human investigation.**
4. **One source of truth per concern.** One normalizer, one split script, one
   manifest, one scorer.
5. **Verification is code, not prose.** The assertions in §6 go into real test
   files, not comments and not printed output alone.
6. **Never invent an identifier.** Especially a speaker ID — see §6, Stage A.
7. **Do not decide licensing.** See §7.
8. **Version and never overwrite.** The new dataset is `v5`. Every version gets
   a manifest recording what went into it.

---

## 5. Anti-patterns — recognise these, and don't reproduce them

Real bugs from the existing pipeline. If you see this shape in code you're asked
to extend, flag it.

**From the notebooks:**

- `combined_test_data.jsonl` was a `shutil.copyfile` of raw input while cleaning
  happened in memory — cleaned hypothesis scored against uncleaned reference.
  **Cost: 30.21% vs 18.54% WER on the same model and audio.**
- `random.shuffle(data)` + 80/20 cut, with speaker IDs available in filenames
  and unused.
- A filter indexing audio with `startswith("efi_")` while filtering transcripts
  with `startswith("eng_")` — can never match. Printed `0 entries, 0.00 hours`
  and overwrote the dataset file with an empty one. Copied from another
  language's project and never adapted.
- `def normalize_transcripts(batch)` with **no `return batch`** — `datasets.map`
  treats a `None` return as "leave unchanged," so NFC normalization was a silent
  no-op in every run.
- `remove_basic_punctuation` used as both a config flag and a function name; the
  function definition shadowed the boolean, so setting the flag to `False` did
  nothing.
- Lowercasing applied *before* noise-token removal, so `<UNK>` became `<unk>`,
  never matched the uppercase token list, and entered training as the literal
  word `unk`.
- `processor_model = ""` with the correct path commented out one line above —
  warm-starting from a fine-tuned checkpoint while re-deriving the vocabulary
  from data.
- `BASE_DIR` set to a `.jsonl` *file*, then `os.path.join`'d with a filename.
- Two directory trees, `NigerianEnglish` and `NigeriaEnglish`, with some code
  writing to one and reading from the other.

**Already in this repo — do not extend these, flag them if touched:**

- `engine/vocab_manager.py` maintains a **second charset** that `validator.py`
  doesn't consult, persisted to `data/vocab/<lang>_custom.json` and mutable via
  an LLM tool call with no review. Two authorities on "is this character
  allowed" is rule 4 already broken.
- `engine/media_processor.py` hardcodes `split="train"` in both ingest paths, and
  defaults `speaker_id` to `"unknown"` (bulk STT) or `"tts_synthetic"` (TTS).
  **`"unknown"` collapses every speaker to one ID**, which silently destroys a
  speaker-disjoint split.
- `transcribe_audio(add_to_manifest=True)` writes **Whisper's own output** into
  the manifest as ground-truth text, unmarked and indistinguishable from human
  transcription.
- `agent/tools/colab_generator.py` generated cell 6 catches audio-read failures
  and substitutes `np.zeros(16000)` — one second of silence paired with the real
  transcript. It prints "Skipping" but does not skip.
- `schemas.ManifestEntry.validate_metadata()` is never called in production code.

---

## 6. The five stages

Work them in order. **Stop at each boundary and hand the verification output to
a human.** Stage C is the recommended starting point: it's small, CPU-only, and
both D and E depend on it.

### Stage A — Inventory

Produce `v5_inventory.jsonl`: one record per audio/transcript pair, before any
filtering or cleaning.

```json
{
  "id": "en-ng_speaker146_asr-studio_...",
  "audio": "asr-studio/en-ng_speaker146_asr-studio_....wav",
  "text_raw": "The pain in my eye is so wicked.",
  "source": "asr-studio",
  "speaker": "speaker146",
  "duration_sec": 4.31
}
```

- `text_raw` is the transcript **exactly as on disk**. No cleaning here — Stage C
  must be reproducible from this file.
- **`audio` is a path relative to a corpus root, as a plain string** — not an
  absolute path, and not a `{"path": ..., "sampling_rate": ...}` dict. The root
  is supplied at runtime by a `--corpus-root` CLI argument (falling back to a
  `V5_CORPUS_ROOT` environment variable). Resolve with
  `os.path.join(corpus_root, rec["audio"])` at the point of use.
- Get duration from `soundfile.info(path).duration` — header only, do not load
  the audio.

**Why relative, not absolute** — this changed in rev 4 and the reasoning is
load-bearing. The inventory gets built in one environment (Colab, where the
corpus is at `/content/...`) and consumed in others (laptops, and later a
persistent VM). Absolute paths are both machine-specific and, on Colab,
destroyed when the runtime recycles. Baking them into the inventory guarantees a
repeat of the old pipeline's `fix_paths` bug, which rewrote 121,074 of 130,256
paths and left ~9,000 silently pointing at the wrong place. A relative path plus
a runtime root makes the file portable and removes the entire class of problem.

**Never write a path-rewriting function.** If you find yourself wanting one,
the root is being set wrong — fix that instead, and say so.

**The corpus is still growing.** Recording continues, so a directory walk today
and the same walk next month produce different datasets. That is fine, but it
means:

- The inventory **is** the definition of v5, not the directory. Once
  `v5_inventory.jsonl` exists, every later stage reads *it* — never re-walks the
  source tree.
- Record `created_at` and the record count in the manifest, and hash the
  inventory file itself (`sha256`) so a rebuild can be proven identical.
- Rebuilding v5 means replaying the same inventory, not rescanning the folder.
  A newer scan is a *new version*, with a new number.
- **Do not modify, rename, move, or delete anything under the source
  directories.** Adding new recordings is expected and fine; reorganising
  existing ones invalidates every inventory that references them, silently. If
  the layout seems wrong, report it rather than fixing it.

**Speaker ID extraction — the part that must not be sloppy.** The entire
evaluation depends on it.

```python
ASR_STUDIO_RE = re.compile(r'^en-ng_(speaker\d+)_asr-studio_')
OPENSLR_RE    = re.compile(r'^(ng[fm]_\d+)_\d+$')

def get_speaker(basename, source):
    """Return a speaker ID, or raise. Never invent one."""
    if source == "asr-studio":
        m = ASR_STUDIO_RE.match(basename)
        if m:
            return m.group(1)
    elif source == "slr70":
        m = OPENSLR_RE.match(basename)          # <lang><gender>_<speakerid>_<lineid>
        if m:
            return m.group(1)
    raise ValueError(f"Cannot parse speaker ID from {basename!r} (source={source!r})")
```

For SLR70, **prefer speaker IDs from the official metadata if the distribution
provides them**, and use the filename parse as the fallback. If the two ever
disagree, raise — don't pick one.

**Never fall back to using the filename as the speaker ID.** That gives every
file a unique speaker, which makes the Stage D split silently do nothing and
produces a good-looking WER that means nothing.

*Acceptance:* file count on disk per source matches inventory count (print the
missing filenames if not); distinct speaker count is plausible (not 1, not
equal to the file count); total hours and duration min/max/mean printed; every
`audio` path resolves — `os.path.exists(os.path.join(corpus_root, rec["audio"]))`
— and **no `audio` value is absolute** (`assert not os.path.isabs(...)`), which
catches the failure mode this design exists to prevent.

### Stage B — License tagging

**[settled] `en_ng_femalenew` provenance is resolved.** Jeff confirmed it is a
copy of the OpenSLR **SLR70** Nigerian English dataset. The decision is: **do not
use the Drive copy. Source SLR70 from OpenSLR directly.**

Reasons, so you preserve the intent: the official distribution comes with
checksums (so the build is verifiable and reproducible), the license and
attribution text (so we can comply), and the official transcript index — none of
which the Drive copy carries. A copy of unknown vintage with no manifest is not
something we can defend in twelve months.

Concrete tasks:

- Download SLR70 from OpenSLR, record the URL and checksums, and verify them.
- Use the official transcript index rather than whatever transcripts sit beside
  the Drive copy. If they disagree, **report the discrepancy** — do not pick one.
- Spot-check the Drive copy against the official files to confirm it really is
  SLR70 unmodified. If it turns out to be a subset or has been altered, say so —
  that changes what v3 actually contained.
- **Check whether the Drive copy was female-only.** The `ngf_` prefix suggests
  it was, and the official set may also contain male (`ngm_`) speakers. Adding
  them is a corpus-composition change and a **decision for Jeff**, not a free win
  — flag it, don't act on it.
- Record the CC BY-SA 4.0 attribution requirement somewhere durable, not just in
  a code comment.

#### [settled] SLR70-derived models — deployment scope

> Legal (CEO) cleared OpenSLR SLR70 for use in models we host ourselves. Absent a
> narrower definition, we interpret "hosted ourselves" as: the model runs only on
> infrastructure Cobalt owns and operates, weights never leave Cobalt-controlled
> infrastructure, and customer access is solely via a network endpoint Cobalt
> operates.
>
> Out of scope under this reading: customer on-prem or air-gapped installs;
> embedded, edge, or on-device deployment; deployment into a customer's cloud
> account even if Cobalt-managed; shipping containers or weights to a customer
> under any license; and delivering fine-tuned derivatives to a customer.
>
> Attribution: SLR70 credit and a CC BY-SA 4.0 notice in the model card, API
> documentation, and public product notices. Archive the original OpenSLR terms
> alongside the data.
>
> This is our own conservative interpretation, not a legal determination. If a
> deal requires anything outside it, escalate before committing — don't
> reinterpret.
>
> **Decided: 2026-08-12 — Jeff Lilly**

Reproduce that block verbatim in the provenance table. Do not paraphrase it, do
not summarise it, and **do not reason from it to any case it does not name.** Its
final sentence is an instruction to you as much as to anyone.

#### Schema consequence: `deployment_scope`, not `commercial_ok`

The original plan specified a boolean `commercial_ok`. **That field cannot
represent this clearance** — the answer is "yes here, no there," and a boolean
forced to that question is wrong in one direction or the other. Use a closed
enum instead:

```python
DEPLOYMENT_SCOPES = ("unrestricted", "cobalt-hosted-only")
```

| Field | Meaning |
|---|---|
| `license` | SPDX-ish identifier, e.g. `"CC-BY-SA-4.0"`, `"internal"` |
| `license_source` | URL or document a human can go read |
| `deployment_scope` | One of `DEPLOYMENT_SCOPES`. **Raise on anything else** — no free text, no default. |
| `attribution_required` | Bool. `true` for SLR70. |

| Source | `license` | `deployment_scope` |
|---|---|---|
| `asr-studio` | `internal` | `unrestricted` |
| `slr70` | `CC-BY-SA-4.0` | `cobalt-hosted-only` |

**The scope of a dataset is the narrowest scope of any record in it**, and it
propagates: `v5-full` is `cobalt-hosted-only` because one source is. Compute this
in code, write it into `v5_manifest.json`, and carry it into the model card of
anything trained on that dataset. **The constraint has to survive into the model,
not stop at the dataset** — a scope recorded only in a JSONL nobody reads at
deployment time is a scope that isn't enforced.

| Dataset | Contents | Scope |
|---|---|---|
| `v5-internal` | `asr-studio` only | `unrestricted` |
| `v5-full` | `asr-studio` + SLR70 | `cobalt-hosted-only` |

`deployment_scope` **defaults to the narrowest value** and must be positively
widened with a `license_source` a person can go read. "It was already in v3" is
not a justification.

Also produce a provenance table as committed markdown: one row per source — where
it came from, who obtained it, when, under what license, with a link, the scope,
and who decided.

*Acceptance, as CI tests rather than print statements:*

- All four license fields present and non-empty on every record.
- Every `deployment_scope` value is in `DEPLOYMENT_SCOPES`.
- No record is `unrestricted` with an empty `license_source`.
- **No record with `license == "CC-BY-SA-4.0"` is `unrestricted`.**
- The manifest's dataset-level scope equals the narrowest record scope in it.
- Every source with `attribution_required` appears in the attribution file.

Plus a printed table of counts by `(source, license, deployment_scope)`.

### Stage C — Normalization

One function. Used identically at training time and scoring time.

```python
ALLOWED = set("abcdefghijklmnopqrstuvwxyz' ")

def normalize(text: str) -> str:
    """Raw transcript -> training/scoring text. The ONLY normalizer."""
```

Seven operations, **in exactly this order**:

| # | Operation | Why this position |
|---|---|---|
| 1 | Unicode NFC | Must precede anything matching on characters |
| 2 | Map lookalikes: `’ ‘ ′`→`'`, `“ ” ″`→`"`, `– —`→`-` | Before punctuation removal, so curly apostrophes survive as straight ones |
| 3 | Expand digits to words (`num2words`) | Before hyphen handling — `num2words` emits hyphens |
| 4 | Replace `-` with a space | After 3, so `twenty-three` → `twenty three` |
| 5 | Remove all remaining punctuation | Now safe |
| 6 | Lowercase | After token-based steps, so uppercase noise tokens still match |
| 7 | Collapse whitespace runs, strip | Cleanup |

If transcripts contain noise tokens (`<UNK>`, `<NON_SPEECH_NOISE>`), handle them
explicitly **before** step 6.

Tests — write these before accepting any implementation:

```python
def test_curly_and_straight_apostrophe_agree():
    assert normalize("Don't") == normalize("Don’t") == "don't"

def test_hyphen_becomes_space():
    assert normalize("well-being") == "well being"

def test_digits_expand():
    assert normalize("in 1995") == "in nineteen ninety five"

def test_idempotent():
    for s in SAMPLES:
        assert normalize(normalize(s)) == normalize(s)

def test_output_charset():
    for s in SAMPLES:
        assert set(normalize(s)) <= ALLOWED

def test_no_empty_from_nonempty():
    assert normalize("Hello there.") != ""
```

`test_idempotent` and `test_output_charset` catch the most bugs. Keep them.

`num2words` needs care for years, ordinals, and currency — write test cases for
the forms that actually appear in the transcripts, not hypothetical ones.

*Acceptance:* tests pass, **plus** a corpus-level run over every `text_raw` in
the inventory reporting out-of-vocabulary characters (must be empty) and
non-empty-to-empty conversions (must be zero). If OOV characters appear, print
five example transcripts per offending character and **ask a human** what to do
— do not add them to a deletion list. The old corpus contained `é ẹ – — ’` and
digits; each deserves a decision. Then a human reads 30 random before/after
pairs.

### Stage D — Speaker-disjoint splits

Three splits, not two. Dev and test **1–2 hours each**, not a percentage cut.
Dev is for all selection; test is touched rarely.

Group by speaker, sort the speaker list for determinism, shuffle with a fixed
seed, then assign **whole speakers** to test and dev until each hits its target
hours; the rest go to train.

```python
speakers = sorted(by_spk)                  # sort first => deterministic
random.Random(seed).shuffle(speakers)
```

*Acceptance — these assertions are the whole point:*

```python
assert not (train_spk & dev_spk),  f"Speaker overlap train/dev: {train_spk & dev_spk}"
assert not (train_spk & test_spk), f"Speaker overlap train/test: {train_spk & test_spk}"
assert not (dev_spk & test_spk),   f"Speaker overlap dev/test: {dev_spk & test_spk}"
```

Plus: **report** prompt-text overlap between train and each of dev/test (the
prompts are read speech, so the same sentence read by different speakers puts
text on both sides). Report it, don't silently eliminate it — **above ~20% it's
a human decision**, because it changes what the metric means.

Also print the per-speaker hour distribution and the source mix per split, and
have a human look at both before accepting the split. Do not choose the split
strategy or the target hours yourself.

Write `v5_manifest.json` with version, creation date, git commit, sources, per-
split utterance/hour/speaker counts, license mix, prompt-overlap percentages,
the dataset-level `deployment_scope`, and the **SHA-256 of the
`v5_inventory.jsonl` it was built from**. That last field is what makes "same
inputs, same outputs" checkable rather than assumed — and it matters more than
usual here, because the source corpus keeps growing underneath you.

### Stage E — Re-measure the two existing models

One `score.py`. Takes a model checkpoint, a processor directory, and a split
file; reports WER and CER.

- **It imports `normalize` from Stage C and applies it to the reference.** Not a
  local regex, not a copy. The import.
- Normalize the hypothesis the same way for symmetry. If that changes the
  hypothesis, print it — that's a signal worth seeing.
- **Chunk long audio** rather than one forward pass per utterance. The team's
  `test_wav2vec_arpa_chunked.py` did this with `--chunk-len-sec 10.0`; it was
  missing from Drive in both runs. Find it or replicate the chunking.
- Write per-utterance `{id, ref, hyp}` to JSONL, not just a summary number.
- **Greedy decoding.** The language model is a later priority.
- **Use each model's own processor directory**, never a freshly derived
  vocabulary. A vocab ordering mismatch produces garbage WER that looks like a
  bad model rather than a bug.

*Acceptance:* a committed table of WER/CER for both checkpoints on `v5_test`,
with the old reported numbers alongside and a one-line note on why they aren't
comparable.

**Expect the June model's WER to be much worse than 6.61%.** That is the bug
being fixed, not a regression. Do not tune anything to bring it back down. Note
in the write-up that both checkpoints trained on data overlapping `v5_test` at
the speaker level, so even these numbers are optimistic — they are a valid
comparison between the two models and a baseline for the next one.

---

## 7. Escalate, don't decide

Bring these to Jeff rather than resolving them:

| Situation | Why |
|---|---|
| **Any deployment question the scope block doesn't name explicitly** | The interpretation is deliberately conservative and is not a legal determination. Escalate; do not reason by analogy from the cases it does list. |
| Anything else about licensing | Do not infer a license from a filename pattern, and do not reason about what a license permits. Report what you find. |
| SLR70 turns out to be a subset, altered, or to contain speakers the Drive copy didn't | Changes what v3 actually contained, and changes corpus composition |
| A request to widen a `deployment_scope`, or to train a customer-delivered model on `v5-full` | Outside the cleared scope by definition |
| Split strategy or target hours | Decisions about what is being measured |
| Prompt overlap above ~20% in dev or test | Same |
| Out-of-vocabulary characters you can't account for | Could be an encoding issue or an unknown data source |
| A filter dropping more than 1% | Data problem upstream |
| Speaker count equal to file count, or a source with 1–2 speakers and hundreds of files | Speaker regex is wrong; the split will be meaningless |
| **WER much better than expected** | Assume leakage first |
| Any script printing zero results without raising | The `startswith` bug in a new costume |

---

## 8. Definition of done for this phase

- `v5-internal` and `v5-full` built, each with a manifest, rebuildable from
  scratch to the same bytes.
- One `normalize()`, imported by both the training path and `score.py`, with
  passing tests including idempotence and charset closure.
- Speaker-disjointness enforced by assertions in a test file that runs in CI.
- A CI test asserting no CC-BY-SA-4.0 record is marked `unrestricted`, and that
  each dataset's manifest scope equals the narrowest record scope in it.
- A committed provenance table, including SLR70's source URL, checksums, the
  deployment-scope block verbatim, and the archived original OpenSLR terms.
- Attribution text drafted for the model card, API documentation, and public
  product notices — SLR70 credit plus a CC BY-SA 4.0 notice.
- One character-set authority, in version control, with no LLM-writable path.
- WER and CER for both existing checkpoints on `v5_test`, from one scorer, with
  its scoring convention recorded in the results file.

Not in scope, and do not start on them: language model / kenlm decoding,
learning rate or SpecAugment changes, checkpoint-selection changes, or comparing
different starting models.
