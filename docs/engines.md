# Engines

The code decides which findings a night has; the engine only explains them. It gets a short text summary of the
night and the detected findings, and returns schema-validated explanations, each with quotes the code checks
against the summary and the findings' evidence. Without an engine, or when it fails, the same alerts go out with "No explanation.". The
engine is chosen **by environment only**; nothing in the code or the bench picks a model.

| Variable | Meaning |
|---|---|
| `MOF_AGENTS_ENGINE` | `pydantic-ai` (default) or `claude-code` |
| `MOF_AGENTS_MODEL` | model name (required for `pydantic-ai`; optional for `claude-code`) |
| `MOF_AGENTS_BASE_URL` | OpenAI-compatible endpoint (`pydantic-ai` only) |
| `MOF_AGENTS_API_KEY` | key for that endpoint (`pydantic-ai` with `MOF_AGENTS_BASE_URL` only) |

## Three ways to plug a model

### 1. A local model (free, private)

Any OpenAI-compatible server works, for example Ollama:

```sh
export MOF_AGENTS_ENGINE=pydantic-ai
export MOF_AGENTS_BASE_URL=http://localhost:11434/v1
export MOF_AGENTS_MODEL=qwen3:8b
```

No key is needed for a local server. For a hosted OpenAI-compatible endpoint, also set `MOF_AGENTS_API_KEY`.

### 2. Your own API key

Leave `MOF_AGENTS_BASE_URL` unset and give a pydantic-ai model string; the key is read from the
provider's usual variable:

```sh
export MOF_AGENTS_ENGINE=pydantic-ai
export MOF_AGENTS_MODEL=openai:gpt-4o-mini
export OPENAI_API_KEY=<your key>
```

The package ships the OpenAI provider only; another provider's model string needs its pydantic-ai extra
installed alongside. Every call is billed to your key.

### 3. A Claude subscription (`claude -p`)

```sh
export MOF_AGENTS_ENGINE=claude-code
export CLAUDE_CODE_OAUTH_TOKEN=<token from `claude setup-token`>
unset ANTHROPIC_API_KEY
# optional: export MOF_AGENTS_MODEL=<model alias>
```

The engine runs the `claude` CLI with no tools and no MCP server. It refuses to start when
`ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_USE_BEDROCK` or `CLAUDE_CODE_USE_VERTEX` is set
(any of them would bill an API account instead of the subscription), and when `CLAUDE_CODE_OAUTH_TOKEN`
is missing. Your own hooks, plugins and user `CLAUDE.md` still load: point `CLAUDE_CONFIG_DIR` at an empty
directory to keep them out of A1. Reports name the model that answered, also when none is configured.

> A subscription is for your **own, individual use**: run this engine on your own machine for your own
> instance only. Do not share a token, put it in CI, or serve other people with it; use an API key for that.

## A1 settings (`MOF_AGENTS_A1_*`)

Every threshold is a setting, overridden by `MOF_AGENTS_A1_<NAME>` (for example
`MOF_AGENTS_A1_VOLUME_DROP_RATIO=0.3`). An out-of-range value stops the run with an error naming the
variable: decimal settings must be finite and above 0 (`TEMPERATURE` may be 0), whole-number settings
must be 0 or more (`VOLUME_WINDOW_NIGHTS`, `SUMMARY_CHAR_BUDGET` and `ERROR_SAMPLE_MAX_CHARS` at least 1).
`EXPLAIN_GREEN_NIGHTS` takes `true`/`false` (also `1`/`0`, `yes`/`no`, `on`/`off`).
There is no "no timeout" value for `ENGINE_TIMEOUT_S`: set a large number instead.

| Setting | Default | Meaning |
|---|---|---|
| `VOLUME_WINDOW_NIGHTS` | 14 | nights of history kept as the volume baseline |
| `VOLUME_DROP_RATIO` | 0.2 | a source is flagged below this share of its median nightly volume |
| `MIN_HISTORY_NIGHTS` | 5 | history nights needed before a volume drop is judged |
| `CADENCE_GRACE_DAYS` | 0 | days of lateness tolerated before cadence is flagged |
| `CADENCE_NEW_MAX_DAYS` | 14 | a series later than this is long overdue: counted in one line, never flagged (0 = no upper bound) |
| `SUMMARY_CHAR_BUDGET` | 12000 | maximum summary size sent to the model (below about 1000 the fixed header lines alone can exceed it) |
| `NOTE_CHAR_BUDGET` | 2000 | characters for the degraded-run and the long-overdue lines, each group at most half of it, never over half the summary budget |
| `FAILING_RUNS_CHAR_BUDGET` | 3000 | characters for the failing-run lines, never over a quarter of the summary budget; the rest is counted |
| `ERROR_SAMPLE_MAX_CHARS` | 300 | maximum length of one error sample |
| `ERROR_SAMPLES_PER_SOURCE` | 3 | error samples kept per source |
| `GENERAL_OUTAGE_MIN_SOURCES` | 5 | more sources than this with the same category (never `cadence_overdue` or `unknown`), or matched by the detector's failed-run row, are grouped as one general outage |
| `MAX_NEW_ISSUES_PER_RUN` | 5 | above this, new issues collapse into one summary issue that carries **titles only** (with their fingerprints): no diagnosis, explanation or evidence line |
| `MAX_SCHEMA_RETRIES` | 2 | retries when the model output fails the schema or names an unknown finding id; the retry tells the model only fixed wording (never its own text or the CLI's output) |
| `TEMPERATURE` | 0.0 | model temperature |
| `EXPLAIN_GREEN_NIGHTS` | false | `true` or `false`: also ask the model about a night with no finding (observations only) |
| `QUOTE_MIN_CHARS` | 8 | a quote shorter than this (whitespace collapsed) counts as a false quote |
| `QUOTES_MAX` | 3 | most quotes accepted in one explanation (at least 1) |
| `OBSERVATIONS_MAX` | 3 | most observations accepted in one answer |
| `OBSERVATION_MAX_CHARS` | 300 | longest accepted observation (at least 1) |
| `FEEDBACK_ERRORS_MAX` | 5 | schema errors named to `claude-code` when it is asked to answer again (at least 1) |
| `ENGINE_ERROR_MAX_CHARS` | 500 | longest engine error text kept on a triage, withheld marker included; the text is scanned before and after the cut (at least 1) |
| `ENGINE_TIMEOUT_S` | 600.0 | seconds allowed for one model request (`pydantic-ai`: each request, SDK retries included) or one `claude -p` call (`claude-code`) |

A few literals remain in the code. They shape text, they do not judge a night, so they are fixed
constants rather than settings: `REASON_CHARS` (300, the longest slice of CLI output quoted in a
`claude-code` error), `FRESH_CADENCE` (12 h, how soon after the night a capture may still trust the
"now only" cadence view), and the 500 / 1900 character cuts on an evidence line and on the body of a
summary issue.

### Failures shared by several sources

A run over several sources stamps its one outcome on each of them. When a **degraded** run (completed, exit
3) has rows with a signal of their own (truncation, fetch error, failed document, bad content, error
sample), only those sources are flagged `failed_run`, tonight and in the history; the others are named on
one line. A crashed run, or a degraded run no row explains, flags every source: an unexplained failure is
never hidden. Long overdue cadence series are counted and their sources named, not flagged.

## Bench commands

The bench replays known nights (`bench/nights/`) through the whole triage: the detector, which must be exact, and
the configured engine, scored on its explanations against `bench/thresholds.toml`.

### Deny-list (`MOF_AGENTS_DENYLIST`)

Bench nights and reports are committed to a public repository. Besides its built-in checks (private
IPv4 and IPv6 addresses, credentials including `Authorization: Basic`, bearer and GitHub tokens, signed
URLs), the sanitiser refuses the host names and local path roots of your deployment, listed in a TOML
file named by `MOF_AGENTS_DENYLIST`:

```toml
hosts = ["nas.example.lan"]
path_roots = ["/srv/private/"]
```

Both commands use it: `bench-capture` refuses a night that carries any listed entry, `bench-run`
refuses or withholds report text that does. `bench-capture` **requires** it: with the variable unset, or
a deny-list with no host and no path root, it exits 2 unless `--no-denylist` is given explicitly
(built-in checks only; not recommended for a real deployment).

### Capture a night

Capture a past night from the read-only vault MCP (needs `MOF_AGENTS_MCP_URL`, `MOF_AGENTS_MCP_TOKEN`
and `MOF_AGENTS_DENYLIST`):

```sh
mof-agents bench-capture --night 2026-11-20 --origin green --description "quiet night" \
  [--cut 06:00] [--corpus central-bank] [--sources us,fr] [--nights bench/nights] [--no-denylist]
```

The night ends on `--night` at `--cut` UTC. A night that carries any infrastructure detail is refused
whole. When newer rows push the night out of the server's newest-first page, the capture asks again with
`until` (needs a `vault_runs_status` that supports it), and refuses if the page is still cut. The captured
night has no expected answer yet: fill and validate `expected.json` and set
`meta.validated_by` before committing it (CI refuses a committed night that is neither a trap nor validated).

### Traps

A trap damages a green night in one known way and states exactly what A1 must report
(`myopenfund_agents.bench.traps`). Traps are not stored: `bench/nights/traps.toml` names the base night and
the traps, and the bench rebuilds them; each one checks that the damaged night really carries its signal.

| Trap | Damage | Expected finding | Expected action |
|---|---|---|---|
| `volume_drop` | one source sees far fewer documents | `(<source>, volume_drop)` | `new_issue` |
| `bad_content` | one source returns unreadable PDFs | `(<source>, unreadable_pdf)` | `new_issue` |
| `general_outage` | every run and source fails (needs more than `GENERAL_OUTAGE_MIN_SOURCES` sources) | `(*, general_outage)` | `new_issue` |
| `new_source` | a new source appears with zero documents | none | none |
| `known_problem` | unreadable PDFs on a source with an open issue | `(<source>, unreadable_pdf)` | `comment` |
| `no_run` | no run and no source run tonight: the scheduler did not fire | `(*, unknown)` | `new_issue` |
| `chronic` | one source fails on three consecutive nights (three nights made) | `(<source>, source_site_down)` each night | `new_issue`, then `comment` twice |

`no_run` is an addition to the spec's trap list (approved 2026-11-23). When the summary says
`No run finished during this night.`, the detector makes one `(*, unknown)` error finding for the missing run, in
addition to any finding about unusual sources. On such a night that finding gets its own fingerprint,
`a1:<default_repo>:*:no_run`, so it is never mistaken for the chronic `a1:<default_repo>:*:unknown` issue of a
failing run without per-source detail: a no-run night opens (or comments on) its own no-run issue. The trap
refuses a base that already carries an open `*:no_run` issue, or that still shows unusual sources once its runs
are removed (discovery errors or overdue cadence), so those two situations are not measured.

### Run the bench

CI runs no model on pull requests, but checks the detector on every committed night; the `Local model`
workflow is a manual smoke test. Run the bench on your own machine before changing a model or the prompt.

Run the bench with the configured engine:

```sh
mof-agents bench-run --profile local [--repeat 3] [--only 2026-11-20,2026-11-21] \
  [--nights bench/nights] [--thresholds bench/thresholds.toml] [--results bench/results] \
  [--corpus-repos corpus-repos.json]
```

It prints one line per verdict and per reported figure, then writes `<UTC stamp>-<engine>-<model>.json` and `.md`
into `--results` and prints the Markdown path. At least one selected night must have a detected finding: otherwise
there is nothing to explain, no model is called and the run stops with exit code 2.

| Criterion | What it counts | Threshold |
|---|---|---|
| `detector exact` | nights whose findings and actions equal the expected answer in every attempt (no model) | 100% |
| `findings explained` | findings with a kept explanation / detected findings | `min_explained` |
| `valid output` | attempts whose engine answer was usable / attempts | `min_valid` |
| `quotes exact (of quotes given)` | quotes found verbatim in the summary or in one evidence line of a finding / quotes given | `min_quotes_exact` |
| `fallback nights` | nights where some attempt sent a finding out unexplained | `max_fallback_nights` (optional) |
| `slowest night (s)` | slowest attempt | `max_seconds_per_night` (optional) |

`max_fallback_nights` is set for `prod` (`local` only reports the figure), `max_seconds_per_night` for `local`. An
explanation returned without a quote counts as one inexact quote; a missing explanation is measured by `findings
explained`; no quote in the whole run is 0%. A failed attempt lasts the time it took, so a timed-out engine can fail
the time limit. When the detector is wrong on a night, in any attempt, `bench-run` prints a `MISMATCH` line and the
report has a "Detector mismatches" section: expected and detected findings and actions.

The model criteria run over **every** attempt of every night with at least one detected finding: a failed engine is
0 explained, never skipped. A night without finding is listed with its detector result and is in no model figure.
Reported without a threshold: fallback nights, numbers in a diagnosis that neither the summary nor the evidence lines show,
`likely_cause` agreement per finding over the repeats (mean and worst; an attempt that gave no explanation for the
finding agrees with no other attempt, so a finding never explained is at 0%; `-` with a single attempt), mean and
worst seconds, and the tokens the engine reports (`partial (n of m attempts)` when some are unknown, `-` when all are).
A threshold file with an unknown or missing key is refused.

The JSON report records how it was produced (`repeat`, every A1 setting, the threshold profile values) and, for each
night and attempt: the detector's findings (title, evidence, other flags) and actions, the engine error, each kept
explanation (diagnosis, likely cause, severity, quotes: kept so diagnoses can be graded later), why the other
explanations were dropped (`false quote`, `no quote`, `empty`, `withheld`), a sanitised copy of each dropped one with
its reason (`dropped_explanations`), the observations, quote and number counts, seconds, schema retries and tokens.
The model's raw answer is never written, and a dropped explanation never reaches an issue. A finding title, an evidence line, a diagnosis, a quote, an observation or an engine error that carries an
infrastructure detail is kept as `withheld (<reasons>)` (an explanation already caught during the triage
is dropped, its copy keeping the marker; such an observation is dropped entirely); any other infrastructure detail refuses the report (nothing written).

### Profiles

`bench/thresholds.toml` holds two threshold profiles, chosen with `--profile`:

- `prod`: the bar for the engine that will run A1 every night. Every finding explained, every output valid, every
  quote exact, no fallback night; no time limit.
- `local`: the bar for a free local model on modest hardware. At least 90% of findings explained, 90% of outputs
  valid and 95% of quotes exact, and at most 600 s per night; fallback nights are only reported.

## Exit codes

| Code | `bench-capture` | `bench-run` |
|---|---|---|
| 0 | night captured | every verdict passes |
| 1 | capture refused or failed; night directory already exists | a verdict fails; report refused |
| 2 | configuration error | configuration error |

Configuration errors:

- `bench-capture`: missing environment variable, deny-list unset or empty (without `--no-denylist`), unreadable
  deny-list or bad argument.
- `bench-run`: engine misconfigured, unreadable deny-list, bad thresholds or argument, unreadable night, unknown
  `--only` id, no night selected, no detected finding in the selected nights.
