# Engines

Agent A1 sends a short text summary of the night to a language model and gets back schema-validated
findings. The engine is chosen **by environment only**; nothing in the code or the bench picks a model.

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
is missing.

> A subscription is for your **own, individual use**: run this engine on your own machine for your own
> instance only. Do not share a token, put it in CI, or serve other people with it; use an API key for that.

## A1 settings (`MOF_AGENTS_A1_*`)

Every threshold is a setting, overridden by `MOF_AGENTS_A1_<NAME>` (for example
`MOF_AGENTS_A1_VOLUME_DROP_RATIO=0.3`). An out-of-range value stops the run with an error naming the
variable: decimal settings must be finite and above 0 (`TEMPERATURE` may be 0), whole-number settings
must be 0 or more (`VOLUME_WINDOW_NIGHTS`, `SUMMARY_CHAR_BUDGET` and `ERROR_SAMPLE_MAX_CHARS` at least 1).
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
| `GENERAL_OUTAGE_MIN_SOURCES` | 5 | more failing sources than this are grouped as one general outage |
| `MAX_NEW_ISSUES_PER_RUN` | 5 | above this, new issues collapse into one summary issue |
| `MAX_SCHEMA_RETRIES` | 2 | retries when the model output fails the schema |
| `TEMPERATURE` | 0.0 | model temperature |
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

The bench replays known nights (`bench/nights/`) through the configured engine and scores it against
`bench/thresholds.toml`.

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

`no_run` is an addition to the spec's trap list (approved 2026-11-23). The instructions tell the model to
report one `(*, unknown)` error finding for the missing run when the summary says
`No run finished during this night.`, in addition to findings for any unusual sources. On such a night
the deterministic layer gives that finding its own fingerprint, `a1:<default_repo>:*:no_run`, so it is
never mistaken for the chronic `a1:<default_repo>:*:unknown` issue of a failing run without per-source
detail: a no-run night opens (or comments on) its own no-run issue. The trap refuses a base that already
carries an open `*:no_run` issue, or that still shows unusual sources once its runs are removed
(discovery errors or overdue cadence), so those two situations are not measured.

### Run the bench

Run the bench with the configured engine:

```sh
mof-agents bench-run --profile local [--repeat 3] [--only 2026-11-20,2026-11-21] \
  [--nights bench/nights] [--thresholds bench/thresholds.toml] [--results bench/results] \
  [--corpus-repos corpus-repos.json]
```

It prints one line per verdict, then writes `<UTC stamp>-<engine>-<model>.json` and `.md` into `--results`
and prints the Markdown path.

The JSON report records how it was produced (`repeat`, every A1 setting, the threshold profile values) and,
for each night and attempt, the engine's findings before grouping (title, diagnosis, evidence, severity:
kept so diagnoses can be graded later), the planned actions, seconds, schema retries and input/output
tokens when the engine reports them. An engine error, a finding title, a diagnosis or an evidence line
that carries an infrastructure detail is kept as `withheld (<reasons>)`; any other infrastructure detail
refuses the report (nothing written).

### Profiles

`bench/thresholds.toml` holds two threshold profiles, chosen with `--profile`:

- `prod`: the bar for the engine that will run A1 every night. Every expected problem found, no false
  alarm, every action right, every general outage grouped, every output valid, identical answers on every
  repeat; no time limit.
- `local`: the bar for a free local model on modest hardware. At least 80% of expected problems found,
  no false alarm, every action, grouping and output right, agreement of at least 66% on the worst night,
  and at most 600 s per night.

## Exit codes

| Code | `bench-capture` | `bench-run` |
|---|---|---|
| 0 | night captured | every verdict passes |
| 1 | capture refused or failed; night directory already exists | a verdict fails; report refused |
| 2 | missing environment variable, deny-list unset or empty (without `--no-denylist`), unreadable deny-list or bad argument | engine misconfigured, unreadable deny-list, bad thresholds/argument, unreadable night, unknown `--only` id, no night selected |
