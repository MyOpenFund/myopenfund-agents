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
`MOF_AGENTS_A1_VOLUME_DROP_RATIO=0.3`).

| Setting | Default | Meaning |
|---|---|---|
| `VOLUME_WINDOW_NIGHTS` | 14 | nights of history kept as the volume baseline |
| `VOLUME_DROP_RATIO` | 0.2 | a source is flagged below this share of its median nightly volume |
| `MIN_HISTORY_NIGHTS` | 5 | history nights needed before a volume drop is judged |
| `CADENCE_GRACE_DAYS` | 0 | days of lateness tolerated before cadence is flagged |
| `SUMMARY_CHAR_BUDGET` | 12000 | maximum summary size sent to the model |
| `ERROR_SAMPLE_MAX_CHARS` | 300 | maximum length of one error sample |
| `ERROR_SAMPLES_PER_SOURCE` | 3 | error samples kept per source |
| `GENERAL_OUTAGE_MIN_SOURCES` | 5 | more failing sources than this are grouped as one general outage |
| `MAX_NEW_ISSUES_PER_RUN` | 5 | above this, new issues collapse into one summary issue |
| `MAX_SCHEMA_RETRIES` | 2 | retries when the model output fails the schema |
| `TEMPERATURE` | 0.0 | model temperature |

## Bench commands

The bench replays known nights (`bench/nights/`) through the configured engine and scores it against
`bench/thresholds.toml`.

Capture a past night from the read-only vault MCP (needs `MOF_AGENTS_MCP_URL` and `MOF_AGENTS_MCP_TOKEN`;
optional `MOF_AGENTS_DENYLIST`, a TOML file of extra `hosts` and `path_roots` to refuse):

```sh
mof-agents bench-capture --night 2026-11-20 --origin green --description "quiet night" \
  [--cut 06:00] [--corpus central-bank] [--sources us,fr] [--nights bench/nights]
```

The night ends on `--night` at `--cut` UTC. A night that carries any infrastructure detail is refused
whole. The captured night has no expected answer yet: fill and validate `expected.json` before relying on it.

Run the bench with the configured engine:

```sh
mof-agents bench-run --profile local [--repeat 3] [--only 2026-11-20,2026-11-21] \
  [--nights bench/nights] [--thresholds bench/thresholds.toml] [--results bench/results] \
  [--corpus-repos corpus-repos.json]
```

It writes `<UTC stamp>-<engine>-<model>.json` and `.md` into `--results` (refused, nothing written, if the
report would carry an infrastructure detail), prints the Markdown path and one line per verdict.

## Exit codes

| Code | `bench-capture` | `bench-run` |
|---|---|---|
| 0 | night captured | every verdict passes |
| 1 | capture refused or failed; night directory already exists | a verdict fails; report refused |
| 2 | missing environment variable or bad argument | engine misconfigured, bad thresholds/argument, no night selected |
