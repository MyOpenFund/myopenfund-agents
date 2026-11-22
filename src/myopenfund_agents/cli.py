"""Command line: `mof-agents bench-capture` and `mof-agents bench-run`."""

import argparse
import json
import os
import sys
from datetime import UTC, date, datetime, time
from pathlib import Path

from myopenfund_agents.bench.capture import CaptureError, capture_night, mcp_caller
from myopenfund_agents.bench.fixtures import Expected, Meta, Night, load_all, save
from myopenfund_agents.bench.runner import run_bench, write_report
from myopenfund_agents.bench.score import load_thresholds
from myopenfund_agents.engines.base import EngineError
from myopenfund_agents.engines.factory import engine_from_env
from myopenfund_agents.sanitise import DenyList
from myopenfund_agents.settings import A1Settings

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_CONFIG = 2

CAPTURE_NOTES = [
    "discovery errors kept by first_seen_at <= night_end (approximation)",
    "expected answers pending validation",
]


def _err(message: str) -> None:
    print(f"mof-agents: {message}", file=sys.stderr)


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [v.strip() for v in value.split(",") if v.strip()]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mof-agents", description="MyOpenFund agents: A1 triage engine bench.")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("bench-capture", help="capture one past night from the read-only vault MCP as a bench night")
    c.add_argument("--night", required=True, help="night date YYYY-MM-DD (the night ends that day at --cut UTC)")
    c.add_argument("--cut", default="06:00", help="UTC end of the night, HH:MM (default 06:00)")
    c.add_argument("--corpus", default="central-bank", help="corpus to capture (default central-bank)")
    c.add_argument("--sources", help="comma list of source codes (default: every source of the corpus)")
    c.add_argument("--origin", required=True, choices=["green", "incident"])
    c.add_argument("--description", required=True)
    c.add_argument("--nights", type=Path, default=Path("bench/nights"), help="bench nights directory")

    r = sub.add_parser("bench-run", help="run the configured engine over the bench nights and score it")
    r.add_argument("--profile", required=True, choices=["prod", "local"])
    r.add_argument("--repeat", type=int, default=3, help="attempts per night (default 3)")
    r.add_argument("--only", help="comma list of night ids to run")
    r.add_argument("--nights", type=Path, default=Path("bench/nights"))
    r.add_argument("--thresholds", type=Path, default=Path("bench/thresholds.toml"))
    r.add_argument("--results", type=Path, default=Path("bench/results"))
    r.add_argument("--corpus-repos", type=Path, help="JSON file mapping corpus -> repository")
    return p


def _capture(args: argparse.Namespace) -> int:
    for name in ("MOF_AGENTS_MCP_URL", "MOF_AGENTS_MCP_TOKEN"):
        if not os.environ.get(name):
            _err(f"{name} is not set")
            return EXIT_CONFIG
    try:
        night_end = datetime.combine(date.fromisoformat(args.night), time.fromisoformat(args.cut), UTC)
        settings = A1Settings.from_env()
        deny = DenyList.load(os.environ.get("MOF_AGENTS_DENYLIST"))
    except (ValueError, OSError) as exc:
        _err(str(exc))
        return EXIT_CONFIG
    target = args.nights / night_end.date().isoformat()
    if target.exists():
        _err(f"night directory already exists, refusing to overwrite: {target}")
        return EXIT_FAIL
    call = mcp_caller(os.environ["MOF_AGENTS_MCP_URL"], os.environ["MOF_AGENTS_MCP_TOKEN"])
    try:
        inputs = capture_night(
            call,
            night_end=night_end,
            corpus=args.corpus,
            window_nights=settings.volume_window_nights,
            sources=_csv(args.sources),
            deny=deny,
        )
    except CaptureError as exc:
        _err(f"capture failed: {exc}")
        return EXIT_FAIL
    meta = Meta(
        origin=args.origin,
        description=args.description,
        base_night=None,
        validated_by=None,
        validated_on=None,
        notes=list(CAPTURE_NOTES),
    )
    d = save(Night(inputs=inputs, expected=Expected(findings=[], actions=[]), meta=meta), args.nights)
    print(f"captured {d}")
    return EXIT_OK


def _run(args: argparse.Namespace) -> int:
    if args.repeat < 1:
        _err("--repeat must be >= 1")
        return EXIT_CONFIG
    try:
        settings = A1Settings.from_env()
        engine = engine_from_env(settings)
        thresholds = load_thresholds(args.thresholds, args.profile)
        corpus_repos = None
        if args.corpus_repos is not None:
            corpus_repos = json.loads(args.corpus_repos.read_text(encoding="utf-8"))
            if not isinstance(corpus_repos, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in corpus_repos.items()
            ):
                raise ValueError(f"{args.corpus_repos}: expected a JSON object of corpus -> repository")
    except (EngineError, ValueError, OSError) as exc:
        _err(str(exc))
        return EXIT_CONFIG
    nights = load_all(args.nights)
    only = _csv(args.only)
    if only is not None:
        nights = [n for n in nights if n.inputs.night_id in only]
    if not nights:
        _err(f"no night selected in {args.nights}")
        return EXIT_CONFIG
    scores, verdict_list = run_bench(nights, engine, settings, args.repeat, thresholds, corpus_repos)
    try:
        md = write_report(args.results, engine, args.profile, scores, verdict_list)
    except ValueError as exc:
        _err(str(exc))
        return EXIT_FAIL
    print(md)
    for v in verdict_list:
        print(f"{'PASS' if v.passed else 'FAIL'} {v.criterion}: {v.value:g} (threshold {v.threshold:g})")
    return EXIT_OK if all(v.passed for v in verdict_list) else EXIT_FAIL


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "bench-capture":
        return _capture(args)
    return _run(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
