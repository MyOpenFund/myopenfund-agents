"""One night, end to end: the code detects and plans, the model only words what the code decided.

An alert never waits for the model: the actions are planned from the detected findings before the engine is
called, and any failure of the model leaves them as they are (spec sections 2 and 4).
"""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from myopenfund_agents.core.actions import Action, plan_actions
from myopenfund_agents.core.detector import Detected, detect
from myopenfund_agents.core.explanations import WITHHELD, Checked, apply, check, id_error
from myopenfund_agents.core.findings import Finding, Findings
from myopenfund_agents.core.inputs import NightInputs
from myopenfund_agents.core.prompt import explain_prompt
from myopenfund_agents.core.summary import NightSummary, build_summary
from myopenfund_agents.engines.base import Engine, EngineResult
from myopenfund_agents.sanitise import DenyList, error_text, violations
from myopenfund_agents.settings import A1Settings


@dataclass(frozen=True)
class Triage:
    detected: list[Detected]
    findings: list[Finding]  # the detected findings, with the model's words on those it explained
    actions: list[Action]
    checked: Checked | None = None
    engine_error: str | None = None  # sanitised; set when the engine was called and its answer was not used
    # the model's raw answer: unsanitised, never publish it
    result: EngineResult | None = field(default=None, repr=False)
    summary: NightSummary | None = field(default=None, repr=False)

    @property
    def unexplained(self) -> list[str]:
        kept = self.checked.kept if self.checked else {}
        return [d.finding_id for d in self.detected if d.finding_id not in kept]

    @property
    def degraded(self) -> bool:
        """True when a finding went out without the model's words, or the engine failed when it was called."""
        return bool(self.unexplained) or self.engine_error is not None


def _plan(
    findings: list[Finding],
    inputs: NightInputs,
    repo_of: Mapping[str, str],
    default_repo: str,
    settings: A1Settings,
    summary: NightSummary,
) -> list[Action]:
    return plan_actions(
        Findings(findings=findings),
        inputs.open_issues,
        dict(repo_of),
        default_repo,
        settings,
        no_run=summary.no_run,
    )


def _violations(f: Finding, deny: DenyList) -> set[str]:
    """What the sanitiser finds in the texts a poster publishes: each one, and the whole body as joined."""
    texts = [f.title, f.diagnosis, *f.evidence, "\n".join([f.title, f.diagnosis, *f.evidence])]
    return {r for t in texts for r in violations(t, deny)}


def _published(checked: Checked, detected: list[Detected], deny: DenyList) -> Checked:
    """Drop the explanations whose applied finding carries an infra detail the detected one did not have."""
    kept = dict(checked.kept)
    dropped = dict(checked.dropped)
    for d in detected:
        e = kept.get(d.finding_id)
        if e is not None and _violations(apply(d, e), deny) - _violations(d.finding, deny):
            del kept[d.finding_id]
            dropped[d.finding_id] = WITHHELD
    return replace(checked, kept=kept, dropped=dropped)


def _reworded(base: list[Action], explained: list[Action]) -> list[Action]:
    """The detector's actions, taking only the text and severity of the explained plan. Raises if they differ."""
    if len(base) != len(explained):
        raise ValueError("explained plan differs from the detected plan")
    out = []
    for b, e in zip(base, explained, strict=True):
        if (b.kind, b.fingerprint, b.repo, b.issue_number) != (e.kind, e.fingerprint, e.repo, e.issue_number):
            raise ValueError("explained plan differs from the detected plan")
        out.append(Action(b.kind, b.fingerprint, b.repo, e.finding, b.issue_number))
    return out


def triage(
    inputs: NightInputs,
    engine: Engine | None,
    settings: A1Settings,
    repo_of: Mapping[str, str],
    default_repo: str,
    deny: DenyList,
) -> Triage:
    summary = build_summary(inputs, settings)
    detected = detect(summary, settings)
    detected_findings = [d.finding for d in detected]
    base = _plan(detected_findings, inputs, repo_of, default_repo, settings, summary)
    plain = Triage(detected, detected_findings, base, summary=summary)
    if engine is None or not (detected or settings.explain_green_nights):
        return plain
    result: EngineResult | None = None
    try:
        result = engine.explain(explain_prompt(summary.text, detected), frozenset(d.finding_id for d in detected))
        if result.explanations is None:
            raise ValueError("the engine returned no explanations")
        error = id_error(result.explanations, frozenset(d.finding_id for d in detected), settings)
        if error is not None:
            raise ValueError(error)
        checked = _published(check(result.explanations, detected, summary.text, settings, deny), detected, deny)
        findings = [apply(d, checked.kept.get(d.finding_id)) for d in detected]
        actions = _reworded(base, _plan(findings, inputs, repo_of, default_repo, settings, summary))
    except Exception as exc:  # noqa: BLE001 - whatever the model side does, the alert goes out
        text = error_text(exc, deny)[: settings.engine_error_max_chars]
        return Triage(detected, detected_findings, base, None, text, result, summary)
    return Triage(detected, findings, actions, checked, None, result, summary)
