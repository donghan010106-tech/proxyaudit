"""Step 5 -- Action Recommender.

Maps an audit outcome ``{error_type, root_channel, timing}`` to a *structured*
recovery recommendation. The structured object is produced deterministically
(no model needed); a natural-language message is produced either by a built-in
template phraser (offline, deterministic) or by an optional language-model
backend that paraphrases the same structured fields.

The contract is intentionally small so the layer is easy to audit:

    recommend(case) -> Recommendation

where ``case`` carries the per-decision certificate already computed by the
pipeline. Nothing here invents numbers; it only *routes* a verdict to an action.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
import datetime as _dt


# ---- error taxonomy -------------------------------------------------------
# error_type is derived purely from the certificate (necessity / sufficiency /
# advantaged-or-disadvantaged), so the mapping is transparent and stable.
PROXY_CAUSED_REJECTION = "proxy_caused_rejection"        # disadvantaged, nec & suf
PROXY_SUPPRESSED_NO_FLIP = "proxy_suppressed_no_flip"    # suppressed, but not a cause
PROXY_INFLATED_SELECTION = "proxy_inflated_selection"    # advantaged, proxy-inflated
NEAR_FAIR_NO_ACTION = "near_fair_no_action"              # no proxy causation found

_SEVERITY = {
    PROXY_CAUSED_REJECTION: "high",
    PROXY_INFLATED_SELECTION: "medium",
    PROXY_SUPPRESSED_NO_FLIP: "low",
    NEAR_FAIR_NO_ACTION: "none",
}

# structured action codes (machine-readable target of the recommendation)
_ACTION = {
    PROXY_CAUSED_REJECTION: dict(
        action_code="REVERSE_AND_REVIEW",
        target="application",
        steps=[
            "Re-score the application with the named channel neutralized.",
            "Route to human review with the certificate attached.",
            "Notify the candidate that the decision is being re-assessed.",
        ],
    ),
    PROXY_INFLATED_SELECTION: dict(
        action_code="RECHECK_SELECTION",
        target="application",
        steps=[
            "Re-score with the named channel neutralized.",
            "Confirm the candidate still clears the bar on competencies alone.",
        ],
    ),
    PROXY_SUPPRESSED_NO_FLIP: dict(
        action_code="LOG_AND_MONITOR",
        target="case_log",
        steps=[
            "Record that the channel suppressed the score without changing the outcome.",
            "Add to the monitoring set in case the margin shifts.",
        ],
    ),
    NEAR_FAIR_NO_ACTION: dict(
        action_code="NO_ACTION",
        target="none",
        steps=["No proxy-caused harm detected for this decision."],
    ),
}


@dataclass
class AuditCase:
    """The minimal audit outcome the recommender consumes."""
    case_id: str
    error_type: str
    root_channel: str               # human name of the leaking channel
    timing: str                     # ISO timestamp of the decision / audit
    group: str = ""                 # e.g. "disadvantaged" / "advantaged"
    effect: Optional[float] = None  # signed score effect of the channel
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Recommendation:
    case_id: str
    error_type: str
    root_channel: str
    timing: str
    severity: str
    action_code: str
    target: str
    steps: List[str]
    message: str                    # natural-language phrasing
    source: str                     # "template" or "llm"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def as_row(self) -> str:
        """One-line end-to-end view: case | verdict | channel | recommendation."""
        return (f"[{self.timing}] case {self.case_id} | {self.error_type} "
                f"via {self.root_channel} | recommendation: {self.message}")


def classify(necessity: bool, sufficiency: bool, group: str) -> str:
    """Derive the error_type from a certificate (transparent routing)."""
    caused = bool(necessity) and bool(sufficiency)
    if group == "advantaged":
        return PROXY_INFLATED_SELECTION if caused else NEAR_FAIR_NO_ACTION
    if caused:
        return PROXY_CAUSED_REJECTION
    if sufficiency or necessity:
        return PROXY_SUPPRESSED_NO_FLIP
    return NEAR_FAIR_NO_ACTION


# ---- template phraser (offline, deterministic) ----------------------------
def _template_message(case: AuditCase) -> str:
    et, ch = case.error_type, case.root_channel
    if et == PROXY_CAUSED_REJECTION:
        return (f"This rejection was caused by the {ch}: removing only that "
                f"channel flips the decision, and the candidate clears the bar on "
                f"competencies alone. Re-evaluate with the channel neutralized, "
                f"route to human review, and notify the candidate of re-assessment.")
    if et == PROXY_INFLATED_SELECTION:
        return (f"This selection was inflated by the {ch}. Re-score with the "
                f"channel neutralized and confirm the candidate still qualifies on "
                f"competencies alone before finalizing.")
    if et == PROXY_SUPPRESSED_NO_FLIP:
        return (f"The {ch} lowered this score but did not change the outcome; the "
                f"rest of the profile already determined it. No reversal is "
                f"required, but log the case and monitor the margin.")
    return (f"No proxy-caused harm was detected for this decision; the {ch} "
            f"carries little information about the protected attribute here.")


# ---- optional LLM phraser --------------------------------------------------
def _llm_message(case: AuditCase, structured: Dict[str, Any],
                 model: str = "claude-haiku-4-5-20251001") -> Optional[str]:
    """Paraphrase the *structured* recommendation with a language model.

    Returns None on any failure so the caller falls back to the template. The
    model only rephrases fields it is given; it is never the source of the
    verdict or of any number.
    """
    try:
        import os, json
        from anthropic import Anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return None
        client = Anthropic()
        prompt = (
            "You are a compliance assistant. Rephrase the following structured "
            "fairness-audit recommendation into one clear, neutral sentence for a "
            "hiring manager. Do not add facts beyond the fields given.\n\n"
            + json.dumps(structured, indent=2)
        )
        resp = client.messages.create(
            model=model, max_tokens=160,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    except Exception:
        return None


def recommend(case: AuditCase, use_llm: bool = False) -> Recommendation:
    """Map an audit case to a structured recovery recommendation (Step 5)."""
    spec = _ACTION[case.error_type]
    structured = dict(
        case_id=case.case_id, error_type=case.error_type,
        root_channel=case.root_channel, timing=case.timing,
        severity=_SEVERITY[case.error_type], **spec,
    )
    message, source = None, "template"
    if use_llm:
        message = _llm_message(case, structured)
        source = "llm" if message else "template"
    if message is None:
        message = _template_message(case)
    return Recommendation(message=message, source=source, **structured)


def recommend_batch(cases: List[AuditCase], use_llm: bool = False) -> List[Recommendation]:
    return [recommend(c, use_llm=use_llm) for c in cases]


def now_iso() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()
