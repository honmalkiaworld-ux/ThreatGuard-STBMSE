"""Quality gate: decide whether a freshly trained model may be registered / deployed.

The gate uses VALIDATION data only. The 2022-2024 test period must stay untouched by any decision, including
"may this model ship?", otherwise it stops being an honest estimate of future performance.

CI runs the pipeline, and a failed gate exits non-zero, so the Docker image is never built or published.
(Course pattern: the quality gate is just an exit code. See fraud_mlops/train.py in the lab repository.)
"""
import os
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score

# The best base learner must beat the persistence baseline by at least this much PR-AUC on validation.
MIN_GAIN_DEFAULT = 0.05


class QualityGateError(RuntimeError):
    """Raised when a trained model does not meet the minimum quality bar."""


@dataclass
class GateResult:
    passed: bool
    checks: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        lines = [("PASS" if c["ok"] else "FAIL") + f"  {c['name']}: {c['detail']}" for c in self.checks]
        return "\n".join(lines)


def min_gain() -> float:
    """Gate strictness; override with the STBMSE_MIN_GAIN environment variable (e.g. 0.90 to see it fail)."""
    return float(os.environ.get("STBMSE_MIN_GAIN", MIN_GAIN_DEFAULT))


def check_quality_gate(y_val, persistence_val, base_val_scores: dict, threshold: float,
                       gain_required: float | None = None) -> GateResult:
    """Evaluate the gate.

    y_val            validation labels (next-month escalation)
    persistence_val  persistence baseline score on the same rows
    base_val_scores  {learner name: validation probabilities}; base learners were fitted on TRAIN only, so these
                     are honest out-of-sample scores (the stacker is fitted on validation, so it is not used here)
    threshold        the cost-tuned alert threshold chosen by the ensemble
    """
    gain_required = min_gain() if gain_required is None else gain_required
    y_val = np.asarray(y_val)
    checks = []

    both = 0 < y_val.sum() < len(y_val)
    checks.append({"name": "validation has both classes", "ok": bool(both),
                   "detail": f"{int(y_val.sum())} positives of {len(y_val)}"})

    finite = all(np.isfinite(np.asarray(s)).all() for s in base_val_scores.values())
    checks.append({"name": "scores are finite", "ok": bool(finite), "detail": "no NaN or infinity" if finite else "NaN/inf found"})

    if both and finite:
        pers = average_precision_score(y_val, persistence_val)
        best_name, best = max(((n, average_precision_score(y_val, s)) for n, s in base_val_scores.items()), key=lambda t: t[1])
        gain = best - pers
        checks.append({"name": f"best learner beats persistence by >= {gain_required:.2f} PR-AUC (validation)",
                       "ok": bool(gain >= gain_required),
                       "detail": f"{best_name} {best:.3f} vs persistence {pers:.3f} (gain {gain:+.3f})"})
    else:
        checks.append({"name": "best learner beats persistence", "ok": False, "detail": "could not be evaluated"})

    interior = 0.02 < threshold < 0.98
    checks.append({"name": "alert threshold is not degenerate", "ok": bool(interior),
                   "detail": f"threshold {threshold:.2f} (grid edge = 0.02 or 0.98 means 'alert on all/none')"})

    return GateResult(passed=all(c["ok"] for c in checks), checks=checks)
