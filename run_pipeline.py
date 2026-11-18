"""One command to reproduce everything:
ingest -> features -> train (+ quality gate + MLflow) -> audits (leakage, PSI drift, per-region) -> Evidently drift
-> sample prediction.

    python run_pipeline.py                 # full run (a few minutes)
    python run_pipeline.py --fast          # fewer bootstrap/permutation repeats, for CI and smoke tests
    python run_pipeline.py --no-tracking   # skip MLflow
    STBMSE_MIN_GAIN=0.9 python run_pipeline.py --fast   # make the quality gate fail, to see it stop the pipeline

Exit code 0 = everything passed. Exit code 1 = the quality gate failed (nothing is deployed).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from stbmse import audit, config as C, monitor  # noqa: E402
from stbmse.data import ingest  # noqa: E402
from stbmse.gate import QualityGateError  # noqa: E402
from stbmse.predict import Scorer  # noqa: E402
from stbmse.train import train  # noqa: E402


def step(name, fn):
    t0 = time.time()
    print(f"\n=== {name} ===", flush=True)
    out = fn()
    print(f"    done in {time.time() - t0:.1f}s", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--skip-audit", action="store_true", help="skip leakage/PSI/region audits and the Evidently report")
    ap.add_argument("--no-tracking", action="store_true", help="do not log to MLflow")
    a = ap.parse_args()
    if a.no_tracking:
        os.environ["STBMSE_TRACKING"] = "off"
    C.REPORT_DIR.mkdir(parents=True, exist_ok=True)

    _, q = step("1. ingest + validate", lambda: ingest())
    print(f"    {q['rows']} rows | {q['counties']} counties | {q['period']} | warnings: {q['warnings'] or 'none'}")

    try:
        m = step("2-3. features + train + quality gate + MLflow + evaluate", lambda: train(fast=a.fast))
    except QualityGateError as e:
        print(f"\nPIPELINE STOPPED\n{e}", file=sys.stderr, flush=True)
        return 1
    r = m["test_results"]
    print(f"    test PR-AUC: ensemble {r['st_bmse_ensemble']['pr_auc']} vs persistence {r['persistence']['pr_auc']}")

    if not a.skip_audit:
        step("4a. leakage audit", lambda: print(audit.leakage_audit().to_string(index=False)))
        step("4b. drift report (PSI)", lambda: print(json.dumps(audit.drift_report(), indent=2)))
        step("4c. subgroup (region) performance", lambda: print(audit.subgroup_report().to_string(index=False)))
        if monitor.available():
            def _evidently():
                s = monitor.drift_report()
                print(f"    {s['columns_drifted']} of {s['columns_tested']} columns drifted ({s['share_drifted']:.0%}); "
                      f"dataset drift detected: {s['dataset_drift_detected']}")
                print(f"    prediction drift: {s['prediction_drift']}")
                print(f"    HTML report: {monitor.HTML_PATH}")
            step("4d. Evidently drift report (train -> test)", _evidently)
        else:
            print("\n=== 4d. Evidently drift report ===\n    skipped: evidently is not installed (pip install evidently)")

    s = step("5. sample prediction", Scorer)
    print(json.dumps(s.score("Nairobi", 2024, 12), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
