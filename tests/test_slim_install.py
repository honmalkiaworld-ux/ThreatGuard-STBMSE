"""The Docker image installs only requirements-api.txt. Every module the pipeline imports at start-up must work
WITHOUT the optional packages (MLflow, Evidently, SHAP, Optuna). This is exactly the failure CI's Docker build caught once:
a module that imported `shap` at the top made `docker build` crash. Here the optional packages are blocked in a fresh
interpreter, so the check does not depend on what happens to be installed on the machine running the tests."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPTIONAL = ("mlflow", "evidently", "shap", "optuna")
MODULES = ("run_pipeline", "stbmse.train", "stbmse.audit", "stbmse.monitor", "stbmse.tracking", "stbmse.explain",
           "stbmse.rolling", "stbmse.predict", "stbmse.api", "stbmse.gate")


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, cwd=str(ROOT))


def _blocked_prelude() -> str:
    block = ", ".join(f"{m!r}" for m in OPTIONAL)
    return (f"import sys\nfor m in ({block},):\n    sys.modules[m] = None   # makes `import m` raise ImportError\n"
            f"sys.path[:0] = [r'{ROOT}', r'{ROOT / 'src'}']\n")


def test_pipeline_modules_import_without_the_optional_packages():
    code = _blocked_prelude() + "import importlib\nfor mod in %r:\n    importlib.import_module(mod)\nprint('ok')\n" % (MODULES,)
    r = _run(code)
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-1500:]


def test_optional_features_report_themselves_unavailable_instead_of_crashing():
    code = _blocked_prelude() + (
        "from stbmse import explain, monitor, tracking\n"
        "print(explain.available(), monitor.available(), tracking.available())\n")
    r = _run(code)
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip() == "False False False"
