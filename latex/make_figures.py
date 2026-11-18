"""Build the figures for the LaTeX report and slides from the generated results in reports/.

    python latex/make_figures.py

Writes latex/figures/<name>.pdf (report size) and latex/figures/<name>_s.pdf (slide size: larger type).
Colours: blue (#2a78d6) = the model / main finding, orange (#eb6834) = the contrast (baseline or the leaky setting),
grey = non-focal. Blue/orange pass the dataviz validator (adjacent CVD dE 24.7, contrast >= 3:1 on white).
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REP, OUT = ROOT / "reports", ROOT / "latex" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#898781"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

metrics = json.loads((ROOT / "artifacts" / "metrics.json").read_text())
test_rate = metrics["escalation_rate"]["test"]
res = pd.read_csv(REP / "results_test.csv").set_index("model")
la = pd.read_csv(REP / "leakage_audit.csv")
cal = pd.read_csv(REP / "calibration_test.csv")
rg = pd.read_csv(REP / "subgroup_by_region.csv").sort_values("false_alarm_rate")
dq = json.loads((REP / "data_quality.json").read_text())
yr = pd.Series(dq["escalation_rate_by_year"])
yr.index = yr.index.astype(int)


def build(K: float, tag: str, sizes: dict):
    """K scales every font, marker and line; `sizes` gives the figure size (inches) per figure."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10 * K, "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
        "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.06,
    })

    def save(fig, name):
        fig.savefig(OUT / f"{name}{tag}.pdf")
        plt.close(fig)
        print("wrote", f"{name}{tag}.pdf")

    # -------------------------------------------------------------- 1. PR-AUC with confidence intervals
    order = ["persistence", "logistic_regression", "xgboost", "random_forest", "st_bmse_ensemble"]
    labels = {"persistence": "Persistence baseline", "logistic_regression": "Logistic regression", "xgboost": "XGBoost",
              "random_forest": "Random forest", "st_bmse_ensemble": "ST-BMSE ensemble"}
    fig, ax = plt.subplots(figsize=sizes["results"])
    for i, m in enumerate(order):
        r = res.loc[m]
        col = ORANGE if m == "persistence" else BLUE if m == "st_bmse_ensemble" else GREY
        y = len(order) - 1 - i
        ax.plot([r.pr_auc_ci_low, r.pr_auc_ci_high], [y, y], color=col, lw=2.2 * K, solid_capstyle="round", zorder=2)
        ax.plot(r.pr_auc, y, "o", ms=9 * K, color=col, mec="white", mew=1.5, zorder=3)
        ax.text(0.815, y, f"{r.pr_auc:.3f}  [{r.pr_auc_ci_low:.3f}, {r.pr_auc_ci_high:.3f}]", va="center", ha="left",
                color=INK, fontsize=9 * K)
    ax.axvline(test_rate, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(test_rate + 0.006, -0.44, f"no-skill = {test_rate:.2f} (test base rate)", ha="left", va="center",
            fontsize=8.5 * K, color=INK2)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[m] for m in reversed(order)])
    ax.set_xlim(0.38, 0.81)
    ax.set_ylim(-0.7, len(order) - 0.3)
    ax.set_xlabel("PR-AUC, 2022–2024 test period (line = 95% county-block bootstrap CI)")
    ax.grid(axis="y", visible=False)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8])
    fig.subplots_adjust(right=0.64)
    save(fig, "fig_results")

    # -------------------------------------------------------------- 2. leakage audit
    short = ["Nowcast: this month's indicators\n→ this month's label", "Forecast: this month's indicators only",
             "Forecast: full ensemble", "Forecast: ensemble without Sentiments", "Forecast: escalation history only"]
    fig, ax = plt.subplots(figsize=sizes["leakage"])
    for i, (s, v) in enumerate(zip(short, la.pr_auc)):
        y = len(short) - 1 - i
        ax.barh(y, v, height=0.62, color=ORANGE if i == 0 else BLUE, zorder=2)
        ax.text(v + 0.012, y, f"{v:.3f}", va="center", fontsize=9.5 * K, color=INK)
    ax.axvline(test_rate, color=INK2, lw=1, ls=(0, (4, 3)), zorder=1)
    ax.text(test_rate + 0.006, -0.8, f"no-skill {test_rate:.2f}", fontsize=8.5 * K, color=INK2, va="center")
    ax.set_yticks(range(len(short)))
    ax.set_yticklabels(list(reversed(short)))
    ax.set_xlim(0, 1.07)
    ax.set_ylim(-1.1, len(short) - 0.4)
    ax.set_xlabel("PR-AUC on the 2022–2024 test period")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[Patch(color=ORANGE, label="Same-month label (not a forecast)"),
                       Patch(color=BLUE, label="Honest next-month forecast")],
              frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, fontsize=9 * K)
    save(fig, "fig_leakage")

    # -------------------------------------------------------------- 3. calibration (reliability) diagram
    fig, ax = plt.subplots(figsize=sizes["calibration"])
    ax.plot([0, 1], [0, 1], color=GREY, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.text(0.70, 0.55, "perfect calibration", color=INK2, fontsize=8.5 * K, rotation=40, ha="center", va="center")
    big = cal.n >= 30
    ax.scatter(cal.mean_predicted[big], cal.observed_rate[big], s=(30 + cal.n[big] / 3) * K, color=BLUE, ec="white", lw=1.2, zorder=3)
    ax.scatter(cal.mean_predicted[~big], cal.observed_rate[~big], s=50 * K, facecolor="white", edgecolor=BLUE, lw=1.8, zorder=3)
    for _, r in cal[~big].iterrows():
        below = r.observed_rate < r.mean_predicted
        ax.annotate(f"n = {int(r.n)}", (r.mean_predicted, r.observed_rate), xytext=(0, (-15 if below else 9) * K),
                    textcoords="offset points", fontsize=8 * K, color=INK2, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted probability (bin)")
    ax.set_ylabel("Observed escalation rate (bin)")
    ax.text(0.03, 0.95, f"ECE = {metrics['ece_test']:.3f}\nfilled = 30+ rows; hollow = fewer", fontsize=8.5 * K,
            color=INK2, va="top")
    save(fig, "fig_calibration")

    # -------------------------------------------------------------- 4. drift: escalation rate by year
    fig, ax = plt.subplots(figsize=sizes["drift"])
    cols = [ORANGE if y >= 2022 else BLUE for y in yr.index]
    ax.bar(yr.index.astype(str), yr.values * 100, color=cols, width=0.7, zorder=2)
    for y in (2016, 2021, 2022, 2023):
        ax.text(str(y), yr[y] * 100 + 1.2, f"{yr[y] * 100:.0f}%", ha="center", fontsize=9 * K, color=INK)
    ax.set_ylabel("Escalating counties (%)")
    ax.set_ylim(0, 58)
    ax.grid(axis="x", visible=False)
    ax.tick_params(axis="x", labelsize=9 * K)
    ax.legend(handles=[Patch(color=BLUE, label="Training and validation years"),
                       Patch(color=ORANGE, label="Test years (2022–2024)")],
              frameon=False, loc="upper left", fontsize=9 * K)
    save(fig, "fig_drift")

    # -------------------------------------------------------------- 5. per-region recall and false-alarm rate
    fig, ax = plt.subplots(figsize=sizes["region"])
    ys = np.arange(len(rg))
    for y, (_, r) in zip(ys, rg.iterrows()):
        ax.plot([r.false_alarm_rate, r.recall], [y, y], color="#c3c2b7", lw=1.5 * K, zorder=1)
    ax.scatter(rg.recall, ys, s=70 * K, color=BLUE, ec="white", lw=1.4, zorder=3, label="Recall (escalations caught)")
    ax.scatter(rg.false_alarm_rate, ys, s=70 * K, color=ORANGE, ec="white", lw=1.4, zorder=3,
               label="False-alarm rate (quiet months flagged)")
    for y, (_, r) in zip(ys, rg.iterrows()):
        if r.recall >= 0.995 and r.false_alarm_rate >= 0.995:      # both markers coincide at 1.00
            ax.annotate("both 1.00", (1.0, y), xytext=(-10 * K, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=8.5 * K, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{r.region} ({int(r.counties)})" for _, r in rg.iterrows()])
    ax.set_xlim(0.55, 1.04)
    ax.set_xlabel("Rate at the tuned threshold, 2022–2024 (counties in brackets)")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower center", fontsize=9 * K, bbox_to_anchor=(0.45, 1.0), ncol=1)
    save(fig, "fig_region")


build(1.0, "", {"results": (7.2, 3.3), "leakage": (7.2, 3.1), "calibration": (4.6, 4.2),
                "drift": (7.2, 3.0), "region": (7.2, 3.5)})
build(1.45, "_s", {"results": (9.6, 4.0), "leakage": (9.6, 4.0), "calibration": (5.6, 5.0),
                   "drift": (9.0, 3.6), "region": (9.2, 4.4)})
