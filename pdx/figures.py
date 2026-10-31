"""Static charts for the README (GitHub shows images, not the interactive app), drawn from outputs/payload.json
in the advisor's palette: aubergine, emerald, rose, old gold on porcelain."""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import config as C

INK, INK2, MUTED, LINE, BG = "#18172B", "#4A4A63", "#73748A", "#D7DAE3", "#F3F4F7"
DAIS, EMERALD, ROSE, GOLD, PLUM = "#2B1B3D", "#1D7A5F", "#B4475A", "#B8862B", "#6B4A8C"
FIG = C.ROOT / "docs" / "figures"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.edgecolor": LINE, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200})


def frame(title, sub, w=11, h=6):
    fig = plt.figure(figsize=(w, h))
    fig.text(0.03, 0.95, title, fontsize=17, fontweight="bold", color=DAIS, va="top", family="Georgia")
    fig.text(0.03, 0.885, sub, fontsize=10, color=MUTED, va="top")
    fig.text(0.03, 0.02, "Participation Design Advisor · data: Participedia (CC BY-SA 4.0), World Bank, Freedom House", fontsize=8, color=MUTED)
    return fig


def clean(ax, grid="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if grid in ("x", "y", "both"):
        ax.grid(axis=grid, color=LINE, lw=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def pct(v):
    return f"{v * 100:.0f}%"


def effects(p):
    e = sorted([x for x in p["effects"] if x["n"] >= 60], key=lambda x: x["effect"])
    rows = e[:6] + e[-12:]
    fig = frame("What goes with documented impact", "Average change in the chance an entry documents impact when each design choice is switched on, documentation and context held fixed. "
                "90% intervals from 200 refits; choices with 60+ cases.", 11, 7.2)
    ax = fig.add_axes([0.36, 0.1, 0.6, 0.74])
    for i, x in enumerate(rows):
        sig = x["lo"] > 0 or x["hi"] < 0
        col = (EMERALD if x["effect"] > 0 else ROSE) if sig else MUTED
        ax.plot([x["lo"] * 100, x["hi"] * 100], [i, i], color=col, lw=3, alpha=0.45, solid_capstyle="round")
        ax.plot(x["effect"] * 100, i, "o", color=col, ms=8)
        ax.text(x["hi"] * 100 + 0.6, i, f"{x['effect'] * 100:+.1f}", va="center", fontsize=9, color=col)
    area = {"method_types": "Method", "participants_interactions": "Interaction", "learning_resources": "Learning", "decision_methods": "Decision",
            "insights_outcomes": "Results shared via", "tools_techniques_types": "Tool", "approaches": "Approach", "targeted_participants": "Outreach to",
            "organizer_types": "Organizer", "funder_types": "Funder", "public_spectrum": "Spectrum", "recruitment_method": "Recruitment",
            "open_limited": "Access", "facetoface_online_or_both": "Format", "facilitator_training": "Facilitation", "time_limited": "Timing"}
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([(area[x["field"]] + ": " if x["field"] in area else "") + x["label"] for x in rows], fontsize=9.5)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_xlabel("Percentage points")
    clean(ax)
    fig.savefig(FIG / "design_effects.png", bbox_inches="tight")
    plt.close(fig)


def evidence(p):
    g = p["evalgap"]
    fig = frame("The evidence gap", f"Only {pct(g['overall'])} of processes are formally evaluated. Evaluated cases document impact {pct(g['impact_if_evaluated'])} of the time, "
                f"against {pct(g['impact_if_not'])} for the rest.", 11, 6)
    ax = fig.add_axes([0.07, 0.12, 0.22, 0.66])
    ax.bar(["Evaluated", "Not evaluated"], [g["impact_if_evaluated"] * 100, g["impact_if_not"] * 100], color=[EMERALD, MUTED], width=0.6)
    for i, v in enumerate([g["impact_if_evaluated"], g["impact_if_not"]]):
        ax.text(i, v * 100 + 2, pct(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Document impact (%)")
    ax.set_ylim(0, 100)
    clean(ax, "y")
    m = g["by_method"]
    ax2 = fig.add_axes([0.58, 0.12, 0.38, 0.66])
    ys = np.arange(len(m))[::-1]
    ax2.barh(ys, [r["evaluated"] * 100 for r in m], color=PLUM, height=0.62)
    for y, r in zip(ys, m):
        ax2.text(r["evaluated"] * 100 + 0.8, y, f"{pct(r['evaluated'])} of {r['n']}", va="center", fontsize=8.5, color=INK2)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([r["label"] for r in m], fontsize=9)
    ax2.set_xlabel("Formally evaluated (%)")
    ax2.set_title("Share formally evaluated, by method", fontsize=10.5, loc="left", color=INK)
    clean(ax2)
    fig.savefig(FIG / "evidence_gap.png", bbox_inches="tight")
    plt.close(fig)


def context(p):
    G = p["country_gaps"]
    fig = frame("Where impact gets documented", "Observed vs expected documented impact by country (15+ cases); above the line = more than design and documentation predict.", 11, 6.4)
    ax = fig.add_axes([0.08, 0.12, 0.42, 0.7])
    hi = max(max(r["obs"], r["exp"]) for r in G) * 1.08
    ax.plot([0, hi * 100], [0, hi * 100], ls="--", color=GOLD, lw=1.3)
    mx = max(r["n"] for r in G)
    for r in G:
        col = EMERALD if r["gap"] >= 0 else ROSE
        ax.scatter(r["exp"] * 100, r["obs"] * 100, s=30 + 500 * (r["n"] / mx), color=col, alpha=0.5, edgecolor=col)
        if r["n"] >= 60 or abs(r["gap"]) > 0.12:
            ax.text(r["exp"] * 100 + 1, r["obs"] * 100 + 1, r["country"], fontsize=8.5, color=INK2)
    ax.set_xlabel("Expected (%)")
    ax.set_ylabel("Observed (%)")
    clean(ax, "both")
    T = p["trend"]
    ax2 = fig.add_axes([0.58, 0.12, 0.38, 0.7])
    x = [f"{t['from']}–{str(t['from'] + 4)[2:]}" for t in T]
    for k, lab, col in [("deliberative", "Deliberative", PLUM), ("online", "Online or hybrid", GOLD), ("impact", "Documents impact", EMERALD), ("evaluated", "Formally evaluated", ROSE)]:
        ax2.plot(x, [t[k] * 100 for t in T], "-o", color=col, lw=2.2, ms=4, label=lab)
    ax2.set_ylabel("Share of cases (%)")
    ax2.set_title("How participation has changed", fontsize=10.5, loc="left")
    ax2.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax2.tick_params(axis="x", labelsize=8, rotation=30)
    clean(ax2, "y")
    fig.savefig(FIG / "context.png", bbox_inches="tight")
    plt.close(fig)


def fit(p):
    F = p["fit"]
    rec = [("stratified", "Stratified random", EMERALD), ("random", "Random", "#5FA88F"), ("appointment", "Appointment", GOLD), ("captive", "Captive", ROSE), ("election", "Election", PLUM)]
    fig = frame("Context sets boundary conditions on recruitment", "Recruitment method by thirds of the archive on each national indicator (the manuscript's Table 8, reproduced). "
                "Context shapes what is used, not whether it works.", 11, 5.8)
    names = {"gini": "Income inequality (Gini)", "freedom": "Freedom House score", "internet": "Internet use"}
    tiers = {"gini": ["Lower", "Middle", "Higher"], "freedom": ["Lower", "Middle", "Higher"], "internet": ["Lower", "Middle", "Higher"]}
    for j, k in enumerate(["gini", "freedom", "internet"]):
        ax = fig.add_axes([0.09 + j * 0.32, 0.16, 0.22, 0.62])
        left = np.zeros(3)
        for code, lab, col in rec:
            v = np.array([F["mix"][k][t][code] * 100 for t in ["Low", "Mid", "High"]])
            ax.barh([2, 1, 0], v, left=left, color=col, height=0.6, label=lab)
            for y, (l, w) in zip([2, 1, 0], zip(left, v)):
                if w >= 9:
                    ax.text(l + w / 2, y, f"{w:.0f}", ha="center", va="center", fontsize=8, color="white")
            left += v
        c = F["cuts"][k]
        unit = "%" if k == "internet" else ""
        ax.set_yticks([2, 1, 0])
        ax.set_yticklabels([f"{tiers[k][0]}\n≤{c[0]:g}{unit}", f"{tiers[k][1]}", f"{tiers[k][2]}\n>{c[1]:g}{unit}"], fontsize=8.5)
        ax.set_title(names[k], fontsize=10.5, loc="left")
        ax.set_xlim(0, 100)
        ax.set_xticks([])
        clean(ax, "none")
    fig.legend(*fig.axes[0].get_legend_handles_labels(), loc="lower center", ncol=5, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.05))
    fig.savefig(FIG / "context_fit.png", bbox_inches="tight")
    plt.close(fig)


def new_cases(p):
    s = p["live_summary"]
    M = s["method_mix"]
    fig = frame("New on Participedia", f"{s['n']} published cases not in the 2,260-case dataset: method mix compared.", 11, 5.6)
    ax = fig.add_axes([0.3, 0.12, 0.64, 0.68])
    ys = np.arange(len(M))[::-1]
    ax.barh(ys + 0.2, [m["before"] * 100 for m in M], height=0.38, color=MUTED, label="2,260-case dataset")
    ax.barh(ys - 0.2, [m["new"] * 100 for m in M], height=0.38, color=GOLD, label="Added since")
    ax.set_yticks(ys)
    ax.set_yticklabels([m["label"] for m in M], fontsize=9.5)
    ax.set_xlabel("Share of cases (%)")
    ax.legend(frameon=False, loc="lower right")
    clean(ax)
    fig.savefig(FIG / "new_cases.png", bbox_inches="tight")
    plt.close(fig)


def build():
    FIG.mkdir(parents=True, exist_ok=True)
    p = json.loads((C.OUT / "payload.json").read_text())
    for f in (effects, evidence, context, fit, new_cases):
        f(p)
    print("figures ->", FIG)


if __name__ == "__main__":
    build()
