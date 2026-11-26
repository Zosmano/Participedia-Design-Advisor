"""Monthly live refresh, run by GitHub Actions without the capstone dataset.

It reads Participedia's public case index, fetches each published case not in the capstone snapshot, scores it
with the model already exported in outputs/payload.json, and updates the parts of the advisor that track the live
archive: the "New on Participedia" tab, the live case count and the date read. Everything learned from the capstone
(the model, design effects, evidence gap, context analysis) is left exactly as the last full run wrote it.

    python run_all.py --refresh
"""
import json
import math

import numpy as np
import pandas as pd

from . import analysis
from . import config as C
from . import live


def exported_predict(model):
    """The advisor's logistic model from its exported numbers: missing -> median, standardize, weighted sum."""
    med, mu, sd, coef = (np.array(model[k], dtype=float) for k in ("med", "mu", "sd", "coef"))

    def predict(X):
        A = X[model["features"]].to_numpy(dtype=float)
        A = np.where(np.isnan(A), med, A)
        z = (A - mu) / np.where(sd == 0, 1.0, sd) @ coef + model["b0"]
        return 1.0 / (1.0 + np.exp(-z))
    return predict


def run():
    payload = json.loads((C.OUT / "payload.json").read_text(encoding="utf-8"))
    context = json.loads(analysis.LIVE_CONTEXT.read_text(encoding="utf-8"))
    ctx = {k: tuple(v) for k, v in context["countries"].items()}

    C.RAW.mkdir(parents=True, exist_ok=True)      # a fresh checkout (or GitHub's runner) has no data folder yet
    idx = live.index()
    snapshot_titles = [c["t"] for c in payload["cases"] if c.get("t")]
    n_new = live.fetch_new(snapshot_titles)
    print(f"Participedia lists {int((idx.type == 'case').sum()):,} cases; {n_new:,} are not in the snapshot")

    recs = analysis.load_live(idx, ctx)
    out, summary = analysis.score_live(recs, exported_predict(payload["model"]), context["method_before"])
    before = payload.get("live_summary", {}).get("n", 0)
    payload["live"], payload["live_summary"] = out, summary
    payload["meta"]["live_cases"] = int((idx.type == "case").sum())
    payload["meta"]["fetched"] = pd.Timestamp.now(tz="UTC").strftime("%B %Y")

    (C.OUT / "payload.json").write_text(json.dumps(payload, default=analysis._j, separators=(",", ":")), encoding="utf-8")
    pd.DataFrame(out).to_csv(C.OUT / "live_new_cases_scored.csv", index=False)
    print(f"new cases scored: {len(out):,} (was {before:,})")
    return payload


def _check_equal(a, b, tol=0.0015):
    """True if two scored-case lists agree (same ids, scores within rounding)."""
    pa, pb = {r["id"]: r["p"] for r in a}, {r["id"]: r["p"] for r in b}
    return pa.keys() == pb.keys() and all(math.isclose(pa[k], pb[k], abs_tol=tol) for k in pa)
