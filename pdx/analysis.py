"""Everything the advisor shows, computed once: model and its export, design effects with intervals, effects by
political context, the evaluation gap, country evidence gaps, case vectors for similarity search, and the live cases
added to Participedia since the capstone snapshot, scored by the model."""
import json
import math

import numpy as np
import pandas as pd

from . import config as C
from . import features as FT
from . import model as M

FREEDOM_BANDS = [("Less free (score under 50)", 0, 50), ("Partly free (50 to 79)", 50, 80), ("Free (80 and above)", 80, 101)]


def load():
    d = pd.read_csv(C.CAPSTONE, low_memory=False)
    idx = pd.read_csv(C.RAW / "live_index.csv")
    cases = idx[idx.type == "case"]
    comp = {FT.norm(t): (c if isinstance(c, str) else "") for t, c in zip(cases.title, cases.completeness)}
    ids = {FT.norm(t): int(i) for t, i in zip(cases.title, cases.id)}
    rec = FT.from_capstone(d, comp)
    for r in rec:
        r["id"] = ids.get(FT.norm(r["title"]))
    # country indicators for live cases: the capstone's values per country
    ctx = {}
    for r in rec:
        if r["country"] and r["country"] not in ctx and not any(pd.isna(v) for v in (r["gini"], r["freedom"], r["internet"])):
            ctx[r["country"]] = (float(r["gini"]), float(r["freedom"]), float(r["internet"]))
    live = []
    p = C.RAW / "live_new_cases.jsonl"
    if p.exists():
        for line in p.open(encoding="utf-8"):
            a = json.loads(line)
            if a.get("published") in (False, "false") or a.get("hidden") in (True, "true"):
                continue
            o = FT.from_live(a, ctx)
            o["completeness"] = (cases.set_index("id").completeness.get(int(a["id"])) if int(a["id"]) in set(cases.id) else "") or ""
            o["posted"] = str(cases.set_index("id").post_date.get(int(a["id"]), ""))[:10]
            live.append(o)
    return d, rec, live, idx


MACRO_COLS = {"gini": "gini_index_model_input", "freedom": "freedom_score_model_input", "internet": "internet_pct_model_input"}
RECRUIT = ["stratified", "random", "captive", "appointment", "election"]


def context_fit(d, y):
    """The manuscript's boundary-condition analysis (Zakaria & Carson, Table 8): recruitment mix within equal-count
    thirds (tertiles) of each national indicator, which is how Table 8 reproduces; plus documented impact by
    recruitment within each tier, to test whether context changes what works or only what is used."""
    out = {"cuts": {}, "mix": {}, "impact": {}}
    ok = d.recruitment_method.notna()
    for k, c in MACRO_COLS.items():
        t = pd.qcut(d[c].rank(method="first"), 3, labels=["Low", "Mid", "High"])
        out["cuts"][k] = [round(float(d.loc[t == "Low", c].max()), 2), round(float(d.loc[t == "Mid", c].max()), 2)]
        out["mix"][k], out["impact"][k] = {}, {}
        for lvl in ["Low", "Mid", "High"]:
            m = (t == lvl) & ok
            out["mix"][k][lvl] = {"n": int(m.sum()), **{r: round(float((d.loc[m, "recruitment_method"] == r).mean()), 4) for r in RECRUIT}}
            out["impact"][k][lvl] = {r: [round(float(y[m & (d.recruitment_method == r)].mean()), 3), int((m & (d.recruitment_method == r)).sum())]
                                     for r in RECRUIT if (m & (d.recruitment_method == r)).sum() >= 8}
    return out


def alignment(X, y):
    """Checks that relate the impact model to the manuscript: what national context adds, and whether context
    changes which recruitment method goes with documented impact."""
    from sklearn.model_selection import cross_val_score
    pipe = lambda: M.make_pipeline(M.SimpleImputer(strategy="median"), M.StandardScaler(), M.LogisticRegression(C=M.LR_C, max_iter=5000))
    macro = ["gini", "freedom", "internet"]
    cv = lambda Z: float(cross_val_score(pipe(), Z, y, cv=5, scoring="roc_auc").mean())
    Xi = X.copy()
    fr, gi = Xi.freedom.fillna(Xi.freedom.median()), Xi.gini.fillna(Xi.gini.median())
    for r in ["stratified", "random", "captive", "appointment"]:
        Xi[f"{r}_x_lowfree"] = Xi[f"recruitment_method={r}"] * (fr < 70)
        Xi[f"{r}_x_highineq"] = Xi[f"recruitment_method={r}"] * (gi > 40)
    return {"auc_all": cv(X), "auc_no_macro": cv(X.drop(columns=macro)), "auc_macro_only": cv(X[macro]), "auc_interactions": cv(Xi)}


def run():
    d, rec, live, idx = load()
    X = FT.matrix(rec)
    y = pd.Series([r["impact"] for r in rec])
    tr = np.array([r["split"] == "train" for r in rec])
    cols = FT.columns()
    feats = [c[0] for c in cols]

    metrics, _ = M.evaluate(X[tr], y[tr], X[~tr], y[~tr])      # also sets the regularization from cross-validation
    eff, gb = M.effects(X, y)                                   # final model on all cases + bootstrap intervals
    p_all = gb.predict_proba(X)[:, 1]

    # effects by political context (same model, averaged within each band)
    het = []
    fr = X["freedom"]
    for name, lo, hi in FREEDOM_BANDS:
        m = (fr >= lo) & (fr < hi)
        sub = X[m]
        row = {"band": name, "n": int(m.sum()), "impact_rate": float(y[m].mean())}
        row["effects"] = {l[0]: round(M.effect(gb, sub, l), 4) for l in M.levers()}
        het.append(row)

    # documentation reference: a well-documented entry (median of complete entries) for advisor scoring
    doc = [c[0] for c in cols if c[1] == "doc"]
    ref = X[X.doc_complete == 1][doc].median().to_dict()
    ref["doc_complete"], ref["doc_stub"] = 1.0, 0.0
    X_full = X.copy()
    for k, v in ref.items():
        X_full[k] = v
    base_full = float(gb.predict_proba(X_full)[:, 1].mean())   # the archive's designs, all written up in full

    # evaluation gap
    def rate_by(field, top=None):
        out = []
        for name, kind, f, lab in cols:
            if f != field:
                continue
            m = X[name] == 1
            if m.sum() >= 25:
                out.append({"label": lab, "n": int(m.sum()), "evaluated": float(X.loc[m, "formal_evaluation"].mean()), "impact": float(y[m].mean())})
        return sorted(out, key=lambda r: -r["evaluated"])[:top] if top else sorted(out, key=lambda r: -r["evaluated"])
    ev_m = X.formal_evaluation == 1
    evalgap = {"overall": float(ev_m.mean()), "impact_if_evaluated": float(y[ev_m].mean()), "impact_if_not": float(y[~ev_m].mean()),
               "by_method": rate_by("method_types"), "by_organizer": rate_by("organizer_types"), "by_funder": rate_by("funder_types")}

    # country evidence gaps: observed vs expected documented impact
    cdf = pd.DataFrame({"country": [r["country"] for r in rec], "y": y, "p": p_all})
    g = cdf.dropna(subset=["country"]).groupby("country").agg(n=("y", "size"), obs=("y", "mean"), exp=("p", "mean")).reset_index()
    g = g[g.n >= 15].assign(gap=lambda t: t.obs - t.exp).sort_values("n", ascending=False)
    countries = {}
    for r in rec:
        c = r["country"]
        if c and c not in countries and not any(v is None or (isinstance(v, float) and math.isnan(v)) for v in (r["gini"], r["freedom"], r["internet"])):
            countries[c] = {"gini": round(float(r["gini"]), 1), "freedom": round(float(r["freedom"]), 0), "internet": round(float(r["internet"]), 1)}
    cn = cdf.groupby("country").agg(n=("y", "size"), rate=("y", "mean"))
    for c in countries:
        if c in cn.index:
            countries[c].update(n=int(cn.loc[c, "n"]), rate=round(float(cn.loc[c, "rate"]), 3))

    # cases for similarity search: active design/context features, outcome, link
    act = [i for i, c in enumerate(cols) if c[1] in ("design", "context") and "=" in c[0] or c[0] in FT.BINARY]
    case_rows = []
    for i, r in enumerate(rec):
        xi = X.iloc[i]
        case_rows.append({"t": r["title"], "c": r["country"] or "", "y": r["start_year"], "id": r["id"], "i": int(r["impact"]),
                          "f": [j for j in act if xi.iloc[j] == 1], "lat": _r(r["lat"]), "lon": _r(r["lon"]),
                          "ch": r["change_types"], "d": (r["description"] or "")[:180]})

    # live cases since the snapshot
    live_out, live_summary = [], {}
    if live:
        XL = FT.matrix(live)
        XL_ref = XL.copy()
        pl = gb.predict_proba(XL)[:, 1]
        for o, p in zip(live, pl):
            live_out.append({"t": o["title"], "c": o["country"] or "", "id": o["id"], "posted": o.get("posted", ""), "y": o["start_year"],
                             "comp": o["completeness"], "imp": o["impact"], "p": round(float(p), 3),
                             "m": [FT.MULTI["method_types"][1].get(k, k) for k in o["method_types"]][:3], "d": (o["description"] or "")[:180]})
        live_out.sort(key=lambda r: r["posted"], reverse=True)
        lm = pd.DataFrame(live_out)
        live_summary = {"n": len(live_out), "with_impact_field": int(lm.imp.notna().sum()), "impact_rate": float(lm.imp.dropna().mean()) if lm.imp.notna().any() else None,
                        "complete": int((lm.comp == "complete").sum()), "stub": int((lm.comp == "stub").sum())}
        # method mix: snapshot vs new
        mix = []
        for code, lab in FT.MULTI["method_types"][1].items():
            a = float(X[f"method_types={code}"].mean())
            b = float(XL[f"method_types={code}"].mean())
            mix.append({"label": lab, "before": a, "new": b})
        live_summary["method_mix"] = sorted(mix, key=lambda r: -(r["before"] + r["new"]))[:10]

    # trends: share of cases by start decade for the main method types
    yrs = pd.Series([r["start_year"] for r in rec])
    trend = []
    for dec in range(1990, 2030, 5):
        m = (yrs >= dec) & (yrs < dec + 5)
        if m.sum() >= 20:
            trend.append({"from": dec, "n": int(m.sum()), "impact": float(y[m].mean()), "evaluated": float(X.loc[m, "formal_evaluation"].mean()),
                          "online": float(((X.loc[m, "facetoface_online_or_both=online"] + X.loc[m, "facetoface_online_or_both=both"]) > 0).mean()),
                          "deliberative": float(X.loc[m, "method_types=deliberative"].mean())})

    schema = {"multi": {f: {"kind": k, "options": m} for f, (k, m) in FT.MULTI.items()}, "single": {f: {"kind": k, "options": m} for f, (k, m) in FT.SINGLE.items()},
              "binary": {f: {"kind": k, "label": lab} for f, (k, lab) in FT.BINARY.items()}, "numeric": {f: {"kind": k, "label": lab} for f, (k, lab) in FT.NUMERIC.items()}}
    medians = X.median(numeric_only=True)
    payload = {
        "meta": {"cases": len(rec), "countries": int(cdf.country.nunique()), "live_cases": int((idx.type == "case").sum()),
                 "years": [int(yrs.min()), int(yrs.max())], "impact_rate": float(y.mean()), "impact_rate_full": base_full, "snapshot": "capstone (ALY6980, 2024)",
                 "fetched": pd.Timestamp((C.RAW / "live_index.csv").stat().st_mtime, unit="s").strftime("%B %Y")},
        "metrics": metrics, "effects": eff.round(4).to_dict("records"), "hetero": het, "evalgap": evalgap,
        "countries": countries, "country_gaps": g.round(3).to_dict("records"), "cases": case_rows, "live": live_out, "live_summary": live_summary,
        "trend": trend, "schema": schema, "features": [{"name": c[0], "kind": c[1], "field": c[2], "label": c[3]} for c in cols],
        "model": M.export(gb, feats), "doc_ref": {k: round(float(v), 4) for k, v in ref.items()},
        "fit": context_fit(d, y), "alignment": alignment(X, y),
        "defaults": {"log_participants": round(float(medians["log_participants"]), 3), "start_year": 2026},
        "fill": {k: round(float(v), 4) for k, v in medians.items()},
    }
    C.OUT.mkdir(parents=True, exist_ok=True)
    (C.OUT / "payload.json").write_text(json.dumps(payload, default=_j, separators=(",", ":")))
    eff.sort_values("effect").to_csv(C.OUT / "design_effects.csv", index=False)
    pd.DataFrame(live_out).to_csv(C.OUT / "live_new_cases_scored.csv", index=False)
    print(json.dumps(metrics, indent=1))
    print(f"live cases: {len(live_out)}; effects: {len(eff)}; payload {len(json.dumps(payload, default=_j)) / 1e6:.2f} MB")
    return payload


def _r(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else round(float(v), 3)


def _j(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if math.isnan(o) else float(o)
    return str(o)
