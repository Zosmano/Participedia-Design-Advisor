"""A public CSV of every Participedia case, for the Power BI dashboard to refresh from on its own.

Written by each run (the monthly cloud refresh included) to docs/data/participedia_cases.csv, which GitHub Pages
serves at a fixed web address. Power BI's Web connector reads it with no gateway, so a scheduled refresh in the
Power BI service keeps the published dashboard current.

One row per case on Participedia's public index. Cases in the capstone snapshot carry what the advisor already
publishes about them (start year, methods, documented impact); cases added since carry the same fields from their
own public records, plus the advisor's model score. Country indicators are the public per-country values in
pdx/live_context.json. Nothing here comes from the private capstone file beyond what the advisor page shows.
"""
import json

import pandas as pd

from . import config as C
from . import features as FT

OUT = C.ROOT / "docs" / "data" / "participedia_cases.csv"


def export(payload):
    idx = pd.read_csv(C.RAW / "live_index.csv")
    idx = idx[idx.type == "case"].copy()
    ctx = json.loads((C.ROOT / "pdx" / "live_context.json").read_text(encoding="utf-8"))["countries"]
    feats = payload["features"]
    method_of = {j: f["label"] for j, f in enumerate(feats) if f["name"].startswith("method_types=")}

    snap = {}
    for c in payload["cases"]:
        if c.get("id") is not None:
            snap[int(c["id"])] = {"start_year": c.get("y"), "methods": "; ".join(method_of[j] for j in c["f"] if j in method_of),
                                  "documented_impact": c.get("i"), "model_score": None, "in_capstone": "Capstone dataset"}
    new = {int(r["id"]): {"start_year": r.get("y"), "methods": "; ".join(r.get("m") or []), "documented_impact": r.get("imp"),
                          "model_score": r.get("p"), "in_capstone": "Added since"} for r in payload["live"] if r.get("id") is not None}

    rows = []
    for r in idx.itertuples():
        cid = int(r.id)
        extra = snap.get(cid) or new.get(cid) or {"start_year": None, "methods": "", "documented_impact": None,
                                                    "model_score": None, "in_capstone": "Not scored (unpublished or renamed)"}
        g = ctx.get(r.country) if isinstance(r.country, str) else None
        rows.append({"case_id": cid, "title": r.title, "country": r.country if isinstance(r.country, str) else "",
                     "latitude": r.latitude, "longitude": r.longitude, "completeness": r.completeness if isinstance(r.completeness, str) else "",
                     "posted_date": str(r.post_date)[:10] if pd.notna(r.post_date) else "",
                     "updated_date": str(r.updated_date)[:10] if pd.notna(r.updated_date) else "",
                     "source": extra["in_capstone"], "start_year": extra["start_year"], "methods": extra["methods"],
                     "documented_impact": extra["documented_impact"], "model_score": extra["model_score"],
                     "gini": g[0] if g else None, "freedom_score": g[1] if g else None, "internet_pct": g[2] if g else None,
                     "case_url": f"https://participedia.net/case/{cid}"})
    df = pd.DataFrame(rows).sort_values("case_id")
    df["start_year"] = pd.to_numeric(df["start_year"], errors="coerce").astype("Int64")
    df["documented_impact"] = pd.to_numeric(df["documented_impact"], errors="coerce").astype("Int64")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Power BI data: {len(df):,} cases -> {OUT}")
    return df
