"""A public CSV of every published Participedia case, for the Power BI dashboard to refresh from on its own.

Written by each run (the monthly cloud refresh included) to docs/data/participedia_cases.csv, which GitHub Pages
serves at a fixed web address. Power BI's Web connector reads it with no gateway, so a scheduled refresh in the
Power BI service keeps the published dashboard current.

Built only from public data: each case's own record on participedia.net (fetched once, then again only when the case
is updated), plus the public per-country indicators in pdx/live_context.json. The column names match the dashboard's
original table, so pointing that table at this file keeps every visual working:
    id, title, description, country, city, start_date, end_date, method_types_1, method_types_2,
    recruitment_method, gini_index, freedom_score, internet_pct
and adds latitude, longitude, completeness, posted_date, updated_date, documented_impact, source, model_score, case_url.
"""
import json
import time

import pandas as pd

from . import config as C
from . import features as FT
from . import live

OUT = C.ROOT / "docs" / "data" / "participedia_cases.csv"
RECORDS = C.RAW / "case_records.jsonl"
METHOD = FT.MULTI["method_types"][1]
RECRUIT = FT.SINGLE["recruitment_method"][1] if "recruitment_method" in FT.SINGLE else {}


def _records(idx, limit=None, pause=0.6):
    """Every case's public record, cached; fetches only cases that are new or updated since the last fetch."""
    have = {}
    if RECORDS.exists():
        for line in RECORDS.open(encoding="utf-8"):
            a = json.loads(line)
            have[int(a["id"])] = a
    new_cases = C.RAW / "live_new_cases.jsonl"                     # records the advisor already fetched
    if new_cases.exists():
        for line in new_cases.open(encoding="utf-8"):
            a = json.loads(line)
            have.setdefault(int(a["id"]), a)
    todo = [int(r.id) for r in idx.itertuples()
            if int(r.id) not in have or str(have[int(r.id)].get("updated_date", ""))[:19] != str(r.updated_date)[:19]]
    if limit is not None:
        todo = todo[:limit]
    print(f"case records: {len(have):,} cached, {len(todo):,} to fetch")
    for i, cid in enumerate(todo):
        try:
            have[cid] = live.get(f"https://participedia.net/case/{cid}?returns=json")["article"]
        except Exception as e:
            print("skip", cid, e)
        if i % 100 == 0:
            print(f"  {i}/{len(todo)}")
        time.sleep(pause)
    with RECORDS.open("w", encoding="utf-8") as f:
        for a in have.values():
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return have


# Newer entries give the country in the contributor's language; use the English names the archive mostly uses
# (and pdx/live_context.json uses), so each country is counted once and gets its indicators.
COUNTRY = {"Argentinien": "Argentina", "Belgique": "Belgium", "Bolivien": "Bolivia", "Brasil": "Brazil",
           "Czech Republic": "Czechia", "Tchéquia": "Czechia", "Deutschland": "Germany", "Germania": "Germany",
           "España": "Spain", "Italia": "Italy", "Mexiko": "Mexico", "Nederland": "Netherlands", "Polska": "Poland",
           "Regno Unito": "United Kingdom", "Republic of North Macedonia": "North Macedonia", "Suisse": "Switzerland",
           "Türkiye": "Turkey", "中国": "China", "Myanmar (Burma)": "Myanmar"}


def _date(v):
    if v is None or (isinstance(v, float) and v != v) or str(v).strip().lower() in ("", "null", "nan", "none", "nat"):
        return ""
    return str(v)[:10]


def export(payload, limit=None):
    idx = pd.read_csv(C.RAW / "live_index.csv")
    idx = idx[idx.type == "case"].copy()
    rec = _records(idx, limit=limit)
    ctx = json.loads((C.ROOT / "pdx" / "live_context.json").read_text(encoding="utf-8"))["countries"]
    snapshot_ids = {int(c["id"]) for c in payload["cases"] if c.get("id") is not None}
    score = {int(r["id"]): r.get("p") for r in payload["live"] if r.get("id") is not None}

    rows = []
    for r in idx.itertuples():
        cid = int(r.id)
        a = rec.get(cid)
        if a is None or a.get("published") in (False, "false") or a.get("hidden") in (True, "true"):
            continue
        country = (a.get("country") or (r.country if isinstance(r.country, str) else "")).strip()
        country = COUNTRY.get(country, country)
        g = ctx.get(country)
        methods = [METHOD.get(m, m) for m in (a.get("method_types") or []) if isinstance(m, str)]
        ie = a.get("impact_evidence")
        rows.append({
            "id": cid, "title": a.get("title") or r.title, "description": (a.get("description") or "").strip(),
            "country": country, "city": a.get("city") or "",
            "start_date": _date(a.get("start_date")), "end_date": _date(a.get("end_date")),
            "method_types_1": methods[0] if methods else "", "method_types_2": methods[1] if len(methods) > 1 else "",
            "recruitment_method": RECRUIT.get(a.get("recruitment_method"), a.get("recruitment_method") or ""),
            "gini_index": g[0] if g else None, "freedom_score": g[1] if g else None, "internet_pct": g[2] if g else None,
            "latitude": a.get("latitude") if a.get("latitude") not in ("", None) else r.latitude,
            "longitude": a.get("longitude") if a.get("longitude") not in ("", None) else r.longitude,
            "completeness": a.get("completeness") or (r.completeness if isinstance(r.completeness, str) else ""),
            "posted_date": _date(a.get("post_date") or r.post_date), "updated_date": _date(a.get("updated_date") or r.updated_date),
            "documented_impact": 1 if ie in (True, "yes", "true", "True") else (0 if ie in (False, "no", "false", "False") else None),
            "source": "Capstone dataset" if cid in snapshot_ids else "Added since",
            "model_score": score.get(cid), "case_url": f"https://participedia.net/case/{cid}"})
    df = pd.DataFrame(rows).sort_values("id")
    df["documented_impact"] = df["documented_impact"].astype("Int64")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, encoding="utf-8")
    print(f"Power BI data: {len(df):,} published cases -> {OUT}")
    return df
