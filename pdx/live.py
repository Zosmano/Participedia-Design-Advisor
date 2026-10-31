"""Fetch the latest cases from Participedia's public JSON (no sign-in): the index of every entry, then the full record
of each case that is not yet in the capstone dataset. Polite: one request at a time with a pause."""
import json
import re
import time
import urllib.request

import pandas as pd

from . import config as C

H = {"User-Agent": "Mozilla/5.0 (research; participedia-design-advisor; non-commercial analysis)"}


def get(url, tries=4):
    for k in range(tries):
        try:
            return json.loads(urllib.request.urlopen(urllib.request.Request(url, headers=H), timeout=120).read())
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(3 * (k + 1))


def norm(t):
    return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


def index():
    """Every Participedia entry: id, type, completeness, title, dates."""
    j = get("https://participedia.net/search?resultType=map&returns=json")
    rows = [{k: r.get(k) for k in ("id", "type", "completeness", "title", "country", "updated_date", "post_date", "latitude", "longitude")}
            for r in j["results"]]
    df = pd.DataFrame(rows)
    df.to_csv(C.RAW / "live_index.csv", index=False)
    return df


def fetch_new(base_titles, pause=0.6):
    """Full records for live cases whose title is not in the capstone dataset."""
    idx = pd.read_csv(C.RAW / "live_index.csv")
    cases = idx[idx.type == "case"].copy()
    have = {norm(t) for t in base_titles}
    new = cases[~cases.title.map(norm).isin(have)]
    out, path = [], C.RAW / "live_new_cases.jsonl"
    done = set()
    if path.exists():
        done = {json.loads(l)["id"] for l in path.open(encoding="utf-8")}
    with path.open("a", encoding="utf-8") as f:
        for i, cid in enumerate(new.id):
            if int(cid) in done:
                continue
            try:
                a = get(f"https://participedia.net/case/{int(cid)}?returns=json")["article"]
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
            except Exception as e:
                print("skip", cid, e)
            if i % 25 == 0:
                print(f"  {i}/{len(new)}")
            time.sleep(pause)
    return len(new)
