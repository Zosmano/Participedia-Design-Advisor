"""Participation Design Advisor, end to end.

    python run_all.py --live      # refresh Participedia's public case index and fetch cases added since the capstone
    python run_all.py             # model, design effects, evidence gap, live scoring -> outputs/ and the app
"""
import json
import sys

import pandas as pd

from pdx import analysis, config as C, live


def build(payload):
    tpl = (C.ROOT / "pdx" / "app_template.html").read_text(encoding="utf-8")
    body = tpl.replace("/*__DATA__*/null", json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=analysis._j).replace("</", "<\\/"))
    (C.OUT / "participation_design_advisor.html").write_text(body, encoding="utf-8")
    head, rest = body.split("<style>", 1)
    full = ('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n'
            + head + "<style>" + rest.replace("</style>", "</style></head><body>", 1) + "\n</body></html>")
    (C.OUT / "participation_design_advisor_local.html").write_text(full, encoding="utf-8")
    (C.ROOT / "docs").mkdir(exist_ok=True)
    (C.ROOT / "docs" / "index.html").write_text(full, encoding="utf-8")      # served by GitHub Pages (main branch, /docs)
    print(f"-> {C.OUT / 'participation_design_advisor.html'} ({len(body) / 1e6:.2f} MB)")


if __name__ == "__main__":
    if "--live" in sys.argv:
        live.index()
        live.fetch_new(pd.read_csv(C.CAPSTONE, usecols=["title"]).title.dropna())
    build(analysis.run())
    from pdx import figures
    figures.build()
