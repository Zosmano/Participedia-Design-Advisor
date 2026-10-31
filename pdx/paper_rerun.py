"""Rerun of the manuscript's classification (Zakaria & Carson): the 65/35 CatBoost + XGBoost soft-voting ensemble
predicting the five method families (final_target_group), on the manuscript's own train/test split.

Variants
  A  as published: text corpus (with linked method titles), recruitment_complexity (missing = 0), tool_title_len,
     Gini, Freedom House, internet, is_government, is_local, is_online — on all 2,260 cases
  B  leak-free, all cases: method titles stripped from the text, tool_title_len dropped, recruitment as a category
     with 'not recorded' as its own value
  C  leak-free, the 1,402 cases with a recorded method type (the question the paper asks)
  D  as C without the three national indicators (what context adds)
Each is run over 5 seeds; baselines and importances are computed on the same test cases."""
import json
import re

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from . import config as C

MACRO = ["gini_index_model_input", "freedom_score_model_input", "internet_pct_model_input"]
FLAGS = ["is_government", "is_local", "is_online"]
CLASSES = ["Collaborative/Governance", "Community/Informal", "Deliberative/Dialogic", "Other/Direct", "Public/Consultative"]
SEEDS = [0, 1, 2, 3, 4]
W_CAT = 0.65


def strip_titles(s):
    """Remove the linked-method tuples, e.g. (4232,method,"Online Deliberation"), from the front of the corpus."""
    return re.sub(r'^\s*(\(\d+,(?:method|case|organization),(?:"[^"]*"|[^)]*)\)\s*)+', "", str(s)).strip()


def frame(d, variant):
    X = pd.DataFrame(index=d.index)
    if variant == "A":
        X["text"] = d.combined_corpus.fillna("").astype(str)
        X["recruitment_complexity"] = d.recruitment_complexity
        X["tool_title_len"] = d.tool_title_len
    else:
        X["text"] = d.combined_corpus.map(strip_titles).replace("", "(no description)")
        X["recruitment"] = d.recruitment_method.fillna("not recorded")
    if variant != "D":
        for c in MACRO:
            X[c] = d[c]
    for c in FLAGS:
        X[c] = d[c]
    return X


def fit_predict(Xtr, ytr, Xte, seed):
    cat_cols = [c for c in Xtr.columns if c == "recruitment"]
    num_cols = [c for c in Xtr.columns if c not in ("text", "recruitment")]
    cb = CatBoostClassifier(iterations=600, learning_rate=0.06, depth=6, auto_class_weights="Balanced", random_seed=seed, verbose=0,
                            class_names=CLASSES, thread_count=-1)
    cb.fit(Pool(Xtr, ytr, text_features=["text"], cat_features=cat_cols))
    p_cb = cb.predict_proba(Pool(Xte, text_features=["text"], cat_features=cat_cols))
    # XGBoost on the structured (non-text) features, one-hot for the recruitment category
    enc = lambda X: pd.get_dummies(X[num_cols + cat_cols], columns=cat_cols, dtype=float)
    Ztr, Zte = enc(Xtr), enc(Xte).reindex(columns=enc(Xtr).columns, fill_value=0.0)
    yi = ytr.map({c: i for i, c in enumerate(CLASSES)})
    xg = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9, colsample_bytree=0.9, random_state=seed,
                       objective="multi:softprob", num_class=len(CLASSES), eval_metric="mlogloss", n_jobs=-1)
    xg.fit(Ztr, yi, sample_weight=compute_sample_weight("balanced", yi))
    p_xg = xg.predict_proba(Zte)
    return W_CAT * p_cb + (1 - W_CAT) * p_xg, cb, xg, Zte


def metrics(y, p):
    pred = np.array(CLASSES)[p.argmax(1)]
    rep = classification_report(y, pred, labels=CLASSES, output_dict=True, zero_division=0)
    return {"accuracy": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, labels=CLASSES, average="macro", zero_division=0),
            "weighted_f1": f1_score(y, pred, labels=CLASSES, average="weighted", zero_division=0), "balanced_acc": balanced_accuracy_score(y, pred),
            "per_class_f1": {c: rep[c]["f1-score"] for c in CLASSES if c in rep}, "support": {c: int((y == c).sum()) for c in CLASSES}}


def baselines(d, te):
    y = d.loc[te, "final_target_group"]
    maj = y.value_counts().idxmax()
    out = {"majority": metrics(y, np.array([[1.0 if c == maj else 0.0 for c in CLASSES]] * len(y)))}
    rule = np.where(d.loc[te, "method_types_1"].isna(), "Other/Direct", "Deliberative/Dialogic")
    out["blank_field_rule"] = metrics(y, np.array([[1.0 if c == r else 0.0 for c in CLASSES] for r in rule]))
    return out


def group_importance(d, tr, te, variant, seed=0, reps=3):
    """Drop in accuracy and macro-F1 when each feature group is shuffled in the test cases (model fixed)."""
    X = frame(d, variant)
    y = d.final_target_group
    p, cb, xg, _ = fit_predict(X.loc[tr], y[tr], X.loc[te], seed)
    base = metrics(y[te], p)
    groups = {"Text (case description)": ["text"], "National indicators": [c for c in MACRO if c in X], "Context flags": FLAGS,
              "Recruitment": [c for c in X.columns if c.startswith("recruitment")], "Tool title length": [c for c in X.columns if c == "tool_title_len"]}
    rng = np.random.default_rng(seed)
    out = {}
    for g, cols in groups.items():
        if not cols:
            continue
        da, df = [], []
        for _ in range(reps):
            Xs = X.loc[te].copy()
            idx = rng.permutation(len(Xs))
            Xs[cols] = Xs[cols].values[idx]
            cat_cols = [c for c in Xs.columns if c == "recruitment"]
            num_cols = [c for c in Xs.columns if c not in ("text", "recruitment")]
            p_cb = cb.predict_proba(Pool(Xs, text_features=["text"], cat_features=cat_cols))
            enc = pd.get_dummies(Xs[num_cols + cat_cols], columns=cat_cols, dtype=float).reindex(columns=xg.get_booster().feature_names, fill_value=0.0)
            m = metrics(y[te], W_CAT * p_cb + (1 - W_CAT) * xg.predict_proba(enc))
            da.append(base["accuracy"] - m["accuracy"])
            df.append(base["macro_f1"] - m["macro_f1"])
        out[g] = {"accuracy_drop": float(np.mean(da)), "macro_f1_drop": float(np.mean(df))}
    tot = sum(max(0, v["accuracy_drop"]) for v in out.values()) or 1
    for v in out.values():
        v["share"] = max(0, v["accuracy_drop"]) / tot
    return out


def macro_contribs(d, tr, te, variant, seed=0):
    """Mean |contribution| of each national indicator by class in the XGBoost component (the paper's Table 6 view)."""
    X = frame(d, variant)
    y = d.final_target_group
    _, _, xg, Zte = fit_predict(X.loc[tr], y[tr], X.loc[te], seed)
    import xgboost as xgb
    contrib = xg.get_booster().predict(xgb.DMatrix(Zte), pred_contribs=True)      # rows x classes x (features + bias)
    cols = list(Zte.columns)
    return {c: {m: float(np.abs(contrib[:, k, cols.index(m)]).mean()) for m in MACRO if m in cols} for k, c in enumerate(CLASSES)}


def run():
    d = pd.read_csv(C.CAPSTONE, low_memory=False)
    d = d[d.final_target_group.isin(CLASSES)]
    rec = d.method_types_1.notna()
    sets = {"A": d.index, "B": d.index, "C": d.index[rec], "D": d.index[rec]}
    res = {}
    for v, idx in sets.items():
        sub = d.loc[idx]
        tr, te = sub.index[sub.split == "train"], sub.index[sub.split == "test"]
        X, y = frame(d, v), d.final_target_group
        runs = [metrics(y[te], fit_predict(X.loc[tr], y[tr], X.loc[te], s)[0]) for s in SEEDS]
        agg = {k: [float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs]))] for k in ("accuracy", "macro_f1", "weighted_f1", "balanced_acc")}
        agg["per_class_f1"] = {c: float(np.mean([r["per_class_f1"].get(c, 0) for r in runs])) for c in CLASSES}
        agg["support"] = runs[0]["support"]
        agg["n_train"], agg["n_test"] = int(len(tr)), int(len(te))
        agg["baselines"] = baselines(d, te)
        if v in ("A", "C"):
            agg["importance"] = group_importance(d, tr, te, v)
            agg["macro_contribs"] = macro_contribs(d, tr, te, v)
        res[v] = agg
        print(v, f"acc {agg['accuracy'][0]:.4f}±{agg['accuracy'][1]:.4f}  macroF1 {agg['macro_f1'][0]:.3f}  wF1 {agg['weighted_f1'][0]:.3f}  "
                 f"| majority {agg['baselines']['majority']['accuracy']:.3f}  blank-rule {agg['baselines']['blank_field_rule']['accuracy']:.3f}", flush=True)
    (C.OUT / "paper_rerun.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    run()
