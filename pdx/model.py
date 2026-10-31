"""Model of documented impact and the design-effect estimates behind the advisor.

The model predicts whether a case's entry documents evidence of impact from its design, its context and how well it
is documented. A design effect is the average change in predicted probability when one design choice is switched on
versus off for every case (other choices, context and documentation held as they are), with a bootstrap interval from
refitting the model on resampled cases. These are associations in a crowd-sourced archive, not causal estimates."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from . import config as C
from . import features as FT

PARAMS = dict(max_iter=250, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0)
LR_C = 0.1          # set by tune(): cross-validated on the training cases


def fit_gb(X, y, seed=C.SEED):
    """Benchmark: gradient-boosted trees (more accurate, but erratic when one design choice is changed)."""
    return HistGradientBoostingClassifier(random_state=seed, **PARAMS).fit(X, y)


def fit(X, y, seed=C.SEED):
    """The advisor's model: L2-regularized logistic regression on imputed, standardized features. Smooth and additive,
    so a single change to a design moves the score by a stable, explainable amount."""
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=LR_C, max_iter=5000)).fit(X, y)


def tune(X, y):
    from sklearn.model_selection import cross_val_score
    global LR_C
    scores = {c: cross_val_score(make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=c, max_iter=5000)),
                                 X, y, cv=5, scoring="roc_auc").mean() for c in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)}
    LR_C = max(scores, key=scores.get)
    return LR_C, scores


def evaluate(X, y, Xte, yte):
    """Held-out performance on the capstone's test split: the advisor's model, the tree benchmark, documentation alone."""
    c, cv = tune(X, y)
    lr = fit(X, y)
    gb = fit_gb(X, y)
    doc = [c_ for c_, k, *_ in FT.columns() if k == "doc"]
    lr_doc = fit(X[doc], y)
    p = lr.predict_proba(Xte)[:, 1]
    return {"auc": float(roc_auc_score(yte, p)), "ap": float(average_precision_score(yte, p)), "base_rate": float(yte.mean()),
            "auc_trees": float(roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])), "C": c,
            "auc_doc_only": float(roc_auc_score(yte, lr_doc.predict_proba(Xte[doc])[:, 1])), "n_train": int(len(y)), "n_test": int(len(yte))}, lr


def levers():
    """Design choices the advisor can switch: (label, field, columns set on, columns cleared)."""
    out = []
    for name, kind, f, lab in FT.columns():
        if kind != "design" or f == "log_participants":
            continue
        if "=" in name and f in FT.SINGLE:
            sib = [c for c, _, g, _ in FT.columns() if g == f and c != name]
            out.append((name, f, lab, [name], sib))
        else:
            out.append((name, f, lab, [name], []))
    return out


def effect(model, X, lev):
    name, f, lab, on, off = lev
    X1, X0 = X.copy(), X.copy()
    X1[on] = 1.0
    X1[off] = 0.0
    X0[on] = 0.0
    return float(model.predict_proba(X1)[:, 1].mean() - model.predict_proba(X0)[:, 1].mean())


def effects(X, y, B=200, seed=C.SEED):
    """Average marginal effect of each lever, with a bootstrap interval (refit on resampled cases)."""
    L = levers()
    gb = fit(X, y)
    point = {l[0]: effect(gb, X, l) for l in L}
    rng = np.random.default_rng(seed)
    boots = {l[0]: [] for l in L}
    for b in range(B):
        idx = rng.integers(0, len(X), len(X))
        m = fit(X.iloc[idx], y.iloc[idx], seed + b + 1)
        for l in L:
            boots[l[0]].append(effect(m, X, l))
    prev = {l[0]: float(X[l[0]].mean()) for l in L}
    rows = []
    for name, f, lab, on, off in L:
        lo, hi = np.percentile(boots[name], [5, 95])
        rows.append({"feature": name, "field": f, "label": lab, "effect": point[name], "lo": float(lo), "hi": float(hi),
                     "share": prev[name], "n": int(X[name].sum())})
    return pd.DataFrame(rows), gb


def export(lr, feats):
    """The fitted model as plain numbers: missing -> median, standardize, weighted sum, logistic."""
    imp, sc, m = lr.named_steps["simpleimputer"], lr.named_steps["standardscaler"], lr.named_steps["logisticregression"]
    r = lambda a: [round(float(v), 6) for v in a]
    return {"features": feats, "med": r(imp.statistics_), "mu": r(sc.mean_), "sd": r(sc.scale_), "coef": r(m.coef_[0]), "b0": float(m.intercept_[0])}
