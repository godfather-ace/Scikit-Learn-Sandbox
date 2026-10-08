"""
scikit-learn Lab — local server
-----------------------------------
Runs the 4-step ML workflow (Load & split -> Fit -> Predict -> Score & tune)
on the server with scikit-learn and streams each step back to the browser.

Run:  uvicorn app:app --reload --port 8000
Open: http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn import datasets as skds
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE = Path(__file__).parent
app = FastAPI(title="scikit-learn Lab")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------
def make_churn(n: int = 1200, seed: int = 7) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """A realistic-feeling synthetic telecom churn dataset."""
    rng = np.random.default_rng(seed)
    contract = rng.choice([0, 1, 2], size=n, p=[0.55, 0.25, 0.20])  # 0=monthly,1=1yr,2=2yr
    has_support = rng.binomial(1, 0.35, n)
    monthly = np.round(rng.normal(68, 18, n).clip(20, 120), 2)
    payment_auto = rng.binomial(1, 0.45, n)
    tenure = np.round(rng.gamma(2.0, 14, n).clip(1, 72)).astype(int)
    total = np.round(monthly * tenure * rng.uniform(0.9, 1.05, n), 2)
    senior = rng.binomial(1, 0.16, n)

    logit = (
        2.4
        - 1.4 * contract
        - 0.8 * has_support
        + 0.025 * (monthly - 68)
        - 0.6 * payment_auto
        - 0.07 * tenure
        + 0.5 * senior
    )
    p = 1 / (1 + np.exp(-6 * logit))  # sharpened: ~34% churners, ~92% accuracy
    y = (rng.uniform(size=n) < p).astype(int)

    X = pd.DataFrame(
        {
            "contract": contract.astype(float),
            "has_support": has_support.astype(float),
            "monthly_charges": monthly,
            "payment_auto": payment_auto.astype(float),
            "tenure_months": tenure.astype(float),
            "total_charges": total,
            "senior_citizen": senior.astype(float),
        }
    )
    return X, pd.Series(y, name="churned"), ["stayed", "churned"]


def _from_sklearn(loader) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    d = loader()
    X = pd.DataFrame(d.data, columns=[c.replace(" ", "_") for c in d.feature_names])
    return X, pd.Series(d.target, name="target"), [str(t) for t in d.target_names]


DATASETS: dict[str, dict[str, Any]] = {
    "churn": {
        "label": "Customer Churn (synthetic)",
        "blurb": "Telecom customers: will they leave? Generated in code for teaching.",
        "load": make_churn,
    },
    "iris": {
        "label": "Iris (flowers)",
        "blurb": "Classic 3-class problem: predict iris species from petal and sepal size.",
        "load": lambda: _from_sklearn(skds.load_iris),
    },
    "wine": {
        "label": "Wine cultivars",
        "blurb": "13 chemical measurements → which of 3 cultivars produced the wine.",
        "load": lambda: _from_sklearn(skds.load_wine),
    },
    "breast_cancer": {
        "label": "Breast Cancer (diagnostic)",
        "blurb": "30 cell-nucleus measurements → malignant or benign.",
        "load": lambda: _from_sklearn(skds.load_breast_cancer),
    },
    "digits": {
        "label": "Handwritten Digits (8×8)",
        "blurb": "64 pixel intensities → which digit 0–9. A 10-class challenge.",
        "load": lambda: _from_sklearn(skds.load_digits),
    },
}

_cache: dict[str, tuple[pd.DataFrame, pd.Series, list[str]]] = {}


def load_dataset(key: str):
    if key not in DATASETS:
        raise HTTPException(404, f"Unknown dataset '{key}'")
    if key not in _cache:
        _cache[key] = DATASETS[key]["load"]()
    return _cache[key]


# --------------------------------------------------------------------------
# Models (with tunable hyperparameters exposed to the UI)
# --------------------------------------------------------------------------
MODELS: dict[str, dict[str, Any]] = {
    "RandomForestClassifier": {
        "label": "RandomForestClassifier",
        "blurb": "Many decision trees voting together. A reliable first choice.",
        "import": "from sklearn.ensemble import RandomForestClassifier",
        "cls": RandomForestClassifier,
        "scale": False,
        "params": {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "step": 10},
            "max_depth": {"type": "int", "default": 0, "min": 0, "max": 30, "step": 1, "help": "0 = no limit"},
            "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 20, "step": 1},
        },
        "sweep": {"param": "max_depth", "values": [1, 2, 3, 4, 6, 8, 10, 15, 20]},
    },
    "LogisticRegression": {
        "label": "LogisticRegression",
        "blurb": "A linear model that outputs probabilities. Fast, interpretable baseline.",
        "import": "from sklearn.linear_model import LogisticRegression",
        "cls": LogisticRegression,
        "scale": True,
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 100, "step": 0.1, "help": "Inverse regularisation strength"},
            "max_iter": {"type": "int", "default": 1000, "min": 100, "max": 5000, "step": 100},
        },
        "sweep": {"param": "C", "values": [0.001, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100]},
    },
    "DecisionTreeClassifier": {
        "label": "DecisionTreeClassifier",
        "blurb": "A single tree of yes/no questions. Easy to explain, easy to overfit.",
        "import": "from sklearn.tree import DecisionTreeClassifier",
        "cls": DecisionTreeClassifier,
        "scale": False,
        "params": {
            "max_depth": {"type": "int", "default": 0, "min": 0, "max": 30, "step": 1, "help": "0 = no limit"},
            "min_samples_leaf": {"type": "int", "default": 1, "min": 1, "max": 30, "step": 1},
            "criterion": {"type": "choice", "default": "gini", "options": ["gini", "entropy", "log_loss"]},
        },
        "sweep": {"param": "max_depth", "values": [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]},
    },
    "KNeighborsClassifier": {
        "label": "KNeighborsClassifier",
        "blurb": "Predicts by majority vote of the k most similar training rows.",
        "import": "from sklearn.neighbors import KNeighborsClassifier",
        "cls": KNeighborsClassifier,
        "scale": True,
        "params": {
            "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 50, "step": 1},
            "weights": {"type": "choice", "default": "uniform", "options": ["uniform", "distance"]},
        },
        "sweep": {"param": "n_neighbors", "values": [1, 3, 5, 7, 9, 15, 21, 31, 45]},
    },
    "SVC": {
        "label": "SVC (Support Vector Machine)",
        "blurb": "Finds the widest boundary between classes; kernels bend it.",
        "import": "from sklearn.svm import SVC",
        "cls": SVC,
        "scale": True,
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 100, "step": 0.1},
            "kernel": {"type": "choice", "default": "rbf", "options": ["rbf", "linear", "poly"]},
        },
        "sweep": {"param": "C", "values": [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]},
    },
    "GradientBoostingClassifier": {
        "label": "GradientBoostingClassifier",
        "blurb": "Trees built one after another, each fixing the last one's mistakes.",
        "import": "from sklearn.ensemble import GradientBoostingClassifier",
        "cls": GradientBoostingClassifier,
        "scale": False,
        "params": {
            "n_estimators": {"type": "int", "default": 100, "min": 10, "max": 500, "step": 10},
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01},
            "max_depth": {"type": "int", "default": 3, "min": 1, "max": 10, "step": 1},
        },
        "sweep": {"param": "n_estimators", "values": [10, 25, 50, 100, 150, 200, 300]},
    },
    "GaussianNB": {
        "label": "GaussianNB (Naive Bayes)",
        "blurb": "Probability-based and extremely fast. Assumes features are independent.",
        "import": "from sklearn.naive_bayes import GaussianNB",
        "cls": GaussianNB,
        "scale": False,
        "params": {
            "var_smoothing": {"type": "float", "default": 1e-9, "min": 1e-12, "max": 1, "step": 1e-9},
        },
        "sweep": {"param": "var_smoothing", "values": [1e-12, 1e-9, 1e-6, 1e-4, 1e-2, 1e-1, 1]},
    },
}


def clean_params(model_key: str, raw: dict[str, Any] | None) -> dict[str, Any]:
    spec = MODELS[model_key]["params"]
    out: dict[str, Any] = {}
    for name, p in spec.items():
        v = (raw or {}).get(name, p["default"])
        if p["type"] == "int":
            v = int(v)
            if name == "max_depth" and v == 0:
                v = None
        elif p["type"] == "float":
            v = float(v)
        elif p["type"] == "choice" and v not in p["options"]:
            v = p["default"]
        out[name] = v
    return out


def build_model(model_key: str, params: dict[str, Any], seed: int, scale: bool | None):
    if model_key not in MODELS:
        raise HTTPException(404, f"Unknown model '{model_key}'")
    m = MODELS[model_key]
    kwargs = dict(params)
    if "random_state" in m["cls"]().get_params():
        kwargs["random_state"] = seed
    if model_key == "SVC":
        kwargs["probability"] = True
    est = m["cls"](**kwargs)
    use_scale = m["scale"] if scale is None else scale
    return (make_pipeline(StandardScaler(), est) if use_scale else est), use_scale


def final_estimator(model):
    return model.steps[-1][1] if hasattr(model, "steps") else model


# --------------------------------------------------------------------------
# Request schemas
# --------------------------------------------------------------------------
class RunRequest(BaseModel):
    dataset: str = "churn"
    model: str = "RandomForestClassifier"
    test_size: float = Field(0.25, ge=0.05, le=0.5)
    random_state: int = 42
    params: dict[str, Any] | None = None
    scale: bool | None = None  # None = model's sensible default
    cv_folds: int = Field(5, ge=0, le=10)


class SweepRequest(RunRequest):
    param: str | None = None
    values: list[float] | None = None


class PredictRequest(RunRequest):
    row: dict[str, float]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def fmt_row(row: pd.Series, n: int = 4) -> str:
    parts = []
    for k, v in list(row.items())[:n]:
        parts.append(f"{k.split('_')[0]}={v:.2f}")
    return " ".join(parts) + (" …" if len(row) > n else "")


def code_snippet(req: RunRequest, params: dict[str, Any], use_scale: bool) -> str:
    m = MODELS[req.model]
    ds = req.dataset
    p_str = ", ".join(f"{k}={v!r}" for k, v in params.items() if v != m["params"][k]["default"] and not (k == "max_depth" and v is None))
    lines = ["import pandas as pd", "from sklearn.model_selection import train_test_split", m["import"]]
    if use_scale:
        lines += ["from sklearn.pipeline import make_pipeline", "from sklearn.preprocessing import StandardScaler"]
    lines.append("")
    if ds == "churn":
        lines.append("# X, y = your churn DataFrame and target column")
    else:
        loader = {"iris": "load_iris", "wine": "load_wine", "breast_cancer": "load_breast_cancer", "digits": "load_digits"}[ds]
        lines += [f"from sklearn.datasets import {loader}", f"X, y = {loader}(return_X_y=True, as_frame=True)"]
    lines += [
        "",
        "# 1. Load & split",
        f"X_train, X_test, y_train, y_test = train_test_split(",
        f"    X, y, test_size={req.test_size}, random_state={req.random_state}, stratify=y)",
        "",
        "# 2. Fit",
    ]
    ctor = f"{m['cls'].__name__}({p_str})"
    lines.append(f"model = make_pipeline(StandardScaler(), {ctor})" if use_scale else f"model = {ctor}")
    lines += [
        "model.fit(X_train, y_train)",
        "",
        "# 3. Predict",
        "preds = model.predict(X_test)",
        "",
        "# 4. Score & tune",
        "print(model.score(X_test, y_test))",
    ]
    return "\n".join(lines)


def ev(kind: str, **data) -> str:
    return json.dumps({"event": kind, **data}, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n"


def _thin(a, b, k: int = 120):
    """Downsample paired curve arrays to at most k points for the browser."""
    if len(a) <= k:
        return [round(float(v), 4) for v in a], [round(float(v), 4) for v in b]
    idx = np.unique(np.linspace(0, len(a) - 1, k).astype(int))
    return [round(float(a[i]), 4) for i in idx], [round(float(b[i]), 4) for i in idx]


def build_visuals(model, X_tr, X_te, y_te, preds, proba, names):
    out: dict[str, Any] = {}
    # per-class precision / recall / F1
    p, r, f, sup = precision_recall_fscore_support(y_te, preds, labels=range(len(names)), zero_division=0)
    out["per_class"] = [
        {"label": names[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(sup[i])}
        for i in range(len(names))
    ]
    # ROC curves (one-vs-rest for multi-class)
    if proba is not None:
        curves = []
        classes = [1] if len(names) == 2 else range(len(names))
        for c in classes:
            yt = (y_te.values == c).astype(int)
            if yt.min() == yt.max():
                continue
            fpr, tpr, _ = roc_curve(yt, proba[:, c])
            fx, ty = _thin(fpr, tpr)
            curves.append({"label": names[c], "fpr": fx, "tpr": ty, "auc": float(roc_auc_score(yt, proba[:, c]))})
        out["roc"] = curves
        out["confidence"] = [round(float(v), 4) for v in proba.max(axis=1)]
        out["correct"] = (preds == y_te.values).astype(int).tolist()
    # 2-D PCA map of the test set (fit on scaled training data)
    sc = StandardScaler().fit(X_tr)
    pca = PCA(n_components=2, random_state=0).fit(sc.transform(X_tr))
    pts = pca.transform(sc.transform(X_te))
    n = min(600, len(X_te))
    out["pca"] = {
        "x": [round(float(v), 3) for v in pts[:n, 0]],
        "y": [round(float(v), 3) for v in pts[:n, 1]],
        "actual": [names[int(v)] for v in y_te.values[:n]],
        "predicted": [names[int(v)] for v in preds[:n]],
        "explained": [round(float(v), 4) for v in pca.explained_variance_ratio_],
    }
    return out


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/")
def index():
    return FileResponse(BASE / "static" / "index.html")


@app.get("/vendor/plotly.min.js")
def plotly_js():
    """Serve Plotly from the installed Python package so charts work offline."""
    try:
        import plotly

        return FileResponse(Path(plotly.__file__).parent / "package_data" / "plotly.min.js", media_type="application/javascript")
    except Exception:
        raise HTTPException(404, "plotly not installed — run: pip install plotly")


@app.get("/api/columns/{dataset}")
def columns(dataset: str, x: str, y: str, hist: str | None = None):
    """Raw values for the Data Explorer scatter / histogram."""
    X, yy, names = load_dataset(dataset)
    for c in [x, y] + ([hist] if hist else []):
        if c not in X.columns:
            raise HTTPException(400, f"Unknown column {c}")
    out = {"x": X[x].round(4).tolist(), "y": X[y].round(4).tolist(), "label": [names[i] for i in yy], "classes": names}
    if hist:
        out["hist"] = X[hist].round(4).tolist()
    corr = X.corr().round(3)
    if X.shape[1] <= 30:
        out["corr"] = {"cols": list(corr.columns), "z": corr.fillna(0).values.tolist()}
    return out


@app.get("/api/meta")
def meta():
    ds = {}
    for k, d in DATASETS.items():
        X, y, names = load_dataset(k)
        ds[k] = {
            "label": d["label"],
            "blurb": d["blurb"],
            "rows": len(X),
            "features": list(X.columns),
            "classes": names,
            "feature_ranges": {c: [float(X[c].min()), float(X[c].median()), float(X[c].max())] for c in X.columns},
        }
    models = {
        k: {"label": m["label"], "blurb": m["blurb"], "params": m["params"], "scale": m["scale"], "sweep": m["sweep"]}
        for k, m in MODELS.items()
    }
    return {"datasets": ds, "models": models}


@app.get("/api/preview/{dataset}")
def preview(dataset: str, n: int = 12):
    X, y, names = load_dataset(dataset)
    df = X.copy()
    df["→ target"] = [names[i] for i in y]
    counts = pd.Series([names[i] for i in y]).value_counts().to_dict()
    desc = X.describe().T[["mean", "std", "min", "max"]].round(3)
    return {
        "columns": list(df.columns),
        "rows": df.head(n).round(3).values.tolist(),
        "shape": [len(X), X.shape[1]],
        "class_counts": counts,
        "missing": int(X.isna().sum().sum()),
        "summary": [{"feature": f, **desc.loc[f].to_dict()} for f in desc.index],
    }


@app.post("/api/run")
def run(req: RunRequest):
    """Streams NDJSON: one event per pipeline step, then a final 'done'."""

    def gen():
        X, y, names = load_dataset(req.dataset)
        params = clean_params(req.model, req.params)
        model, use_scale = build_model(req.model, params, req.random_state, req.scale)

        # 1. Load & split
        t0 = time.perf_counter()
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=req.test_size, random_state=req.random_state, stratify=y
        )
        t_split = (time.perf_counter() - t0) * 1000
        yield ev("step", step=1, ms=round(t_split, 1), info=f"{len(X_tr)} train / {len(X_te)} test")

        # 2. Fit
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        t_fit = (time.perf_counter() - t0) * 1000
        yield ev("step", step=2, ms=round(t_fit, 1), info=type(final_estimator(model)).__name__)

        # 3. Predict
        t0 = time.perf_counter()
        preds = model.predict(X_te)
        t_pred = (time.perf_counter() - t0) * 1000
        yield ev("step", step=3, ms=round(t_pred, 1), info=f"{len(preds)} predictions")

        # 4. Score & tune
        t0 = time.perf_counter()
        binary = len(names) == 2
        avg = "binary" if binary else "macro"
        test_acc = accuracy_score(y_te, preds)
        train_acc = accuracy_score(y_tr, model.predict(X_tr))
        metrics = {
            "precision": precision_score(y_te, preds, average=avg, zero_division=0),
            "recall": recall_score(y_te, preds, average=avg, zero_division=0),
            "f1": f1_score(y_te, preds, average=avg, zero_division=0),
        }
        try:
            proba = model.predict_proba(X_te)
            metrics["roc_auc"] = (
                roc_auc_score(y_te, proba[:, 1]) if binary else roc_auc_score(y_te, proba, multi_class="ovr")
            )
        except Exception:
            proba = None

        cv, t_cv = None, 0.0
        if req.cv_folds >= 2:
            t_cv0 = time.perf_counter()
            cv_model, _ = build_model(req.model, params, req.random_state, req.scale)
            scores = cross_val_score(cv_model, X_tr, y_tr, cv=req.cv_folds)
            cv = {"folds": [round(float(s), 4) for s in scores], "mean": float(scores.mean()), "std": float(scores.std())}
            t_cv = (time.perf_counter() - t_cv0) * 1000

        est = final_estimator(model)
        if hasattr(est, "feature_importances_"):
            imp, imp_kind = est.feature_importances_, "built-in (impurity)"
        elif hasattr(est, "coef_"):
            imp, imp_kind = np.abs(est.coef_).mean(axis=0), "|coefficients| (scaled)" if use_scale else "|coefficients|"
        else:
            pi = permutation_importance(model, X_te, y_te, n_repeats=5, random_state=req.random_state)
            imp, imp_kind = np.clip(pi.importances_mean, 0, None), "permutation"
        imp = np.asarray(imp, dtype=float)
        imp = imp / imp.sum() if imp.sum() > 0 else imp
        order = np.argsort(imp)[::-1][:10]
        top = [{"feature": X.columns[i], "importance": float(imp[i])} for i in order]

        samples = []
        for i in range(min(10, len(X_te))):
            row = X_te.iloc[i]
            s = {
                "features": fmt_row(row),
                "actual": names[int(y_te.iloc[i])],
                "predicted": names[int(preds[i])],
            }
            if proba is not None:
                s["confidence"] = float(proba[i].max())
            samples.append(s)

        t_score = (time.perf_counter() - t0) * 1000 - t_cv
        yield ev("step", step=4, ms=round(t_score, 1), info=f"{test_acc:.1%} accuracy")

        # ---- extra visuals (not counted in step timing) ----
        visuals = build_visuals(model, X_tr, X_te, y_te, preds, proba, names)

        gap = train_acc - test_acc
        yield ev(
            "done",
            accuracy=test_acc,
            train_accuracy=train_acc,
            fit_warning=(
                "big gap — the model is overfitting" if gap > 0.10
                else "gap suggests mild overfitting" if gap > 0.04
                else "train and test agree — generalises well" if gap > -0.02
                else "test beats train — likely a lucky split"
            ),
            test_rows=len(X_te),
            correct=int((preds == y_te.values).sum()),
            train_rows=len(X_tr),
            n_features=X.shape[1],
            metrics=metrics,
            average=avg,
            cv=cv,
            confusion={"labels": names, "matrix": confusion_matrix(y_te, preds).tolist()},
            importance={"kind": imp_kind, "top": top},
            samples=samples,
            timing={"split": t_split, "fit": t_fit, "predict": t_pred, "score": t_score, "cv": t_cv},
            scaled=use_scale,
            params={k: v for k, v in params.items()},
            code=code_snippet(req, params, use_scale),
            visuals=visuals,
        )

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/sweep")
def sweep(req: SweepRequest):
    """Train once per value of one hyperparameter; stream train/test/CV accuracy."""
    m = MODELS.get(req.model)
    if not m:
        raise HTTPException(404, "Unknown model")
    param = req.param or m["sweep"]["param"]
    values = req.values or m["sweep"]["values"]
    if param not in m["params"]:
        raise HTTPException(400, f"{param} is not tunable for {req.model}")

    def gen():
        X, y, _ = load_dataset(req.dataset)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=req.test_size, random_state=req.random_state, stratify=y
        )
        best = None
        for v in values:
            params = clean_params(req.model, {**(req.params or {}), param: v})
            model, _ = build_model(req.model, params, req.random_state, req.scale)
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            ms = (time.perf_counter() - t0) * 1000
            tr, te = model.score(X_tr, y_tr), model.score(X_te, y_te)
            if best is None or te > best["test"]:
                best = {"value": v, "test": te}
            yield ev("point", param=param, value=v, train=tr, test=te, ms=round(ms, 1))
        yield ev("done", param=param, best=best)

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/compare")
def compare(req: RunRequest):
    """Run every model with default settings on the same split — a leaderboard."""

    def gen():
        X, y, _ = load_dataset(req.dataset)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=req.test_size, random_state=req.random_state, stratify=y
        )
        for key in MODELS:
            params = clean_params(key, None)
            model, scaled = build_model(key, params, req.random_state, None)
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            fit_ms = (time.perf_counter() - t0) * 1000
            preds = model.predict(X_te)
            yield ev(
                "row",
                model=key,
                label=MODELS[key]["label"],
                test=accuracy_score(y_te, preds),
                train=model.score(X_tr, y_tr),
                f1=f1_score(y_te, preds, average="macro"),
                fit_ms=round(fit_ms, 1),
                scaled=scaled,
            )
        yield ev("done")

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Train on the split, then score one hand-built row ('what-if' playground)."""
    X, y, names = load_dataset(req.dataset)
    params = clean_params(req.model, req.params)
    model, _ = build_model(req.model, params, req.random_state, req.scale)
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=req.test_size, random_state=req.random_state, stratify=y)
    model.fit(X_tr, y_tr)
    row = pd.DataFrame([{c: float(req.row.get(c, X[c].median())) for c in X.columns}])
    pred = int(model.predict(row)[0])
    try:
        proba = model.predict_proba(row)[0].tolist()
    except Exception:
        proba = None
    return {"predicted": names[pred], "classes": names, "proba": proba}


@app.post("/api/quickrun")
def quickrun(req: RunRequest):
    """Back-end for the 'Six lines' slide's ▶ Run it live button."""
    X, y, names = load_dataset(req.dataset)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=req.test_size, random_state=req.random_state, stratify=y)
    model = RandomForestClassifier(random_state=req.random_state).fit(X_tr, y_tr)
    preds = model.predict(X_te)
    score = model.score(X_te, y_te)
    return {
        "score": score,
        "test_rows": len(X_te),
        "first5": [{"pred": names[int(p)], "actual": names[int(a)]} for p, a in zip(preds[:5], y_te.iloc[:5])],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
