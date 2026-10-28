"""
Scenario machinery shared by blindspot_experiment.py and build_alert_stream.py,
so a scenario means exactly the same thing in both.

A scenario (config.BLIND_SPOT_SCENARIOS) is one of:
  catalog   retrain with a class relabelled Normal ("mask") or removed ("withhold")
  telemetry zero feature groups at inference
  evasion   blend attack-flow features toward real benign flows
Everything except "catalog" runs on the frozen model; "catalog" retrains it with
the SAME rows, hyper-parameters and seed as train_model.py.
"""
import joblib
import numpy as np
import pandas as pd
import config as C
from common import build_features, training_frame
from train_model import make_rf


def injection_label(sc):
    if sc is None:
        return "none"
    if sc["kind"] == "catalog":
        return ("label suppression (by construction)" if sc["mode"] == "mask"
                else "unseen class (empirical)")
    return {"telemetry": "feature loss (empirical)", "evasion": "mimicry (empirical)"}[sc["kind"]]


def apply_evasion(Xev, ev_cat, Xnormal, sc, seed):
    """Blend attack-flow features toward real benign flows with the same proto/service."""
    rng = np.random.default_rng(seed)
    cols = sorted({c for g in sc["groups"] for c in C.TELEMETRY_GROUPS[g]})
    is_attack = ev_cat != "Normal"
    sel = is_attack & (np.isin(ev_cat, sc["categories"]) if sc["categories"] else True)
    idx = np.where(sel)[0]
    X = Xev.copy()
    if len(idx) == 0:
        return X
    pool_key = Xnormal["proto"].astype(int) * 10_000 + Xnormal["service"].astype(int)
    pools = {k: np.flatnonzero((pool_key == k).to_numpy()) for k in pool_key.unique()}
    all_pool = np.arange(len(Xnormal))
    tgt_key = (Xev["proto"].astype(int) * 10_000 + Xev["service"].astype(int)).to_numpy()[idx]
    donors = np.empty(len(idx), dtype=np.int64)
    for k in np.unique(tgt_key):
        m = tgt_key == k
        donors[m] = rng.choice(pools.get(k, all_pool), size=int(m.sum()))
    s = float(sc["strength"])
    X.iloc[idx, [X.columns.get_loc(c) for c in cols]] = (
        (1 - s) * Xev.iloc[idx][cols].to_numpy() + s * Xnormal.iloc[donors][cols].to_numpy()
    ).astype("float32")
    return X


def predict_frame(model, X, classes):
    """Probabilities laid out over ALL classes (absent classes get 0)."""
    proba = model.predict_proba(X)
    mc = [c if isinstance(c, str) else classes[int(c)] for c in model.classes_]   # frozen model uses int codes
    out = pd.DataFrame(0.0, index=X.index, columns=[f"p_{c}" for c in classes], dtype="float32")
    for i, c in enumerate(mc):
        out[f"p_{c}"] = proba[:, i].astype("float32")
    out["pred_class"] = np.array(mc, dtype=object)[proba.argmax(1)]
    out["p_max"] = proba.max(1).astype("float32")
    return out


def train_arm(Xtr, ytr, params, scenario):
    y = ytr.copy()
    if scenario and scenario["kind"] == "catalog":
        if scenario["mode"] == "mask":
            y = np.where(np.isin(y, scenario["classes"]), "Normal", y)
        else:                                                    # withhold
            keep = ~np.isin(y, scenario["classes"])
            return make_rf(params).fit(Xtr[keep], y[keep])
    return make_rf(params).fit(Xtr, y)


def planted_categories(sc):
    """Stage categories the injection targets; None = every stage / not applicable."""
    if sc and sc["kind"] == "catalog":
        return set(sc["classes"])
    if sc and sc["kind"] == "evasion" and sc["categories"]:
        return set(sc["categories"])
    return None


def make_context(flows, normal_cap=None):
    """Frozen model + everything needed to apply any scenario to arbitrary flows."""
    bundle = joblib.load(C.ART_DIR / "rf_batch1_frozen.joblib")
    cap = normal_cap if normal_cap is not None else bundle.get("normal_cap")
    return {"bundle": bundle, "classes": bundle["classes"], "enc": bundle["encoders"],
            "params": dict(bundle["params"]), "normal_cap": cap, "flows": flows, "_train": None}


def training_arrays(ctx):
    """Batch-1 training features/labels (built lazily; same rows as train_model.py)."""
    if ctx["_train"] is None:
        b1 = training_frame(ctx["flows"], ctx["normal_cap"], C.SEED)
        ctx["_train"] = (build_features(b1, ctx["enc"]), b1["attack_cat"].astype(str).to_numpy())
    return ctx["_train"]


def scenario_predict(ctx, sc, rows):
    """Per-flow probabilities under scenario `sc` (None = frozen baseline).
    rows: flows with all feature columns and `attack_cat` (true label; used only to
    choose which attack flows an evasion scenario modifies)."""
    classes = ctx["classes"]
    X = build_features(rows, ctx["enc"])
    model = ctx["bundle"]["model"]
    if sc is not None and sc["kind"] == "catalog":
        Xtr, ytr = training_arrays(ctx)
        model = train_arm(Xtr, ytr, ctx["params"], sc)
    if sc is not None and sc["kind"] == "telemetry":
        X = X.copy()
        X[sorted({c for g in sc["groups"] for c in C.TELEMETRY_GROUPS[g]})] = 0.0
    if sc is not None and sc["kind"] == "evasion":
        Xtr, ytr = training_arrays(ctx)
        X = apply_evasion(X, rows["attack_cat"].astype(str).to_numpy(), Xtr[ytr == "Normal"], sc, C.SEED)
    return predict_frame(model, X, classes)
