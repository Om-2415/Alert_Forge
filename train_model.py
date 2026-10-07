"""
Step 2 - train the Random Forest on BATCH 1 ONLY and freeze it.

Adapted from the original UNSW-NB15 "RF model training with Optuna" notebook
(same Random Forest + Optuna TPE approach and search space; credit to its
original author). Changes:

  * trains on batch 1 only; batch 2 is never seen (frozen-model experiment)
  * removes the target leak (`Label` stayed in the original 46 features) and
    the raw timestamps (`Stime`, `Ltime`)
  * Optuna validates on the LAST 20% OF BATCH 1 BY TIME, not a random split
  * Optuna objective is macro-F1 (default) instead of weighted-F1, so rare
    classes (Analysis, Backdoors, Worms...) are not ignored
  * no StandardScaler (it has no effect on a Random Forest)
  * saves a SHA-256 of the model file so "frozen" can be demonstrated

Usage:
    python train_model.py                      # Optuna, 20 trials
    python train_model.py --trials 30
    python train_model.py --no-optuna          # use the original notebook's best params
    python train_model.py --normal-cap 300000  # dev: cap Normal rows for speed
"""
import argparse, hashlib, json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
import config as C
from common import build_features, fit_encoders, load_df

# Best parameters found by the original notebook (30 Optuna trials).
NOTEBOOK_BEST = dict(n_estimators=150, max_depth=20, min_samples_split=20,
                     min_samples_leaf=10, max_features="log2", max_samples=0.8,
                     bootstrap=True, class_weight="balanced_subsample")


def make_rf(params, n_jobs=-1):
    return RandomForestClassifier(**params, random_state=C.SEED, n_jobs=n_jobs)


def run_optuna(Xtr, ytr, Xva, yva, n_trials, average):
    import optuna
    from optuna.samplers import TPESampler
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = {   # same search space as the original notebook
            "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
            "max_depth": trial.suggest_int("max_depth", 5, 20, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 20, 100, step=20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 10, 50, step=10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            "max_samples": trial.suggest_float("max_samples", 0.5, 0.8, step=0.1),
            "bootstrap": True,
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"]),
        }
        pred = make_rf(params).fit(Xtr, ytr).predict(Xva)
        score = f1_score(yva, pred, average=average, zero_division=0)
        print(f"  trial {trial.number:>2}: {average}-F1 = {score:.4f}", flush=True)
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=C.SEED))
    study.optimize(objective, n_trials=n_trials)
    best = dict(study.best_params, bootstrap=True)
    return best, float(study.best_value)


def main(a):
    flows = load_df(C.ART_DIR / "flows")
    b1 = flows[flows["batch"] == 1].copy()
    if a.normal_cap:
        normal = b1[b1["attack_cat"] == "Normal"]
        if len(normal) > a.normal_cap:
            drop = normal.sample(len(normal) - a.normal_cap, random_state=C.SEED).index
            b1 = b1.drop(index=drop)
    b1 = b1.sort_values("Stime").reset_index(drop=True)

    classes = sorted(b1["attack_cat"].astype(str).unique())
    y = b1["attack_cat"].astype(str).map({c: i for i, c in enumerate(classes)}).to_numpy()
    enc = fit_encoders(b1)
    X = build_features(b1, enc)
    print(f"batch-1 rows: {len(X):,} | features: {X.shape[1]} | classes: {classes}")
    print("features:", enc["feature_names"])

    # time-ordered validation inside batch 1 (never random)
    n_val = int(0.2 * len(X))
    Xtr, ytr, Xva, yva = X.iloc[:-n_val], y[:-n_val], X.iloc[-n_val:], y[-n_val:]

    if a.no_optuna:
        params, best_val = dict(NOTEBOOK_BEST), None
        print("using the original notebook's best parameters (no Optuna)")
    else:
        try:
            params, best_val = run_optuna(Xtr, ytr, Xva, yva, a.trials, a.objective)
        except ImportError:
            print("optuna not installed -> falling back to the notebook's best parameters")
            params, best_val = dict(NOTEBOOK_BEST), None
    print("final params:", params)

    # validation report with the chosen params (time-ordered hold-out inside batch 1)
    t = time.time()
    pv = make_rf(params).fit(Xtr, ytr).predict(Xva)
    val = {"macro_f1": float(f1_score(yva, pv, average="macro", zero_division=0)),
           "weighted_f1": float(f1_score(yva, pv, average="weighted", zero_division=0)),
           "per_class_recall": {c: float(r) for c, r in zip(
               classes, recall_score(yva, pv, labels=range(len(classes)), average=None, zero_division=0))}}

    # FINAL model: refit on ALL of batch 1, then freeze
    model = make_rf(params).fit(X, y)
    out = C.ART_DIR / "rf_batch1_frozen.joblib"
    joblib.dump({"model": model, "classes": classes, "encoders": enc, "params": params,
                 "batch_cutoff_utc": C.BATCH_CUTOFF_UTC, "n_train_rows": int(len(X))}, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    report = {"params": params, "optuna_best_value": best_val, "objective": a.objective,
              "validation_last20pct_of_batch1": val, "model_sha256": sha,
              "train_seconds": round(time.time() - t)}
    (C.ART_DIR / "training_report.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved {out}\nSHA-256 {sha}\nvalidation macro-F1 {val['macro_f1']:.4f} "
          f"| weighted-F1 {val['weighted_f1']:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--objective", choices=["macro", "weighted"], default="macro")
    ap.add_argument("--no-optuna", action="store_true")
    ap.add_argument("--normal-cap", type=int, default=None,
                    help="dev only: cap the number of Normal rows used for training")
    main(ap.parse_args())
