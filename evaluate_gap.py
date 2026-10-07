"""
Step 5 - Attack-Chain Detection Gap analysis.

Compares the frozen model's per-flow predictions against the ground-truth
chains and reports, per stage and per chain:

  stage status   DETECTED / PARTIAL / MISLABELED / MISSED   (thresholds in config.py)
  strict_coverage       share of stages DETECTED (right stage named)
  visibility_coverage   share of stages the model alerted on AT ALL
  attribution gap       visibility - strict  (seen, but named as the wrong stage)
  detection_gap_pct     100 * (1 - strict_coverage)
  breakpoint            first stage that is not DETECTED
  visibility break      first stage that is MISSED (model effectively blind)

Batch 1 chains are IN the training period (optimistic baseline);
batch 2 chains are the honest future-data result.

Outputs (artifacts/): flow_metrics.json, stage_eval.csv, chain_eval.csv,
                      status_by_category.csv, gap_summary.json
Usage:
    python evaluate_gap.py
    python evaluate_gap.py --alert-rule pattack --alert-thr 0.5
"""
import argparse, json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
import config as C
from common import load_df

LETTER = {"DETECTED": "D", "PARTIAL": "P", "MISLABELED": "L", "MISSED": "M"}


def effective_pred(pred_df, classes, rule, thr):
    """Turn stored probabilities into the final alert label (Normal = no alert)."""
    if rule == "argmax":
        return pred_df["pred_class"].astype(str).to_numpy(dtype=object)
    P = pred_df[[f"p_{c}" for c in classes]].to_numpy()
    ni = classes.index("Normal")
    p_attack = 1.0 - P[:, ni]
    P = P.copy()
    P[:, ni] = -1.0                                   # best NON-Normal class
    best = np.array(classes, dtype=object)[P.argmax(1)]
    return np.where(p_attack >= thr, best, "Normal")


def flow_metrics(true, pred, classes):
    att, nrm = true != "Normal", true == "Normal"
    rec = recall_score(true, pred, labels=classes, average=None, zero_division=0)
    return {
        "n_flows": int(len(true)),
        "accuracy": float(accuracy_score(true, pred)),
        "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
        "macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
        "attack_vs_normal_recall": float((pred[att] != "Normal").mean()),
        "normal_false_positive_rate": float((pred[nrm] != "Normal").mean()) if nrm.any() else None,
        "per_class_recall": {c: float(r) for c, r in zip(classes, rec)},
    }


def stage_status(frac_correct, frac_alerted):
    return np.select(
        [frac_correct >= C.STAGE_DETECTED, frac_correct >= C.STAGE_PARTIAL, frac_alerted >= C.STAGE_ALERTED],
        ["DETECTED", "PARTIAL", "MISLABELED"], default="MISSED")


def main(rule, thr):
    flows = load_df(C.ART_DIR / "flows")[["flow_id", "attack_cat", "batch"]]
    flows["attack_cat"] = flows["attack_cat"].astype(str)
    pred = load_df(C.ART_DIR / "predictions")
    classes = sorted(c[2:] for c in pred.columns if c.startswith("p_") and c != "p_max")
    pred["eff"] = effective_pred(pred, classes, rule, thr)
    df = flows.merge(pred[["flow_id", "eff", "p_max"]], on="flow_id")

    # ---------------- flow-level metrics (all flows, per batch)
    fm = {f"batch{b}": flow_metrics(g["attack_cat"].to_numpy(), g["eff"].to_numpy(), classes)
          for b, g in df.groupby("batch")}
    (C.ART_DIR / "flow_metrics.json").write_text(json.dumps(fm, indent=2))

    # ---------------- stage-level
    truth = pd.read_csv(C.ART_DIR / "chain_truth.csv")
    cflows = pd.read_csv(C.ART_DIR / "chain_flows.csv").merge(df[["flow_id", "attack_cat", "eff", "p_max"]], on="flow_id")
    cflows["correct"] = cflows["eff"] == cflows["attack_cat"]
    cflows["alerted"] = cflows["eff"] != "Normal"
    key = ["chain_id", "stage_idx"]
    agg = cflows.groupby(key).agg(frac_correct=("correct", "mean"), frac_alerted=("alerted", "mean"),
                                  mean_p_max=("p_max", "mean")).reset_index()
    top = (cflows.groupby(key)["eff"].agg(lambda s: s.value_counts().index[0])
           .rename("top_pred").reset_index())
    stage = truth.merge(agg, on=key).merge(top, on=key)
    stage["status"] = stage_status(stage["frac_correct"].to_numpy(), stage["frac_alerted"].to_numpy())
    stage.to_csv(C.ART_DIR / "stage_eval.csv", index=False)

    # ---------------- chain-level
    rows = []
    for cid, g in stage.sort_values("stage_idx").groupby("chain_id", sort=False):
        n = len(g)
        n_det, n_par = (g.status == "DETECTED").sum(), (g.status == "PARTIAL").sum()
        strict, vis = n_det / n, (g.frac_alerted >= C.STAGE_ALERTED).mean()
        nd = g[g.status != "DETECTED"]
        ms = g[g.status == "MISSED"]
        rows.append(dict(
            chain_id=cid, batch=int(g.batch.iloc[0]), template=g.template.iloc[0], n_stages=n,
            n_detected=int(n_det), n_partial=int(n_par),
            n_mislabeled=int((g.status == "MISLABELED").sum()), n_missed=int(len(ms)),
            strict_coverage=strict, weighted_coverage=(n_det + 0.5 * n_par) / n,
            visibility_coverage=vis, detection_gap_pct=100 * (1 - strict),
            attribution_gap_pct=100 * (vis - strict),
            breakpoint_stage=nd.stage_name.iloc[0] if len(nd) else "none",
            visibility_break_stage=ms.stage_name.iloc[0] if len(ms) else "none",
            pattern="-".join(LETTER[s] for s in g.status)))
    chains = pd.DataFrame(rows)
    chains.to_csv(C.ART_DIR / "chain_eval.csv", index=False)

    by_cat = pd.crosstab([stage.batch, stage.unsw_category], stage.status)
    by_cat.to_csv(C.ART_DIR / "status_by_category.csv")

    # ---------------- summary
    summary = {"settings": {"alert_rule": rule, "alert_threshold": thr, "stage_detected": C.STAGE_DETECTED,
                            "stage_partial": C.STAGE_PARTIAL, "stage_alerted": C.STAGE_ALERTED}}
    for b, g in chains.groupby("batch"):
        s = stage[stage.batch == b]
        summary[f"batch{b}"] = {
            "n_chains": int(len(g)), "n_stages": int(len(s)),
            "mean_strict_coverage": float(g.strict_coverage.mean()),
            "mean_weighted_coverage": float(g.weighted_coverage.mean()),
            "mean_visibility_coverage": float(g.visibility_coverage.mean()),
            "mean_detection_gap_pct": float(g.detection_gap_pct.mean()),
            "mean_attribution_gap_pct": float(g.attribution_gap_pct.mean()),
            "fully_detected_chains_pct": float(100 * (g.n_detected == g.n_stages).mean()),
            "stage_status_counts": s.status.value_counts().to_dict(),
            "breakpoint_histogram": g.breakpoint_stage.value_counts().to_dict(),
            "visibility_break_histogram": g.visibility_break_stage.value_counts().to_dict(),
        }
    (C.ART_DIR / "gap_summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- console report
    print("\n=== FLOW-LEVEL (all flows) ===")
    for b, m in fm.items():
        note = " [in training period - in-sample]" if b == "batch1" else " [future - held out]"
        print(f"{b}{note}: macro-F1 {m['macro_f1']:.3f} | weighted-F1 {m['weighted_f1']:.3f} | "
              f"attack-vs-normal recall {m['attack_vs_normal_recall']:.4f} | "
              f"normal FPR {m['normal_false_positive_rate']:.4f}")
        print("   per-class recall:", {k: round(v, 2) for k, v in m["per_class_recall"].items()})
    print("\n=== CHAIN-LEVEL ===")
    for b in sorted(chains.batch.unique()):
        s = summary[f"batch{b}"]
        tag = "(in training period - optimistic)" if b == 1 else "(future - honest result)"
        print(f"batch {b} {tag}: {s['n_chains']} chains / {s['n_stages']} stages")
        print(f"   strict coverage {s['mean_strict_coverage']:.1%} | visibility {s['mean_visibility_coverage']:.1%} "
              f"| detection gap {s['mean_detection_gap_pct']:.1f}% | attribution gap {s['mean_attribution_gap_pct']:.1f}%")
        print(f"   stage status: {s['stage_status_counts']}")
        print(f"   breakpoints: {s['breakpoint_histogram']}")
    print("\nstatus by batch / category:\n", by_cat.to_string())
    print("\nexample chain:")
    ex = stage[stage.chain_id == chains.chain_id.iloc[-1]][
        ["stage_idx", "stage_name", "unsw_category", "frac_correct", "top_pred", "status"]]
    print(ex.round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--alert-rule", choices=["argmax", "pattack"], default=C.ALERT_RULE)
    ap.add_argument("--alert-thr", type=float, default=C.ALERT_THRESHOLD,
                    help="P(attack) threshold, used only with --alert-rule pattack")
    a = ap.parse_args()
    main(a.alert_rule, a.alert_thr)
