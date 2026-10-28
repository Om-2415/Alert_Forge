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
from gap_engine import chain_table, effective_pred, stage_table, summarize

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

    # ---------------- stage-level + chain-level (shared engine)
    truth = pd.read_csv(C.ART_DIR / "chain_truth.csv")
    chain_flows = pd.read_csv(C.ART_DIR / "chain_flows.csv")
    stage = stage_table(truth, chain_flows, df)
    stage.to_csv(C.ART_DIR / "stage_eval.csv", index=False)
    chains = chain_table(stage)
    chains.to_csv(C.ART_DIR / "chain_eval.csv", index=False)
    by_cat = pd.crosstab([stage.batch, stage.unsw_category], stage.status)
    by_cat.to_csv(C.ART_DIR / "status_by_category.csv")

    summary = {"settings": {"alert_rule": rule, "alert_threshold": thr,
                            "stage_detected": C.STAGE_DETECTED, "stage_partial": C.STAGE_PARTIAL,
                            "stage_alerted": C.STAGE_ALERTED}}
    summary.update(summarize(chains, stage))
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
