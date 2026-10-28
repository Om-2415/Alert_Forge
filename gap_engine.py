"""
Detection-gap engine (Phase 2): the single place where GROUND TRUTH is compared
with DETECTED EVENTS, stage by stage.

    chain_truth + chain_flows   (what really happened, per stage)
                 +
    per-flow detections         (what the detector alerted on, after the alert rule)
                 =
    stage table  ->  chain table  ->  summary

Used by evaluate_gap.py (baseline) and blindspot_experiment.py (scenarios), so
every number in the project is produced by identical logic.

Stage status (thresholds in config.py):
    DETECTED    frac_correct >= STAGE_DETECTED         right stage named
    PARTIAL     frac_correct >= STAGE_PARTIAL
    MISLABELED  frac_alerted >= STAGE_ALERTED          seen, named as another class
    MISSED      otherwise                              effectively invisible
Simple 3-way view (status3): DETECTED / PARTIAL (= PARTIAL or MISLABELED) / MISSED
"""
import numpy as np
import pandas as pd
import config as C

LETTER = {"DETECTED": "D", "PARTIAL": "P", "MISLABELED": "L", "MISSED": "M"}
STATUS3 = {"DETECTED": "DETECTED", "PARTIAL": "PARTIAL", "MISLABELED": "PARTIAL", "MISSED": "MISSED"}


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


def stage_status(frac_correct, frac_alerted):
    return np.select(
        [frac_correct >= C.STAGE_DETECTED, frac_correct >= C.STAGE_PARTIAL, frac_alerted >= C.STAGE_ALERTED],
        ["DETECTED", "PARTIAL", "MISLABELED"], default="MISSED")


def stage_table(truth, chain_flows, flow_df):
    """truth: chain_truth rows; chain_flows: chain_id,stage_idx,flow_id;
    flow_df: flow_id, attack_cat (true), eff (alert label), p_max."""
    cf = chain_flows.merge(flow_df[["flow_id", "attack_cat", "eff", "p_max"]], on="flow_id")
    cf["correct"] = cf["eff"] == cf["attack_cat"]
    cf["alerted"] = cf["eff"] != "Normal"
    key = ["chain_id", "stage_idx"]
    agg = cf.groupby(key).agg(frac_correct=("correct", "mean"), frac_alerted=("alerted", "mean"),
                              mean_p_max=("p_max", "mean")).reset_index()
    top = (cf.groupby(key)["eff"].agg(lambda s: s.value_counts().index[0])
           .rename("top_pred").reset_index())
    st = truth.merge(agg, on=key).merge(top, on=key)
    st["status"] = stage_status(st["frac_correct"].to_numpy(), st["frac_alerted"].to_numpy())
    st["status3"] = st["status"].map(STATUS3)
    return st


def chain_table(stage):
    rows = []
    for cid, g in stage.sort_values("stage_idx").groupby("chain_id", sort=False):
        n = len(g)
        n_det, n_par = (g.status == "DETECTED").sum(), (g.status == "PARTIAL").sum()
        strict, vis = n_det / n, (g.frac_alerted >= C.STAGE_ALERTED).mean()
        nd, ms = g[g.status != "DETECTED"], g[g.status == "MISSED"]
        rows.append(dict(
            chain_id=cid, batch=int(g.batch.iloc[0]), template=g.template.iloc[0], n_stages=n,
            n_detected=int(n_det), n_partial=int(n_par),
            n_mislabeled=int((g.status == "MISLABELED").sum()), n_missed=int(len(ms)),
            strict_coverage=strict, weighted_coverage=(n_det + 0.5 * n_par) / n,
            visibility_coverage=vis, detection_gap_pct=100 * (1 - strict),
            attribution_gap_pct=100 * (vis - strict),
            breakpoint_stage=nd.stage_name.iloc[0] if len(nd) else "none",
            visibility_break_stage=ms.stage_name.iloc[0] if len(ms) else "none",
            visibility_break_category=ms.unsw_category.iloc[0] if len(ms) else "none",
            pattern="-".join(LETTER[s] for s in g.status)))
    return pd.DataFrame(rows)


def summarize(chains, stage):
    out = {}
    for b, g in chains.groupby("batch"):
        s = stage[stage.batch == b]
        out[f"batch{b}"] = {
            "n_chains": int(len(g)), "n_stages": int(len(s)),
            "mean_strict_coverage": float(g.strict_coverage.mean()),
            "mean_weighted_coverage": float(g.weighted_coverage.mean()),
            "mean_visibility_coverage": float(g.visibility_coverage.mean()),
            "mean_detection_gap_pct": float(g.detection_gap_pct.mean()),
            "mean_attribution_gap_pct": float(g.attribution_gap_pct.mean()),
            "fully_detected_chains_pct": float(100 * (g.n_detected == g.n_stages).mean()),
            "chains_with_missed_stage_pct": float(100 * (g.n_missed > 0).mean()),
            "stage_status_counts": s.status.value_counts().to_dict(),
            "stage_status3_counts": s.status3.value_counts().to_dict(),
            "breakpoint_histogram": g.breakpoint_stage.value_counts().to_dict(),
            "visibility_break_histogram": g.visibility_break_stage.value_counts().to_dict(),
        }
    return out
