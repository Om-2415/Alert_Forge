"""
Phase 2 - controlled blind-spot experiment (fault injection into the detector).

Question: if a detection capability is really missing, does the stage-level gap
analysis (a) find it, (b) localise where visibility breaks, and (c) do so while
aggregate ML metrics barely move?

Arms (all use the SAME Batch-1 training rows, hyper-parameters and seed):
  EVx    evasion scenarios: attacker blends behaviour features toward benign flows
  CTRL   no intervention (must reproduce the frozen model - checked below)
  BSx    catalog scenarios: class relabelled Normal ("mask") or removed ("withhold")
  TLx    telemetry scenarios: feature groups zeroed at inference

HONESTY NOTE: "mask" scenarios miss their stages BY CONSTRUCTION. They validate the
measurement instrument (known blind spot -> is it recovered?). "withhold" and
"telemetry" scenarios are EMPIRICAL: nothing forces the model to miss anything.
Report them separately.

Outputs (artifacts/blindspot[_TAG]/): scenario_summary.csv, scenario_stage_eval.csv,
                                scenario_chain_eval.csv, blindspot_report.json
Usage:
    python blindspot_experiment.py --normal-cap 300000 --alert-rule pattack --alert-thr 0.5  --tag thr05
    python blindspot_experiment.py --normal-cap 300000 --alert-rule pattack --alert-thr 0.99 --tag thr099
        (--normal-cap must equal the value used in train_model.py)
    python blindspot_experiment.py --scenarios BS1_no_backdoor_rule TL1_conn_tracking_lost
    python blindspot_experiment.py --n-estimators 60        # fast dev run
"""
import argparse, json, time
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import config as C
from common import build_features, load_df, training_frame
from gap_engine import chain_table, effective_pred, stage_table, summarize
from scenarios import (apply_evasion, injection_label, planted_categories, predict_frame, train_arm)
from train_model import make_rf

def agg_metrics(true, pred, classes):
    att, nrm = true != "Normal", true == "Normal"
    return {"accuracy": float(accuracy_score(true, pred)),
            "weighted_f1": float(f1_score(true, pred, average="weighted", zero_division=0)),
            "macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
            "attack_vs_normal_recall": float((pred[att] != "Normal").mean()),
            "normal_fpr": float((pred[nrm] != "Normal").mean())}


def main(a):
    t0 = time.time()
    out_dir = C.ART_DIR / ("blindspot" + (f"_{a.tag}" if a.tag else ""))
    out_dir.mkdir(exist_ok=True)
    bundle = joblib.load(C.ART_DIR / "rf_batch1_frozen.joblib")
    classes, enc, params = bundle["classes"], bundle["encoders"], dict(bundle["params"])
    if a.n_estimators:
        params["n_estimators"] = a.n_estimators
    normal_cap = a.normal_cap if a.normal_cap is not None else bundle.get("normal_cap")
    print(f"params: {params} | normal_cap: {normal_cap}")

    flows = load_df(C.ART_DIR / "flows")
    truth = pd.read_csv(C.ART_DIR / "chain_truth.csv")
    cflows = pd.read_csv(C.ART_DIR / "chain_flows.csv")

    # ---- training rows: identical to train_model.py
    b1 = training_frame(flows, normal_cap, C.SEED)
    Xtr, ytr = build_features(b1, enc), b1["attack_cat"].astype(str).to_numpy()

    # ---- evaluation frame: every chain flow + a natural-mix sample of batch 2
    b2 = flows[flows["batch"] == 2]
    bg = b2.sample(min(a.bg_sample, len(b2)), random_state=C.SEED)
    ev = flows[flows["flow_id"].isin(set(cflows["flow_id"]) | set(bg["flow_id"]))].copy()
    ev["attack_cat"] = ev["attack_cat"].astype(str)
    ev = ev.reset_index(drop=True)
    Xev = build_features(ev, enc)
    is_bg = ev["flow_id"].isin(set(bg["flow_id"])).to_numpy()
    print(f"eval frame: {len(ev):,} flows ({len(set(cflows['flow_id'])):,} chain flows, "
          f"{is_bg.sum():,} background sample)")

    arms = {"CTRL": None}
    arms.update({k: v for k, v in C.BLIND_SPOT_SCENARIOS.items()
                 if not a.scenarios or k in a.scenarios})
    ctrl_model, ctrl_stage, results = None, None, {}
    summary_rows, stage_frames, chain_frames = [], [], []

    for name, sc in arms.items():
        t1 = time.time()
        if sc is None or sc["kind"] == "catalog":
            model = train_arm(Xtr, ytr, params, sc)
            if name == "CTRL":
                ctrl_model = model
        else:
            model = ctrl_model
        X = Xev
        if sc and sc["kind"] == "evasion":
            X = apply_evasion(Xev, ev["attack_cat"].to_numpy(), Xtr[ytr == "Normal"], sc, C.SEED)
        if sc and sc["kind"] == "telemetry":
            X = Xev.copy()
            cols = sorted({c for g in sc["groups"] for c in C.TELEMETRY_GROUPS[g]})
            X[cols] = 0.0
        pred = predict_frame(model, X, classes)
        pred["eff"] = effective_pred(pred, classes, a.alert_rule, a.alert_thr)

        if name == "CTRL" and not a.n_estimators:
            # The frozen model was fitted on integer class codes (0..9); map them to names
            fm = bundle["model"]
            f_names = [c if isinstance(c, str) else classes[int(c)] for c in fm.classes_]
            p_frozen = fm.predict_proba(Xev)
            p_ctrl = pred[[f"p_{c}" for c in f_names]].to_numpy()
            agree = float((np.array(f_names, dtype=object)[p_frozen.argmax(1)]
                           == pred["pred_class"].to_numpy()).mean())
            max_dp = float(np.abs(p_frozen - p_ctrl).max())
            print(f"CTRL vs frozen model: same predicted class on {agree:.2%} of eval flows, "
                  f"max |dP| = {max_dp:.4f}"
                  + ("" if agree > 0.999 else "  <-- MISMATCH: check --normal-cap matches train_model.py"))

        df = pd.DataFrame({"flow_id": ev["flow_id"], "attack_cat": ev["attack_cat"],
                           "eff": pred["eff"].to_numpy(), "p_max": pred["p_max"].to_numpy()})
        stage = stage_table(truth, cflows, df)
        chains = chain_table(stage)
        planted = planted_categories(sc)
        stage["planted"] = stage["unsw_category"].isin(planted) if planted else (sc is not None)
        if name == "CTRL":
            ctrl_stage = stage.set_index(["chain_id", "stage_idx"])["status"]
        stage["scenario"], chains["scenario"] = name, name
        stage_frames.append(stage), chain_frames.append(chains)

        flow_agg = agg_metrics(df["attack_cat"].to_numpy()[is_bg], df["eff"].to_numpy()[is_bg], classes)
        results[name] = {"stage": stage, "chains": chains, "flow": flow_agg, "sc": sc}
        print(f"  [{name}] done in {time.time() - t1:.0f}s", flush=True)

    # ---- derived metrics, per scenario x batch
    ctrl = results["CTRL"]
    for name, r in results.items():
        st, ch, sc = r["stage"], r["chains"], r["sc"]
        for b in (1, 2):
            s, c = st[st.batch == b], ch[ch.batch == b]
            row = {"scenario": name, "batch": b,
                   "injection": injection_label(sc),
                   "n_chains": len(c), "n_stages": len(s)}
            p = s[s["planted"] == True] if sc else s.iloc[0:0]
            u = s[s["planted"] != True]
            row.update({"planted_stages": len(p),
                        "planted_missed_pct": 100 * (p.status == "MISSED").mean() if len(p) else np.nan,
                        "planted_not_detected_pct": 100 * (p.status != "DETECTED").mean() if len(p) else np.nan})
            has_targets = planted_categories(sc) is not None
            if sc and has_targets and len(u):
                base = ctrl_stage.reindex(pd.MultiIndex.from_frame(u[["chain_id", "stage_idx"]]))
                row["collateral_delta_pp"] = 100 * ((u.status != "DETECTED").mean()
                                                    - (base.to_numpy() != "DETECTED").mean())
            else:
                row["collateral_delta_pp"] = np.nan
            row["strict_coverage"] = 100 * c.strict_coverage.mean()
            row["delta_vs_ctrl_pp"] = row["strict_coverage"] - 100 * ctrl["chains"][ctrl["chains"].batch == b].strict_coverage.mean()
            row["chains_with_missed_stage_pct"] = 100 * (c.n_missed > 0).mean()
            # breakpoint recovery: first MISSED stage == first planted stage in the chain
            if sc and has_targets:
                fp = (s[s["planted"] == True].sort_values("stage_idx").groupby("chain_id")["unsw_category"].first())
                got = c.set_index("chain_id")["visibility_break_category"].reindex(fp.index)
                row["breakpoint_recovery_pct"] = 100 * (got == fp).mean() if len(fp) else np.nan
                row["chains_with_planted_stage"] = len(fp)
            else:
                row["breakpoint_recovery_pct"], row["chains_with_planted_stage"] = np.nan, 0
            for k in ("weighted_f1", "macro_f1", "normal_fpr", "attack_vs_normal_recall"):
                row[f"flow_{k}"] = r["flow"][k] if b == 2 else np.nan   # background sample is batch 2
            row["delta_weighted_f1"] = (r["flow"]["weighted_f1"] - ctrl["flow"]["weighted_f1"]) if b == 2 else np.nan
            summary_rows.append(row)

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / "scenario_summary.csv", index=False)
    pd.concat(stage_frames).to_csv(out_dir / "scenario_stage_eval.csv", index=False)
    pd.concat(chain_frames).to_csv(out_dir / "scenario_chain_eval.csv", index=False)
    report = {"settings": {"alert_rule": a.alert_rule, "alert_threshold": a.alert_thr,
                           "normal_cap": normal_cap, "params": params, "bg_sample": a.bg_sample},
              "scenarios": {k: v for k, v in C.BLIND_SPOT_SCENARIOS.items() if k in results},
              "summary": json.loads(summ.to_json(orient="records"))}
    (out_dir / "blindspot_report.json").write_text(json.dumps(report, indent=2))

    # ---- console report
    pd.set_option("display.width", 250)
    t = summ[summ.batch == 2].set_index("scenario").copy()
    t["flow_normal_fpr"] *= 100
    t = t.rename(columns={"planted_stages": "planted", "planted_missed_pct": "miss%",
                          "planted_not_detected_pct": "notDet%", "collateral_delta_pp": "collat_dpp",
                          "breakpoint_recovery_pct": "bkpt_rec%", "strict_coverage": "strict%",
                          "delta_vs_ctrl_pp": "d_strict", "chains_with_missed_stage_pct": "ch>=1miss%",
                          "flow_weighted_f1": "wF1", "delta_weighted_f1": "d_wF1",
                          "flow_macro_f1": "mF1", "flow_normal_fpr": "FPR%"})
    t["injection"] = t["injection"].str.split(" ").str[0]
    cols = ["injection", "planted", "miss%", "notDet%", "collat_dpp", "bkpt_rec%", "strict%", "d_strict",
            "ch>=1miss%", "wF1", "d_wF1", "mF1", "FPR%"]
    print(f"\n=== BLIND-SPOT RESULTS - batch 2 (future, held-out) | alert rule {a.alert_rule}"
          f"{' @ ' + str(a.alert_thr) if a.alert_rule == 'pattack' else ''} ===")
    print(t[cols].round(2).to_string())
    print("columns: planted = stages hit by the injection | miss% / notDet% = planted stages MISSED / not DETECTED | "
          "collat_dpp = extra non-detected on UNplanted stages vs CTRL | bkpt_rec% = first MISSED stage == first planted stage | "
          "strict% = mean chain coverage | wF1/mF1/FPR% = flow-level on a natural-mix batch-2 sample")
    print("\nstatus counts per scenario (batch 2):")
    for name, r in results.items():
        s = r["stage"]; s = s[s.batch == 2]
        print(f"  {name:<28}", s.status.value_counts().reindex(["DETECTED", "PARTIAL", "MISLABELED", "MISSED"]).fillna(0).astype(int).to_dict())
    print(f"\nsaved to {out_dir}  (total {time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=None)
    ap.add_argument("--normal-cap", type=int, default=None,
                    help="must equal the value used in train_model.py (default: read from model file)")
    ap.add_argument("--bg-sample", type=int, default=100_000,
                    help="natural-mix batch-2 sample for aggregate flow metrics")
    ap.add_argument("--n-estimators", type=int, default=None, help="dev only: fewer trees")
    ap.add_argument("--tag", default="", help="output folder suffix, e.g. --tag thr099 -> artifacts/blindspot_thr099")
    ap.add_argument("--alert-rule", choices=["argmax", "pattack"], default=C.ALERT_RULE)
    ap.add_argument("--alert-thr", type=float, default=C.ALERT_THRESHOLD)
    main(ap.parse_args())
