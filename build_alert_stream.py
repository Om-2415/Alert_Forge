"""
Phase 2 - build the SOC-style alert stream: a few hidden ground-truth attack
chains buried in thousands of decoy and benign events (the "3,000 alerts" setting).

Every event is a REAL UNSW-NB15 flow. The EVENT SET (which flows, which chains)
depends only on --seed / --chains / --events / --decoy-attack-share / --batch.
The DETECTOR applied to it is a separate choice:

    (default)           the frozen baseline model
    --scenario NAME     a blind-spot scenario from config.BLIND_SPOT_SCENARIOS
                        (e.g. EV1_mimic_volume_postexploit, BS2_no_persist_exec_rules)

so the SAME events can be replayed through different detectors. Each stream stores a
fingerprint of its event set; identical fingerprint = identical events.

Outputs (suffix _<tag>, tag defaults to the scenario name):
    alert_stream*.csv       all events + ground truth (kind, chain_id, stage) + detection
    alerts*.csv             only the events the detector alerted on  (input to Phase 3)
    stream_stage_eval*.csv  stage-by-stage ground truth vs detection for the hidden chains
    stream_summary*.json

Event kinds: chain | decoy_attack (real attack flows outside any chain) | benign
(Normal flows; alerts on these are false positives).

Difficulty axis (Phase 3): --chain-attackers K and --same-attacker-decoy-share S control how
separable the noise is by entity. Report ALL levels; do not pick the flattering one:
    hard    (default)                                       all noise may share the chains' attackers
    medium  --chain-attackers 2 --same-attacker-decoy-share 0.5
    easy    --chain-attackers 2 --same-attacker-decoy-share 0.0

Usage:
    python build_alert_stream.py --events 8475 --alert-rule pattack --alert-thr 0.99
    python build_alert_stream.py --events 8475 --alert-rule pattack --alert-thr 0.99 \
           --scenario EV1_mimic_volume_postexploit --normal-cap 300000
    python build_alert_stream.py --target-alerts 3000 ...   # sizes the stream with the BASELINE detector
For comparable streams, reuse the same --events (printed on each run) and --seed.
"""
import argparse, hashlib, json
import numpy as np
import pandas as pd
import config as C
from common import load_df
from gap_engine import chain_table, effective_pred, stage_table
from scenarios import make_context, scenario_predict


def pick_chains(truth, n, rng, batch, k_attackers=None):
    t = truth[truth.batch == batch]
    starts, tmpl = t.groupby("chain_id").t_start.min(), t.groupby("chain_id").template.first()
    att = t.groupby("chain_id").attacker.first()
    anchor = float(starts.iloc[rng.integers(len(starts))])
    width = 3600.0
    while True:                                   # chains that start near the anchor time
        cand = starts[(starts >= anchor - width) & (starts <= anchor + width)].index
        if k_attackers:                           # optional: chains run by at most k attacker IPs
            top = set(att[cand].value_counts().index[:k_attackers])
            cand = [c for c in cand if att[c] in top]
        if len(cand) >= n or width > 14 * 86400:
            break
        width *= 2
    cdf = pd.DataFrame({"chain_id": cand, "template": tmpl[cand].to_numpy()}).sample(
        frac=1, random_state=int(rng.integers(1_000_000)))
    queues = [list(g.chain_id) for _, g in cdf.groupby("template")]
    order = []
    while any(queues) and len(order) < n:         # round-robin over templates -> diverse chains
        for q in queues:
            if q and len(order) < n:
                order.append(q.pop())
    return order


def attach(ev, det):
    """Add detection columns. det: flow_id, eff (alert label), p_attack, p_max."""
    ev = ev.merge(det[["flow_id", "eff", "p_attack", "p_max"]], on="flow_id").sort_values("event_id")
    ev = ev.rename(columns={"eff": "detected_as"})
    ev["alerted"] = ev["detected_as"] != "Normal"
    return ev


def main(a):
    rng = np.random.default_rng(a.seed)
    scenario = None
    if a.scenario:
        if a.scenario not in C.BLIND_SPOT_SCENARIOS:
            raise SystemExit(f"unknown scenario {a.scenario}; choose from {list(C.BLIND_SPOT_SCENARIOS)}")
        scenario = C.BLIND_SPOT_SCENARIOS[a.scenario]
    tag = a.tag if a.tag is not None else (a.scenario or "")
    sfx = f"_{tag}" if tag else ""

    flows_full = load_df(C.ART_DIR / "flows")
    flows = flows_full[["flow_id", "srcip", "dstip", "Stime", "attack_cat", "batch"]].copy()
    flows["attack_cat"] = flows["attack_cat"].astype(str)
    flows["srcip"], flows["dstip"] = flows["srcip"].astype(str), flows["dstip"].astype(str)
    truth = pd.read_csv(C.ART_DIR / "chain_truth.csv")
    cflows = pd.read_csv(C.ART_DIR / "chain_flows.csv")

    # baseline detections: used to size the stream (and as the detector unless --scenario)
    pred = load_df(C.ART_DIR / "predictions")
    classes = sorted(c[2:] for c in pred.columns if c.startswith("p_") and c != "p_max")
    pred["eff"] = effective_pred(pred, classes, a.alert_rule, a.alert_thr)
    pred["p_attack"] = 1.0 - pred["p_Normal"]

    chosen = pick_chains(truth, a.chains, rng, a.batch, a.chain_attackers)
    tsel = truth[truth.chain_id.isin(chosen)]
    t0, t1 = tsel.t_start.min() - a.pad_s, tsel.t_end.max() + a.pad_s

    ev_chain = (cflows[cflows.chain_id.isin(chosen)].merge(flows, on="flow_id").assign(kind="chain"))
    pool = flows[(flows.batch == a.batch) & flows.Stime.between(t0, t1) & ~flows.flow_id.isin(cflows.flow_id)]
    atk, nrm = pool[pool.attack_cat != "Normal"], pool[pool.attack_cat == "Normal"]

    chain_attackers = set(tsel.attacker.astype(str).unique())
    atk_same, atk_other = atk[atk.srcip.isin(chain_attackers)], atk[~atk.srcip.isin(chain_attackers)]

    def draw_decoy_attacks(n_atk):
        """Default (None): any attack flow, which is the worst case because UNSW has only 4 attacker
        IPs. Otherwise a fixed share comes from attackers that also run a hidden chain."""
        if a.same_attacker_decoy_share is None:
            return atk.sample(n_atk, random_state=a.seed)
        n_same = min(int(round(a.same_attacker_decoy_share * n_atk)), len(atk_same))
        n_other = min(n_atk - n_same, len(atk_other))
        return pd.concat([atk_same.sample(n_same, random_state=a.seed),
                          atk_other.sample(n_other, random_state=a.seed)])

    def assemble(events):
        n_decoy = max(events - len(ev_chain), 0)
        n_atk = min(int(round(a.decoy_attack_share * n_decoy)), len(atk))
        n_nrm = min(n_decoy - n_atk, len(nrm))
        ev = pd.concat([ev_chain,
                        draw_decoy_attacks(n_atk).assign(kind="decoy_attack"),
                        nrm.sample(n_nrm, random_state=a.seed).assign(kind="benign")], ignore_index=True)
        ev["shares_attacker_with_chain"] = ev["srcip"].isin(chain_attackers)
        ev = ev.sort_values("Stime").reset_index(drop=True)
        ev.insert(0, "event_id", np.arange(len(ev)))
        return ev.rename(columns={"attack_cat": "truth_category"}), n_nrm

    events = a.events
    ev0, n_nrm = assemble(events)
    if a.target_alerts:                       # size the stream with the BASELINE detector
        for _ in range(5):
            n_al = int(attach(ev0, pred).alerted.sum())
            if abs(n_al - a.target_alerts) <= 0.02 * a.target_alerts or len(ev0) >= len(pool) + len(ev_chain):
                break
            events = int(events * a.target_alerts / max(n_al, 1))
            ev0, n_nrm = assemble(events)

    # ---- detector: frozen baseline, or a blind-spot scenario applied to the SAME events
    if scenario is None:
        det = pred
    else:
        rows = flows_full[flows_full.flow_id.isin(ev0.flow_id)].copy()
        rows["attack_cat"] = rows["attack_cat"].astype(str)
        ctx = make_context(flows_full, a.normal_cap)
        if scenario["kind"] == "catalog" and ctx["normal_cap"] is None:
            print("note: --normal-cap not given and not stored in the model file; "
                  "using ALL Normal rows for retraining (must match train_model.py)")
        det = scenario_predict(ctx, scenario, rows)
        det["flow_id"] = rows["flow_id"].to_numpy()
        det["eff"] = effective_pred(det, classes, a.alert_rule, a.alert_thr)
        det["p_attack"] = 1.0 - det["p_Normal"]
    ev = attach(ev0, det)
    fingerprint = hashlib.sha1(np.sort(ev["flow_id"].to_numpy()).tobytes()).hexdigest()[:12]

    ev.to_csv(C.ART_DIR / f"alert_stream{sfx}.csv", index=False)
    ev[ev.alerted].to_csv(C.ART_DIR / f"alerts{sfx}.csv", index=False)

    df = ev[ev.kind == "chain"][["flow_id", "truth_category", "detected_as", "p_max"]].rename(
        columns={"truth_category": "attack_cat", "detected_as": "eff"})
    stage = stage_table(tsel, cflows[cflows.chain_id.isin(chosen)], df)
    chains = chain_table(stage)
    stage.to_csv(C.ART_DIR / f"stream_stage_eval{sfx}.csv", index=False)

    al = ev[ev.alerted]
    summ = {"scenario": a.scenario, "batch": a.batch, "event_set_fingerprint": fingerprint,
            "events": int(len(ev)), "events_param": int(events), "alerts": int(len(al)),
            "hidden_chains": chosen, "alert_rule": a.alert_rule, "alert_threshold": a.alert_thr,
            "seed": a.seed, "decoy_attack_share": a.decoy_attack_share,
            "chain_attackers": sorted(chain_attackers),
            "same_attacker_decoy_share": a.same_attacker_decoy_share,
            "decoy_alerts_sharing_chain_attacker": float(al[al.kind == "decoy_attack"].shares_attacker_with_chain.mean())
            if (al.kind == "decoy_attack").any() else None,
            "alerts_by_kind": al.kind.value_counts().to_dict(),
            "events_by_kind": ev.kind.value_counts().to_dict(),
            "benign_false_positive_rate": float(ev[ev.kind == "benign"].alerted.mean()) if n_nrm else None,
            "chain_stage_status": stage.status.value_counts().to_dict(),
            "mean_strict_coverage": float(chains.strict_coverage.mean()),
            "time_window_hours": float((t1 - t0) / 3600)}
    (C.ART_DIR / f"stream_summary{sfx}.json").write_text(json.dumps(summ, indent=2))

    print(f"detector: {a.scenario or 'frozen baseline'} | alert rule {a.alert_rule}"
          f"{' @ ' + str(a.alert_thr) if a.alert_rule == 'pattack' else ''}")
    print(f"stream: {len(ev):,} events (use --events {events} to reproduce) over {summ['time_window_hours']:.1f} h  "
          f"{summ['events_by_kind']}  | event-set fingerprint {fingerprint}")
    print(f"alerts: {len(al):,}  {summ['alerts_by_kind']}  ({len(al) / len(chosen):.0f} alerts per hidden chain)")
    if summ["benign_false_positive_rate"] is not None:
        print(f"benign false-positive rate in stream: {summ['benign_false_positive_rate']:.2%}")
    if summ["decoy_alerts_sharing_chain_attacker"] is not None:
        print(f"decoy-attack alerts sharing an attacker IP with a hidden chain: "
              f"{summ['decoy_alerts_sharing_chain_attacker']:.1%}  (chain attackers: {sorted(chain_attackers)})")
    print(f"hidden chains: {len(chosen)} {chosen}")
    print(f"chain stage status: {summ['chain_stage_status']} | mean strict coverage {summ['mean_strict_coverage']:.1%}")
    if a.verbose:
        print("\nstage-by-stage (ground truth vs detected), hidden chains:")
        print(stage[["chain_id", "stage_idx", "unsw_category", "n_flows", "frac_correct", "top_pred",
                     "status", "status3"]].round(2).to_string(index=False))
        print("\nchain pattern (D=detected P=partial L=mislabeled M=missed):")
        print(chains[["chain_id", "template", "pattern", "strict_coverage", "breakpoint_stage"]]
              .round(2).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--chains", type=int, default=6)
    ap.add_argument("--events", type=int, default=3000)
    ap.add_argument("--target-alerts", type=int, default=None,
                    help="resize the stream so the BASELINE detector yields about this many alerts")
    ap.add_argument("--decoy-attack-share", type=float, default=0.35,
                    help="share of decoy events that are real attack flows outside any chain")
    ap.add_argument("--chain-attackers", type=int, default=None,
                    help="difficulty: restrict the hidden chains to at most K attacker IPs (default: any)")
    ap.add_argument("--same-attacker-decoy-share", type=float, default=None,
                    help="difficulty: share of decoy attack events from attackers that also run a hidden "
                         "chain. Default (unset) = any attack flow (worst case, 4 attackers in total)")
    ap.add_argument("--pad-s", type=float, default=1800, help="time padding around the chains (s)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--alert-rule", choices=["argmax", "pattack"], default=C.ALERT_RULE)
    ap.add_argument("--alert-thr", type=float, default=C.ALERT_THRESHOLD)
    ap.add_argument("--scenario", default=None, help="apply a blind-spot scenario as the detector")
    ap.add_argument("--normal-cap", type=int, default=None,
                    help="catalog scenarios only: must equal the value used in train_model.py")
    ap.add_argument("--tag", default=None, help="output suffix (default: scenario name)")
    ap.add_argument("--verbose", action="store_true", help="also print the stage-by-stage tables")
    main(ap.parse_args())
