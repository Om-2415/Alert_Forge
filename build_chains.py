"""
Step 3 - build ground-truth attack chains by STITCHING REAL UNSW FLOWS.

UNSW-NB15 attack traffic is a concurrent stress-test, not a staged campaign,
so stage ORDER is imposed (templates in config.CHAIN_TEMPLATES) while the
flows, IPs and timestamps inside every stage are real.

For each chain: pick an attacker (config.CHAIN_ENTITY = "attacker") or an
attacker->victim pair ("pair") and a start flow; for each stage in the
template take the next run of REAL flows of that category for the same entity
(after the previous stage ended; no flow is reused across chains).

Outputs (artifacts/):
    chain_truth.csv  one row per stage  (chain_id, batch, template, stage_idx,
                     stage_name, tactic, unsw_category, attacker, victim,
                     t_start, t_end, n_flows)
    chain_flows.csv  chain_id, stage_idx, flow_id   (which real flows)

Phase 2 changes: fixed quota PER TEMPLATE (no silent refilling with easy
templates), variable stage sizes drawn per category, shortfall report.

Usage:
    python build_chains.py                  # CHAIN_QUOTA_PER_TEMPLATE per template per batch
    python build_chains.py --quota 10
"""
import argparse
import numpy as np
import pandas as pd
import config as C
from common import load_df


def build_groups(att: pd.DataFrame):
    """key = (attacker, victim-or-'*', category) -> (times, flow_ids, victims)."""
    by_pair = C.CHAIN_ENTITY == "pair"
    keys = ["srcip", "dstip", "attack_cat"] if by_pair else ["srcip", "attack_cat"]
    groups = {}
    for k, g in att.groupby(keys, observed=True, sort=False):
        s, c = str(k[0]), str(k[-1])
        d = str(k[1]) if by_pair else "*"
        groups[(s, d, c)] = (g["Stime"].to_numpy(), g["flow_id"].to_numpy(), g["dstip"].astype(str).to_numpy())
    return groups


def try_chain(rng, groups, pairs, template, used):
    src, dst = pairs[rng.integers(len(pairs))]
    if any((src, dst, cat) not in groups for cat in template):
        return None
    t0_arr = groups[(src, dst, template[0])][0]
    cursor = t0_arr[rng.integers(len(t0_arr))]
    stages = []
    for k, cat in enumerate(template):
        t_arr, id_arr, v_arr = groups[(src, dst, cat)]
        i = int(np.searchsorted(t_arr, cursor, side="left"))
        while i < len(t_arr) and id_arr[i] in used:
            i += 1
        if i >= len(t_arr):
            return None
        if k > 0 and t_arr[i] - cursor > C.CHAIN_MAX_GAP_S:
            return None
        cap = C.CHAIN_MAX_FLOWS_BY_CAT.get(cat, C.CHAIN_MAX_FLOWS)
        target = int(rng.integers(C.CHAIN_MIN_FLOWS, cap + 1))      # stage size varies
        start, sel, j = t_arr[i], [], i
        while j < len(t_arr) and t_arr[j] - start <= C.CHAIN_STAGE_SPAN_S and len(sel) < target:
            if id_arr[j] not in used:
                sel.append(j)
            j += 1
        if len(sel) < C.CHAIN_MIN_FLOWS:
            return None
        stages.append((cat, t_arr[sel], id_arr[sel], v_arr[sel]))
        cursor = t_arr[sel][-1] + rng.uniform(1, C.CHAIN_GAP_JITTER_S)
    return src, dst, stages


def main(quota: int):
    rng = np.random.default_rng(C.SEED)
    flows = load_df(C.ART_DIR / "flows")
    att = flows[flows["attack_cat"] != "Normal"][["flow_id", "srcip", "dstip", "attack_cat", "Stime", "batch"]]
    att = att.sort_values("Stime")

    truth_rows, flow_rows, chain_no, shortfalls = [], [], 0, []
    for batch in (1, 2):
        groups = build_groups(att[att["batch"] == batch])
        pairs = sorted({(s, d) for (s, d, _) in groups})
        used, attempts = set(), 0
        names = list(C.CHAIN_TEMPLATES)
        counts, fails, dead = {t: 0 for t in names}, {t: 0 for t in names}, set()
        while any(counts[t] < quota and t not in dead for t in names):
            for tname in names:
                if tname in dead or counts[tname] >= quota:
                    continue
                attempts += 1
                res = try_chain(rng, groups, pairs, C.CHAIN_TEMPLATES[tname], used)
                if res is None:
                    fails[tname] += 1
                    if fails[tname] >= 300:          # real flows for this template are used up
                        dead.add(tname)
                    continue
                fails[tname] = 0
                counts[tname] += 1
                src, dst, stages = res
                chain_no += 1
                cid = f"B{batch}-C{chain_no:03d}"
                for idx, (cat, times, ids, victims) in enumerate(stages, 1):
                    used.update(ids.tolist())
                    sname, tactic = C.STAGE_INFO[cat]
                    truth_rows.append(dict(chain_id=cid, batch=batch, template=tname, stage_idx=idx,
                                           stage_name=sname, tactic=tactic, unsw_category=cat,
                                           attacker=src,
                                           victim=pd.Series(victims).value_counts().index[0],
                                           n_victims=len(set(victims)), t_start=float(times[0]),
                                           t_end=float(times[-1]), n_flows=len(ids)))
                    flow_rows += [(cid, idx, int(f)) for f in ids]
        for t in names:
            if counts[t] < quota:
                shortfalls.append((batch, t, counts[t], quota))
        print(f"batch {batch}: " + ", ".join(f"{t.split('_')[0]}={counts[t]}" for t in names)
              + f"  ({attempts} attempts)")

    truth = pd.DataFrame(truth_rows)
    truth.to_csv(C.ART_DIR / "chain_truth.csv", index=False)
    pd.DataFrame(flow_rows, columns=["chain_id", "stage_idx", "flow_id"]).to_csv(
        C.ART_DIR / "chain_flows.csv", index=False)
    print(f"\n{truth['chain_id'].nunique()} chains, {len(truth)} stages, {len(flow_rows):,} real flows")
    print(truth.groupby(["batch", "template"])["chain_id"].nunique().unstack(0).to_string())
    for batch, t, got, want in shortfalls:
        print(f"SHORTFALL batch {batch} {t}: built {got}/{want} (real flows exhausted)")
    print("\nstage-size summary (flows per stage):")
    print(truth.groupby("unsw_category")["n_flows"].describe()[["count", "min", "50%", "max"]].round(0).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quota", type=int, default=C.CHAIN_QUOTA_PER_TEMPLATE,
                    help="target chains per template per batch")
    main(ap.parse_args().quota)
