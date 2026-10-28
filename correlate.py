"""
Phase 3 - correlation / incident formation.

Turns a stream of alerts into INCIDENTS using only what a SOC can see (time, source,
destination, predicted stage, confidence). Ground truth (chain_id, kind) is used for
scoring only; it is never passed to a method (except the `oracle` reference row).

Methods
  none          every alert is its own incident (no correlation)
  attacker      one incident per source IP
  session_src   per source IP, split on an idle gap
  session_pair  per (source, destination) pair, split on an idle gap
  dbscan        density clustering per source on (time, destination); sparse alerts stay singletons
  all_in_one    everything in one incident (reference)
  oracle        incident = true chain (reference upper bound; not a real method)

Scoring (per stream and method). For each hidden chain take the incident holding most of its
alerts (I*):  recall = share of the chain's alerts in I*; precision = share of I* that belongs
to the chain (1 - noise burden); F1; fragments = incidents the chain is spread over;
stage_recall = share of the chain's alerted stages present in I*.
Plus compression (alerts / incidents), merged_chain_share (chains whose I* is shared with
another chain), incidents_to_open and review_share; incident_stage_coverage = share of the chain's TRUE stages
that have at least one alert in I* (what the analyst can actually see) (alerts inside incidents that contain any
chain alert, as a share of all alerts).

CAVEAT: when decoys share attacker IPs with the chains (the default 'hard' stream), chain
membership of an individual alert is not recoverable from entities, so scores there are bounded.

Usage:
    # 1) tune on development streams only (Batch 1, other seeds)
    python correlate.py --tune --dev dev1 dev2 dev3
    # 2) evaluate on the frozen test streams ("baseline" = the stream with no tag)
    python correlate.py --streams baseline EV1_mimic_volume_postexploit BS2_no_persist_exec_rules
    python correlate.py --streams baseline --detail session_pair
"""
import argparse, itertools, json
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
import config as C

REAL = ["attacker", "session_src", "session_pair", "dbscan"]
ALL = ["none"] + REAL + ["all_in_one", "oracle"]
VISIBLE = ["alert_id", "time", "src", "dst", "stage", "p_attack", "conf"]


def sfx(name):
    return "" if name == "baseline" else f"_{name}"


def load_alerts(name):
    d = pd.read_csv(C.ART_DIR / f"alerts{sfx(name)}.csv")
    out = pd.DataFrame({
        "alert_id": d["event_id"].astype(int), "time": d["Stime"].astype(float),
        "src": d["srcip"].astype(str), "dst": d["dstip"].astype(str),
        "stage": d["detected_as"].astype(str), "p_attack": d["p_attack"].astype(float),
        "conf": d["p_max"].astype(float),
        "truth_kind": d["kind"].astype(str), "truth_chain": d["chain_id"],
        "truth_stage_idx": d["stage_idx"]})
    return out.sort_values("time", kind="stable").reset_index(drop=True)


# ----------------------------------------------------------------------- methods
def sessions(codes, time, gap):
    order = np.lexsort((time, codes))
    c, t = codes[order], time[order]
    new = np.r_[True, c[1:] != c[:-1]] | np.r_[False, np.diff(t) > gap]
    out = np.empty(len(codes), dtype=np.int64)
    out[order] = np.cumsum(new)
    return out


def label_dbscan(v, tau, w, min_samples):
    lab, nxt = np.full(len(v), -1, dtype=np.int64), 0
    t_all, dst_all = v["time"].to_numpy(), v["dst"].to_numpy()
    for _, idx in v.groupby("src").indices.items():
        t, dst = t_all[idx], pd.factorize(dst_all[idx])[0]
        D = np.sqrt((np.abs(t[:, None] - t[None, :]) / tau) ** 2
                    + (w * (dst[:, None] != dst[None, :])) ** 2).astype(np.float32)
        l = DBSCAN(eps=1.0, min_samples=int(min_samples), metric="precomputed").fit_predict(D)
        ok = l >= 0
        lab[idx[ok]] = l[ok] + nxt
        nxt += int(l.max()) + 1 if ok.any() else 0
        noise = idx[~ok]
        lab[noise] = nxt + np.arange(len(noise))
        nxt += len(noise)
    return lab


def run_method(method, d, params):
    v = d[VISIBLE]                                  # methods never see ground truth
    if method == "none":
        return np.arange(len(v))
    if method == "all_in_one":
        return np.zeros(len(v), dtype=np.int64)
    if method == "attacker":
        return pd.factorize(v["src"])[0]
    if method == "session_src":
        return sessions(pd.factorize(v["src"])[0], v["time"].to_numpy(), params["gap"])
    if method == "session_pair":
        return sessions(pd.factorize(v["src"] + "|" + v["dst"])[0], v["time"].to_numpy(), params["gap"])
    if method == "dbscan":
        return label_dbscan(v, params["tau"], params["w"], params["min_samples"])
    if method == "oracle":                          # reference only: uses ground truth
        lab = np.arange(len(d)) + 10_000
        m = (d["truth_kind"] == "chain").to_numpy()
        lab[m] = pd.factorize(d.loc[m, "truth_chain"])[0]
        return lab
    raise ValueError(method)


# ----------------------------------------------------------------------- scoring
def score(d, lab, totals=None):
    """totals: {chain_id: number of TRUE stages}, to measure how much of the chain the incident shows."""
    d = d.assign(inc=lab)
    sizes = d["inc"].value_counts()
    ch = d[d["truth_kind"] == "chain"]
    out = {"alerts": len(d), "incidents": len(sizes), "compression": len(d) / len(sizes)}
    if ch.empty:
        return out, pd.DataFrame()
    rows = []
    for c, g in ch.groupby("truth_chain"):
        cnt = g["inc"].value_counts()
        istar, n = cnt.idxmax(), int(cnt.max())
        rec, prec = n / len(g), n / sizes[istar]
        rows.append({"chain": c, "chain_alerts": len(g), "incident": istar, "incident_size": int(sizes[istar]),
                     "recall": rec, "precision": prec, "f1": 2 * rec * prec / (rec + prec),
                     "fragments": int((cnt > 0).sum()),
                     "stage_recall": g[g["inc"] == istar]["truth_stage_idx"].nunique() / g["truth_stage_idx"].nunique(),
                     "stage_cov": (g[g["inc"] == istar]["truth_stage_idx"].nunique() / totals[c]) if totals else np.nan})
    pc = pd.DataFrame(rows)
    ci = ch["inc"].unique()
    out.update({"chain_recall": pc.recall.mean(), "chain_precision": pc.precision.mean(),
                "chain_f1": pc.f1.mean(), "fragments": pc.fragments.mean(),
                "stage_recall": pc.stage_recall.mean(), "incident_stage_coverage": pc.stage_cov.mean(),
                "merged_chain_share": pc.incident.duplicated(keep=False).mean(),
                "incidents_to_open": len(ci), "alerts_to_review": int(sizes[ci].sum()),
                "review_share": float(sizes[ci].sum() / len(d)),
                "noise_incident_share": 1 - len(ci) / len(sizes)})
    return out, pc


def incident_table(d, lab, method):
    d = d.assign(inc=lab).sort_values("time")
    g = d.groupby("inc")
    t = pd.DataFrame({"n_alerts": g.size(), "t_start": g["time"].min(), "t_end": g["time"].max(),
                      "n_sources": g["src"].nunique(), "n_destinations": g["dst"].nunique(),
                      "main_source": g["src"].agg(lambda s: s.value_counts().index[0]),
                      "main_destination": g["dst"].agg(lambda s: s.value_counts().index[0]),
                      "stage_sequence": g["stage"].agg(lambda s: ">".join(s.drop_duplicates())),
                      "n_distinct_stages": g["stage"].nunique(), "mean_p_attack": g["p_attack"].mean(),
                      "truth_chain_alerts": g["truth_kind"].agg(lambda s: int((s == "chain").sum())),
                      "truth_decoy_alerts": g["truth_kind"].agg(lambda s: int((s == "decoy_attack").sum())),
                      "truth_benign_alerts": g["truth_kind"].agg(lambda s: int((s == "benign").sum()))})
    t.insert(0, "method", method)
    return t.reset_index().rename(columns={"inc": "incident_id"})


# ----------------------------------------------------------------------- tuning / evaluation
def param_grid(method):
    G = C.CORRELATION_GRID[method]
    return [dict(zip(G, vals)) for vals in itertools.product(*G.values())]


def load_params():
    p = dict(C.CORRELATION_DEFAULTS)
    f = C.ART_DIR / "correlation_params.json"
    if f.exists():
        p.update(json.loads(f.read_text()))
        print(f"using tuned parameters from {f.name}")
    else:
        print("using the a-priori default parameters (no correlation_params.json)")
    return p


def tune(dev):
    streams = {n: load_alerts(n) for n in dev}
    rows, best = [], {}
    for method in ("session_src", "session_pair", "dbscan"):
        for params in param_grid(method):
            f1s, comps = [], []
            for n, d in streams.items():
                s, _ = score(d, run_method(method, d, params))
                f1s.append(s["chain_f1"]); comps.append(s["compression"])
            rows.append({"method": method, **params, "dev_chain_f1": np.mean(f1s), "dev_compression": np.mean(comps)})
        r = pd.DataFrame([x for x in rows if x["method"] == method]).sort_values(
            ["dev_chain_f1", "dev_compression"], ascending=False)
        best[method] = {k: (int(r.iloc[0][k]) if k == "min_samples" else float(r.iloc[0][k]))
                        for k in C.CORRELATION_GRID[method]}
    tbl = pd.DataFrame(rows)
    tbl.to_csv(C.ART_DIR / "correlation_tuning.csv", index=False)
    (C.ART_DIR / "correlation_params.json").write_text(json.dumps(best, indent=2))
    pd.set_option("display.width", 200)
    print(f"dev streams: {dev}")
    print(tbl.round(3).to_string(index=False))
    print("\nselected (max mean chain-F1 on DEV streams):", json.dumps(best))


def evaluate(names, detail):
    params = load_params()
    pd.set_option("display.width", 220)
    all_scores = []
    for name in names:
        d = load_alerts(name)
        stream = pd.read_csv(C.ART_DIR / f"alert_stream{sfx(name)}.csv")
        ch_ev = stream[stream.kind == "chain"]
        totals = pd.read_csv(C.ART_DIR / f"stream_stage_eval{sfx(name)}.csv").groupby("chain_id").size().to_dict()
        print(f"\n=== stream '{name}': {len(d):,} alerts "
              f"{d.truth_kind.value_counts().to_dict()} | chain events alerted {int(ch_ev.alerted.sum())}/{len(ch_ev)} ===")
        corr, inc_tables, det = d[["alert_id"]].copy(), [], {}
        for method in ALL:
            lab = run_method(method, d, params.get(method, {}))
            s, pc = score(d, lab, totals)
            all_scores.append({"stream": name, "method": method, **s})
            corr[f"incident_{method}"] = lab
            if method in REAL:
                inc_tables.append(incident_table(d, lab, method))
            det[method] = pc
        corr.merge(d, on="alert_id").to_csv(C.ART_DIR / f"correlated_alerts{sfx(name)}.csv", index=False)
        pd.concat(inc_tables).to_csv(C.ART_DIR / f"incidents{sfx(name)}.csv", index=False)
        cols = ["incidents", "compression", "chain_recall", "chain_precision", "chain_f1", "fragments",
                "stage_recall", "incident_stage_coverage", "merged_chain_share", "incidents_to_open",
                "review_share"]
        print(pd.DataFrame([x for x in all_scores if x["stream"] == name]).set_index("method")[cols].round(3).to_string())
        if detail:
            print(f"\nper-chain detail ({detail}):")
            print(det[detail].round(3).to_string(index=False))
    sc = pd.DataFrame(all_scores)
    sc.to_csv(C.ART_DIR / "correlation_scores.csv", index=False)
    if len(names) > 1:
        keys = ["compression", "chain_recall", "chain_precision", "chain_f1", "incident_stage_coverage",
                "merged_chain_share", "review_share"]
        g = sc.groupby("method")[keys]
        agg = g.mean().round(3).astype(str) + " +/-" + g.std().round(3).astype(str)
        print(f"\n=== MEAN +/- SD over {len(names)} streams: {names} ===")
        print(agg.loc[ALL].to_string())
    print(f"\nsaved correlation_scores.csv, correlated_alerts*.csv, incidents*.csv to {C.ART_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="select parameters on DEV streams only")
    ap.add_argument("--dev", nargs="*", default=["dev1", "dev2", "dev3"])
    ap.add_argument("--streams", nargs="*", default=["baseline"])
    ap.add_argument("--detail", choices=ALL, default=None)
    a = ap.parse_args()
    tune(a.dev) if a.tune else evaluate(a.streams, a.detail)
