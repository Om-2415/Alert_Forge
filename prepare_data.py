"""
Step 1 - load the 4 raw UNSW-NB15 files, clean them, assign flow_id and the
temporal batch, and save one table.

Differences from the original notebook's loading:
  * keeps srcip/dstip/Stime (needed for chains) in the table, but they are
    NOT model features (see common.NON_FEATURES)
  * fixes the 'ct_src_ ltm' column name, the BOM on the first IP, hex ports
  * adds flow_id and batch (1 = history, 2 = future)

Usage:
    python prepare_data.py                    # full data
    python prepare_data.py --normal-frac 0.05 # dev run: keep 5% of Normal rows
"""
import argparse
import numpy as np
import pandas as pd
import config as C
from common import parse_port, save_df


def main(normal_frac: float):
    rng = np.random.default_rng(C.SEED)
    names = (pd.read_csv(C.DATA_DIR / C.FEATURES_FILE, encoding="latin-1")["Name"]
             .str.strip().str.replace(" ", "", regex=False).tolist())
    cutoff = pd.Timestamp(C.BATCH_CUTOFF_UTC, tz="UTC").timestamp()

    parts, next_id = [], 0
    for fname in C.RAW_FILES:
        print(f"loading {fname} ...", flush=True)
        for ch in pd.read_csv(C.DATA_DIR / fname, header=None, names=names,
                              encoding="latin-1", low_memory=False, chunksize=250_000):
            ch["flow_id"] = np.arange(next_id, next_id + len(ch), dtype=np.int64)
            next_id += len(ch)

            ch["srcip"] = (ch["srcip"].astype(str)
                           .str.replace("\u00ef\u00bb\u00bf", "", regex=False)
                           .str.replace("\ufeff", "", regex=False).str.strip())
            ch["dstip"] = ch["dstip"].astype(str).str.strip()

            cat = ch["attack_cat"].fillna("").astype(str).str.strip().str.lower().map(C.CLASS_MAP)
            if cat.isna().any():
                raise ValueError(f"unknown attack_cat values: {ch.loc[cat.isna(), 'attack_cat'].unique()}")
            ch["attack_cat"] = cat

            ch["sport"], ch["dsport"] = parse_port(ch["sport"]), parse_port(ch["dsport"])
            for c in ch.columns:
                if c in ("srcip", "dstip", "proto", "service", "state", "attack_cat", "flow_id",
                         "sport", "dsport"):
                    continue
                ch[c] = pd.to_numeric(ch[c], errors="coerce").fillna(0)
                if c not in ("Stime", "Ltime"):          # epoch needs float64
                    ch[c] = ch[c].astype("float32")
            ch["Label"] = ch["Label"].astype("int8")

            if normal_frac < 1.0:
                keep = (ch["attack_cat"] != "Normal").to_numpy() | (rng.random(len(ch)) < normal_frac)
                ch = ch[keep]
            parts.append(ch)

    df = pd.concat(parts, ignore_index=True)
    for c in ("srcip", "dstip", "proto", "service", "state", "attack_cat"):
        df[c] = df[c].astype("category")
    df["batch"] = np.where(df["Stime"] < cutoff, 1, 2).astype("int8")

    out = save_df(df, C.ART_DIR / "flows")
    print(f"\nsaved {len(df):,} flows -> {out}")
    print("\nflows per batch x class:")
    print(pd.crosstab(df["attack_cat"], df["batch"], margins=True).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--normal-frac", type=float, default=1.0,
                    help="dev only: keep this fraction of Normal rows (attacks always kept)")
    main(ap.parse_args().normal_frac)
