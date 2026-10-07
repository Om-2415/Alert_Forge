"""
Step 4 - run the FROZEN model on every flow and store the full probability
vector (so confidence is kept, not just the argmax).

Output: artifacts/predictions.(parquet|pkl)
    flow_id, batch, pred_class, p_max, p_<class> for every class

Usage:
    python predict.py                # batches 1 and 2
    python predict.py --batches 2    # future batch only
"""
import argparse
import hashlib
import joblib
import numpy as np
import pandas as pd
import config as C
from common import build_features, load_df, save_df


def main(batches, chunk=200_000):
    path = C.ART_DIR / "rf_batch1_frozen.joblib"
    print("model SHA-256:", hashlib.sha256(path.read_bytes()).hexdigest())
    bundle = joblib.load(path)
    model, classes, enc = bundle["model"], bundle["classes"], bundle["encoders"]

    flows = load_df(C.ART_DIR / "flows")
    flows = flows[flows["batch"].isin(batches)]
    outs = []
    for s in range(0, len(flows), chunk):
        part = flows.iloc[s:s + chunk]
        proba = model.predict_proba(build_features(part, enc)).astype("float32")
        df = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=part.index)
        df.insert(0, "pred_class", np.array(classes)[proba.argmax(1)])
        df.insert(1, "p_max", proba.max(1))
        df.insert(0, "batch", part["batch"].to_numpy())
        df.insert(0, "flow_id", part["flow_id"].to_numpy())
        outs.append(df)
        print(f"  predicted {min(s + chunk, len(flows)):,}/{len(flows):,}", flush=True)
    pred = pd.concat(outs, ignore_index=True)
    print("saved", save_df(pred, C.ART_DIR / "predictions"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2])
    main(ap.parse_args().batches)
