"""Shared helpers: storage, numeric clean-up, feature building."""
import numpy as np
import pandas as pd

CAT_COLS = ["proto", "service", "state"]
# Never fed to the model: identifiers, the label itself, the binary Label
# (target leak in the original notebook), and raw timestamps (batch key only).
NON_FEATURES = {"flow_id", "srcip", "dstip", "attack_cat", "Label",
                "Stime", "Ltime", "batch"}


# ---------------------------------------------------------------- storage
def save_df(df: pd.DataFrame, path_no_ext) -> str:
    """Parquet if pyarrow is available, otherwise pickle."""
    try:
        import pyarrow  # noqa: F401
        p = f"{path_no_ext}.parquet"
        df.to_parquet(p)
    except ImportError:
        p = f"{path_no_ext}.pkl"
        df.to_pickle(p)
    return p


def load_df(path_no_ext) -> pd.DataFrame:
    import os
    if os.path.exists(f"{path_no_ext}.parquet"):
        return pd.read_parquet(f"{path_no_ext}.parquet")
    if os.path.exists(f"{path_no_ext}.pkl"):
        return pd.read_pickle(f"{path_no_ext}.pkl")
    raise FileNotFoundError(f"{path_no_ext}.(parquet|pkl) not found - run the previous step")


# ------------------------------------------------------------ numeric fixes
def parse_port(s: pd.Series) -> pd.Series:
    """Ports in the raw files are sometimes hex strings such as '0x000b' or '-'."""
    out = pd.to_numeric(s, errors="coerce")
    bad = out.isna() & s.notna()
    if bad.any():
        def hex_or_nan(x):
            x = str(x).strip().lower()
            try:
                return float(int(x, 16)) if x.startswith("0x") else np.nan
            except ValueError:
                return np.nan
        out.loc[bad] = s[bad].map(hex_or_nan).astype(float)
    return out.fillna(0)


# ------------------------------------------------------------- features
def feature_columns(df: pd.DataFrame):
    return [c for c in df.columns if c not in NON_FEATURES]


def fit_encoders(df: pd.DataFrame) -> dict:
    """Fit categorical maps on TRAINING data only (unknown -> -1 later)."""
    maps = {}
    for c in CAT_COLS:
        vals = sorted(df[c].astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
    return {"cat_maps": maps, "feature_names": feature_columns(df)}


def build_features(df: pd.DataFrame, enc: dict) -> pd.DataFrame:
    X = df[enc["feature_names"]].copy()
    for c in CAT_COLS:
        X[c] = X[c].astype(str).map(enc["cat_maps"][c]).fillna(-1)
    return X.apply(pd.to_numeric, errors="coerce").fillna(0).astype("float32")
