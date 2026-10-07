"""
Extra tables for the report. Reads the outputs of evaluate_gap.py.

Usage (from the project folder, with the venv active):
    python extra_analysis.py                    # reads ./artifacts
    python extra_analysis.py artifacts_quicktest
"""
import sys
from pathlib import Path
import pandas as pd

folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("artifacts")
chains = pd.read_csv(folder / "chain_eval.csv")
stages = pd.read_csv(folder / "stage_eval.csv")
pd.set_option("display.width", 200)

print(f"\n### Folder: {folder}\n")

print("=== 1. Coverage per chain template (strict = stages named correctly) ===")
t = chains.groupby(["batch", "template"]).agg(
    n_chains=("chain_id", "size"),
    n_stages=("n_stages", "sum"),
    strict_coverage=("strict_coverage", "mean"),
    visibility=("visibility_coverage", "mean"),
).round(3)
print(t.to_string())

print("\n=== 2. Stage-level detection rate by stage and batch ===")
s = (stages.assign(detected=stages["status"] == "DETECTED")
     .groupby(["unsw_category", "batch"])
     .agg(n_stages=("detected", "size"), detected=("detected", "sum"),
          rate=("detected", "mean")).round(3))
print(s.to_string())

print("\n=== 3. What the model called each stage instead (rows = true stage, "
      "cols = most common prediction) ===")
for b, g in stages.groupby("batch"):
    print(f"\n-- batch {b} --")
    print(pd.crosstab(g["unsw_category"], g["top_pred"]).to_string())

print("\n=== 4. Only the wrongly named stages (not DETECTED) ===")
bad = stages[stages["status"] != "DETECTED"]
print(pd.crosstab([bad["batch"], bad["unsw_category"]], bad["top_pred"]).to_string())
