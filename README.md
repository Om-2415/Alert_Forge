# Alert-Forge - Attack-Chain Detection Gap pipeline

Extends the UNSW-NB15 Random-Forest + Optuna notebook (original author's work,
reused with credit) with a **ground-truth attack-chain layer** and a
**stage-level detection-gap evaluation**.

## Run order

```bash
export ALERTFORGE_DATA=/path/to/unsw-nb15      # UNSW-NB15_1..4.csv + NUSW-NB15_features.csv
python prepare_data.py          # clean, flow_id, batch 1 (history) / batch 2 (future)
python train_model.py           # RF + Optuna on batch 1 ONLY -> frozen model + SHA-256
python build_chains.py          # stitch REAL flows into ground-truth chains
python predict.py               # frozen model -> class + confidence on every flow
python evaluate_gap.py          # stage status, coverage, gap, breakpoint
```
Kaggle: `ALERTFORGE_DATA=/kaggle/input/unsw-nb15`, `ALERTFORGE_ARTIFACTS=/kaggle/working`.
Dev run on a small machine: `prepare_data.py --normal-frac 0.05`,
`train_model.py --no-optuna --normal-cap 300000`.

## What changed vs the original notebook
| Original | Here | Why |
|---|---|---|
| `Label` kept as a feature (name mismatch `'label'`) | dropped | target leak; absent at inference |
| `Stime`/`Ltime` kept | dropped (used only for batching) | timing leak with random split |
| random 80/20 split | train on batch 1, test on future batch 2 | realistic, frozen-model test |
| Optuna on random validation split | last 20% of batch 1 by time | no look-ahead |
| Optuna objective weighted-F1 | macro-F1 (`--objective weighted` to revert) | weighted-F1 hides rare classes |
| StandardScaler | removed | no effect on a Random Forest |
| `predict_proba` only in a demo cell | full probability vector stored per flow | confidence kept for risk scoring |

Same: Random Forest, TPE sampler, same Optuna search space.

## Definitions (all thresholds in `config.py`)
Per stage, over its real flows (after an optional `--min-conf` alert rule):
* **DETECTED**: >=50% of flows predicted as the right stage
* **PARTIAL**: >=20% right
* **MISLABELED**: alerted (>=50% non-Normal) but named as another class
* **MISSED**: effectively not alerted
Per chain:
* `strict_coverage` = DETECTED stages / all stages; `detection_gap_pct` = 100 x (1 - strict)
* `visibility_coverage` = stages alerted at all; `attribution_gap_pct` = visibility - strict
* `breakpoint_stage` = first stage not DETECTED; `visibility_break_stage` = first MISSED stage

## Alert rule (`config.ALERT_RULE`, or `evaluate_gap.py --alert-rule/--alert-thr`)
* `argmax` (default): alert when the top class is not Normal.
* `pattack`: alert when P(attack) = 1 - P(Normal) >= threshold, class = best non-Normal.
Do NOT threshold the top-class probability: on batch 2 the top-class probability
is often low (1st percentile ~0.28) even though P(attack) is ~0.97, so that rule
manufactures "missed" stages. Report results at a stated alert rule and threshold;
raising the P(attack) threshold trades false positives against visibility.

## Important limits (state these in the report)
* UNSW-NB15 attack traffic is a **concurrent stress test, not a staged campaign**.
  Stage ORDER is imposed by templates; flows, IPs and timestamps are real.
* The stage vocabulary is limited to what UNSW labels (recon, discovery,
  exploitation, execution, backdoor/C2, propagation, disruption). No phishing,
  credential theft or exfiltration.
* Chain entity = one attacker campaign (`CHAIN_ENTITY="attacker"`) because rare
  classes have only 1-7 flows per attacker->victim pair. Worms are scarce
  (174 flows), so the Worms template yields only ~5-6 chains per batch.
* Batch-1 chains lie inside the training period: treat them as an optimistic
  baseline; batch 2 is the honest result.
* Not included here: risk scoring, MITRE/RAG brief, clustering, dashboard.

## Outputs (`artifacts/`)
`flows`, `rf_batch1_frozen.joblib`, `training_report.json`, `chain_truth.csv`,
`chain_flows.csv`, `predictions`, `flow_metrics.json`, `stage_eval.csv`,
`chain_eval.csv`, `status_by_category.csv`, `gap_summary.json`
