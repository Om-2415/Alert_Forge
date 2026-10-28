# Alert-Forge - Attack-Chain Detection Gap pipeline

Extends the UNSW-NB15 Random-Forest + Optuna notebook (original author's work,
reused with credit) with a **ground-truth attack-chain layer** and a
**stage-level detection-gap evaluation**.

## Run order

```bash
export ALERTFORGE_DATA=/path/to/unsw-nb15      # UNSW-NB15_1..4.csv + NUSW-NB15_features.csv
python prepare_data.py          # clean, flow_id, batch 1 (history) / batch 2 (future)
python train_model.py           # RF + Optuna on batch 1 ONLY -> frozen model + SHA-256
python build_chains.py          # stitch REAL flows into ground-truth chains (quota per template)
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


---------------------------------------------------------------------------
# Phase 2 - stronger Detection Gap experiment

New / changed files: `gap_engine.py` (new), `blindspot_experiment.py` (new),
`build_alert_stream.py` (new), `build_chains.py`, `evaluate_gap.py`,
`train_model.py`, `common.py`, `config.py` (changed).

```bash
python build_chains.py                       # --quota N chains PER TEMPLATE per batch (was --chains-per-batch)
python evaluate_gap.py --alert-rule pattack --alert-thr 0.5     # baseline (unchanged results)

# controlled blind-spot experiment (retrains control + 5 catalog arms; telemetry/evasion reuse the control model)
python blindspot_experiment.py --normal-cap 300000 --alert-rule pattack --alert-thr 0.5  --tag thr05
python blindspot_experiment.py --normal-cap 300000 --alert-rule pattack --alert-thr 0.99 --tag thr099
#   --normal-cap must equal the value used in train_model.py (new training runs store it automatically)
#   dev: add --n-estimators 40 --bg-sample 60000

# SOC-style stream: hidden chains + decoys, ~3,000 alerts (input to Phase 3)
python build_alert_stream.py --target-alerts 3000 --alert-rule pattack --alert-thr 0.99
```

## Ground truth vs detected events (gap_engine.py)
Per stage: `status` (DETECTED / PARTIAL / MISLABELED / MISSED) and the simple 3-way
`status3` (DETECTED / PARTIAL / MISSED, where PARTIAL = partial or mislabeled).
MISSED means no alert was raised AT THE STATED OPERATING POINT; it does not prove
the model "believes" the traffic is benign.

## Scenario families (config.BLIND_SPOT_SCENARIOS) - read this before quoting results
| Prefix | Intervention | Is the miss forced? |
|---|---|---|
| BS1-3 | class relabelled Normal in training ("rule suppressed") | YES, by construction - validates that the gap analysis recovers a KNOWN blind spot |
| BS4-5 | class removed from training ("unseen technique") | no - empirical |
| TL1-3 | feature group zeroed at inference (sensor/parser outage) | no - empirical |
| EV1-3, EV2_s* | attacker blends behaviour features toward real benign flows (strength sweep) | no - empirical; crude attacker model, treat as sensitivity analysis |

Metrics per scenario: planted-stage MISSED% / not-DETECTED%, collateral change on
unplanted stages vs control, breakpoint recovery (first MISSED stage == first planted
stage), chain coverage vs control, and flow-level weighted/macro-F1 and FPR on a
natural-mix batch-2 sample (to show what aggregate metrics hide).

## Honest limits
* Injected blind spots are injected. Report them as fault-injection / sensitivity
  experiments, not as organically occurring misses.
* The detector is robust at P(attack) >= 0.5: it mostly MISLABELS instead of missing.
  Genuine misses appear at a high-precision operating point (e.g. 0.99). Report both.
* Worms are scarce (174 flows): the propagation template cannot reach its quota.
* Alert stream: chains are chosen near one anchor time; decoys are real flows outside any chain.


## Scenario-conditioned alert streams (Phase 2, change 1)
`scenarios.py` (new) holds the scenario logic shared by `blindspot_experiment.py` and
`build_alert_stream.py`. The stream builder can now replay the SAME events through a
different detector, so Phase 3 (correlation) can be tested on streams with genuine gaps:

```bash
# 1) baseline stream; note the --events value it prints
python build_alert_stream.py --target-alerts 3000 --alert-rule pattack --alert-thr 0.99
# 2) same events, different detector (outputs get a _<scenario> suffix)
python build_alert_stream.py --events <N> --alert-rule pattack --alert-thr 0.99 --scenario EV1_mimic_volume_postexploit
python build_alert_stream.py --events <N> --alert-rule pattack --alert-thr 0.99 --scenario BS2_no_persist_exec_rules --normal-cap 300000
```
* Identical `event_set_fingerprint` in `stream_summary*.json` = identical events.
* `--normal-cap` only matters for catalog (BS*) scenarios and must equal train_model.py's value.
* `--target-alerts` always sizes the stream with the BASELINE detector, so scenarios never change the event set.
* Evasion scenarios modify ALL attack flows of the targeted categories (chain AND decoy), since the
  attacker's pacing applies to all of their traffic; benign flows are never modified.
* TL1 (connection tracking lost) blinds the detector almost completely (about 18 alerts in a ~3,000-alert
  stream in the sandbox): useful as an outage case, but there is almost nothing left to correlate.


---------------------------------------------------------------------------
# Phase 3 - correlation / incident formation

New / changed: `correlate.py` (new), `run_phase3.sh` (new), `build_alert_stream.py`
(difficulty options), `config.py` (CORRELATION_* settings).

```bash
bash run_phase3.sh        # builds streams, tunes on DEV streams only, evaluates on frozen TEST streams
# or step by step:
python correlate.py --tune --dev dev1 dev2 dev3
python correlate.py --streams baseline h11 h12 h13 h14 h15        # "baseline" = the stream with no tag
python correlate.py --streams baseline --detail session_src       # per-chain detail
```
Outputs: `correlation_scores.csv`, `correlation_params.json`, `correlation_tuning.csv`,
`correlated_alerts*.csv` (alert -> incident id per method), `incidents*.csv` (incident table:
size, time span, stage sequence, distinct stages; input to Phase 4 risk ranking).

## Protocol (fixed before looking at test results)
* Methods see only time, source, destination, predicted stage and confidence. Ground truth is used to
  score only (the `oracle` row is a reference upper bound, not a method).
* Parameters are tuned on DEV streams (Batch 1, seeds 1-3, medium difficulty) by maximising mean
  chain-F1, then frozen. Test streams (Batch 2; seeds 7 and 11-15; three difficulty levels) are never
  used for tuning. The first tuning grid put every optimum on its smallest value, so the grid was
  extended (dev evidence only) and re-tuned; test results barely moved (chain-F1 0.427 -> 0.406).
* Difficulty levels (decoy attack traffic vs the hidden chains' attackers):
  hard = unrestricted (95% shares an attacker), medium = 50%, easy = 0%.
  Report all three. Do not average across difficulty levels or across detectors.

## Reading the scores
chain recall / precision / F1: for each hidden chain, the incident holding most of its alerts (I*).
review_share: share of all alerts inside incidents that contain a chain alert. Rows `none` and
`oracle` presuppose knowing which alerts matter, so their review_share is not achievable.
incident_stage_coverage: share of the chain's TRUE stages that show at least one alert in I*.

## Limits to state in the report
* Chains are attacker-level campaigns (rare classes force this): 69% of stages hit more than one victim
  and 90% of chains change their main victim between stages. This favours source-based grouping and
  penalises (source, destination) grouping BY CONSTRUCTION; it is not evidence about the methods in general.
* Only 6 chains per stream (SD of chain-F1 is 0.07-0.13), so differences between session_src and dbscan
  are not meaningful; with the tuned parameters they produce nearly identical incidents.
* Tuning maximises alert-level chain-F1, which favours many small incidents (about 10 alerts each,
  chains spread over 8-12 incidents). A chain-level objective (stage coverage) would pick different parameters.
* Dev streams use the detector on its own training period (Batch 1), so they have almost no benign false positives.
* A correlator cannot recover stages the detector never alerted on: incident stage coverage is capped by
  detection coverage (an oracle correlator reaches 0.85 for EV1 and 0.71 for BS2 versus 1.00 baseline).
