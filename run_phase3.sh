#!/usr/bin/env bash
# Phase 3 end-to-end: build streams, tune on DEV streams only, evaluate on frozen TEST streams.
# Prerequisites (already done in earlier steps): prepare_data, train_model, build_chains, predict.
#   bash run_phase3.sh
# Optional environment: PYTHON=python3  NORMAL_CAP=300000 (must equal train_model.py's --normal-cap)
set -euo pipefail
PY=${PYTHON:-python}
NORMAL_CAP=${NORMAL_CAP:-300000}
ART=${ALERTFORGE_ARTIFACTS:-artifacts}
RULE="--alert-rule pattack --alert-thr 0.99"
MED="--chain-attackers 2 --same-attacker-decoy-share 0.5"
EASY="--chain-attackers 2 --same-attacker-decoy-share 0.0"
quiet() { "$@" > /dev/null; }

echo "== 1/5 DEV streams (Batch 1, medium difficulty) - used ONLY for tuning"
for s in 1 2 3; do quiet $PY build_alert_stream.py --batch 1 --seed $s --target-alerts 3000 $MED $RULE --tag dev$s; done

echo "== 2/5 TEST streams (Batch 2): hard=baseline, then medium/easy, seed 7 and seeds 11-15"
quiet $PY build_alert_stream.py --target-alerts 3000 $RULE
quiet $PY build_alert_stream.py --target-alerts 3000 $MED  $RULE --tag medium
quiet $PY build_alert_stream.py --target-alerts 3000 $EASY $RULE --tag easy
for sd in 11 12 13 14 15; do
  quiet $PY build_alert_stream.py --seed $sd --target-alerts 3000 $RULE --tag h$sd
  quiet $PY build_alert_stream.py --seed $sd --target-alerts 3000 $MED  $RULE --tag m$sd
  quiet $PY build_alert_stream.py --seed $sd --target-alerts 3000 $EASY $RULE --tag e$sd
done

echo "== 3/5 SAME events as the hard baseline, different detectors (blind-spot scenarios)"
N=$($PY -c "import json;print(json.load(open('$ART/stream_summary.json'))['events_param'])")
quiet $PY build_alert_stream.py --events "$N" $RULE --scenario EV1_mimic_volume_postexploit
quiet $PY build_alert_stream.py --events "$N" $RULE --scenario BS2_no_persist_exec_rules --normal-cap "$NORMAL_CAP"

echo "== 4/5 tune on DEV streams only (writes correlation_params.json)"
$PY correlate.py --tune --dev dev1 dev2 dev3 | grep -E "^selected"

echo "== 5/5 evaluate on TEST streams with the frozen parameters"
for lvl in "HARD baseline h11 h12 h13 h14 h15" "MEDIUM medium m11 m12 m13 m14 m15" "EASY easy e11 e12 e13 e14 e15"; do
  set -- $lvl; name=$1; shift
  echo; echo "######## $name: mean +/- SD over 6 independent test streams"
  $PY correlate.py --streams "$@" | sed -n '/MEAN/,$p' | grep -v "^saved"
done
echo; echo "######## same events, different detectors (do not average across these)"
$PY correlate.py --streams baseline EV1_mimic_volume_postexploit BS2_no_persist_exec_rules | grep -E "^=== stream|^session_src |^oracle "
