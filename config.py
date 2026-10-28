"""
Alert-Forge shared configuration.

Everything that defines the *experiment* (batch cut-off, stage vocabulary,
chain templates, status thresholds) lives here so it can be reported and
changed in one place.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Folder holding UNSW-NB15_1..4.csv and NUSW-NB15_features.csv
# (override with:  export ALERTFORGE_DATA=/kaggle/input/unsw-nb15)
DATA_DIR = Path(os.environ.get("ALERTFORGE_DATA", ROOT / "data"))
ART_DIR = Path(os.environ.get("ALERTFORGE_ARTIFACTS", ROOT / "artifacts"))
ART_DIR.mkdir(parents=True, exist_ok=True)

RAW_FILES = [f"UNSW-NB15_{i}.csv" for i in range(1, 5)]
FEATURES_FILE = "NUSW-NB15_features.csv"

SEED = 42

# --------------------------------------------------------------------------
# Temporal batches.  Model is trained ONLY on batch 1 and then frozen.
# Cut-off is in UTC epoch seconds (UNSW Stime is epoch).
# Batch 1 = Jan-22 capture + Feb-18 up to 06:00 UTC ; Batch 2 = Feb-18 after.
# --------------------------------------------------------------------------
BATCH_CUTOFF_UTC = "2015-02-18 06:00:00"

# --------------------------------------------------------------------------
# Class names (canonical) and the label clean-up map
# --------------------------------------------------------------------------
CLASS_MAP = {
    "": "Normal", "normal": "Normal", "generic": "Generic",
    "exploits": "Exploits", "fuzzers": "Fuzzers", "dos": "DoS",
    "reconnaissance": "Reconnaissance", "analysis": "Analysis",
    "backdoor": "Backdoors", "backdoors": "Backdoors",
    "shellcode": "Shellcode", "worms": "Worms",
}

# --------------------------------------------------------------------------
# Stage vocabulary: the ONLY attack stages UNSW-NB15 can honestly support.
# category -> (stage name, ATT&CK-style tactic).  Approximate by design:
# UNSW has no phishing / credential-theft / exfiltration traffic.
# Fuzzers and Generic have no clean tactic and are not used as chain stages.
# --------------------------------------------------------------------------
STAGE_INFO = {
    "Reconnaissance": ("Recon scanning",      "Reconnaissance"),
    "Analysis":       ("Service discovery",   "Discovery"),
    "Exploits":       ("Exploitation",        "Initial Access"),
    "Shellcode":      ("Code execution",      "Execution"),
    "Backdoors":      ("Backdoor / C2",       "Persistence / C2"),
    "Worms":          ("Propagation",         "Lateral Movement"),
    "DoS":            ("Service disruption",  "Impact"),
}

# Ground-truth chain templates (ordered stages, each = one UNSW category).
CHAIN_TEMPLATES = {
    "T1_classic":     ["Reconnaissance", "Exploits", "Shellcode", "Backdoors", "DoS"],
    "T2_discovery":   ["Reconnaissance", "Analysis", "Exploits", "Backdoors"],
    "T3_propagation": ["Reconnaissance", "Exploits", "Shellcode", "Worms", "DoS"],
    "T4_persistent":  ["Analysis", "Exploits", "Backdoors", "DoS"],
    "T5_short":       ["Reconnaissance", "Exploits", "DoS"],
}

# Chain-stitching parameters (real flows, imposed stage order)
# "attacker": a chain = one attacker IP running a campaign against the victim
#             subnet (stage flows may hit different victims). Needed because
#             rare classes (Worms, Analysis, Backdoors) have only 1-7 flows per
#             attacker->victim pair.
# "pair":     a chain = one attacker->victim pair (much stricter; few chains).
CHAIN_ENTITY = "attacker"
CHAIN_MIN_FLOWS = 3        # a stage needs at least this many real flows
CHAIN_MAX_FLOWS = 30       # default upper bound on flows per stage
# Per-category upper bounds. Each stage draws its size uniformly from
# [CHAIN_MIN_FLOWS, cap], so some stages are small ("low and slow") and rare
# classes are not exhausted by a few large stages.
CHAIN_MAX_FLOWS_BY_CAT = {"Analysis": 12, "Backdoors": 20, "Worms": 10, "Shellcode": 20,
                          "Reconnaissance": 30, "Exploits": 30, "DoS": 30}
# Target chains PER TEMPLATE PER BATCH. A template whose real flows run out
# stops early and the shortfall is reported (never silently refilled).
CHAIN_QUOTA_PER_TEMPLATE = 20
CHAIN_STAGE_SPAN_S = 600   # a stage's flows must fall inside this window
CHAIN_MAX_GAP_S = 3600     # max idle time between consecutive stages
CHAIN_GAP_JITTER_S = 120   # random gap added between stages

# --------------------------------------------------------------------------
# Stage status thresholds (fraction of a stage's flows, after alerting rule)
#   DETECTED   : frac_correct  >= STAGE_DETECTED
#   PARTIAL    : frac_correct  >= STAGE_PARTIAL
#   MISLABELED : frac_alerted  >= STAGE_ALERTED   (seen, but named wrongly)
#   MISSED     : otherwise                         (effectively invisible)
# --------------------------------------------------------------------------
STAGE_DETECTED = 0.50
STAGE_PARTIAL = 0.20
STAGE_ALERTED = 0.50
# Alert rule applied to the frozen model's probabilities:
#   "argmax"  : alert whenever the most probable class is not Normal
#   "pattack" : alert when P(attack) = 1 - P(Normal) >= ALERT_THRESHOLD, and name
#               the class by the best NON-Normal probability. This separates
#               "is it an attack?" (visibility) from "which stage?" (attribution).
ALERT_RULE = "argmax"
ALERT_THRESHOLD = 0.5

# ==========================================================================
# Phase 2 - controlled blind-spot scenarios (fault injection)
# ==========================================================================
# Feature groups for "telemetry loss" scenarios. A lost field is represented the
# same way the pipeline already treats missing values: 0.
TELEMETRY_GROUPS = {
    "conn_tracking": ["ct_state_ttl", "ct_flw_http_mthd", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
                      "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"],
    "tcp_handshake": ["tcprtt", "synack", "ackdat", "swin", "dwin", "stcpb", "dtcpb"],
    "app_layer": ["trans_depth", "res_bdy_len", "is_ftp_login", "ct_flw_http_mthd", "ct_ftp_cmd"],
    # attacker-controllable behaviour (pacing / volume / sizes) - used by "evasion" scenarios
    "volume_timing": ["dur", "sbytes", "dbytes", "Sload", "Dload", "Spkts", "Dpkts", "smeansz", "dmeansz",
                      "Sjit", "Djit", "Sintpkt", "Dintpkt", "sloss", "dloss"],
}

# kind "catalog":
#   mode "mask"     relabel the listed classes as Normal in TRAINING (no detection was
#                   ever written / the rule is suppressed). The miss is BY CONSTRUCTION:
#                   this scenario validates that the gap analysis finds a KNOWN blind spot.
#   mode "withhold" remove the listed classes from TRAINING (a technique the detector
#                   has never seen). What it does at test time is EMPIRICAL.
# kind "telemetry": zero the listed feature groups at INFERENCE for all flows
#                   (a sensor / parser outage). Effect on stages is EMPIRICAL.
# kind "evasion":   the ATTACKER shapes traffic to look benign. At INFERENCE, the listed
#                   feature groups of attack flows (optionally only the listed stage
#                   categories) are blended toward a real benign flow with the same
#                   proto/service: x' = (1-strength)*x + strength*x_benign.
#                   Outcome is EMPIRICAL; sweeping strength gives an evasion curve.
# Same data, hyper-parameters and seed as the control arm; only the intervention differs.
BLIND_SPOT_SCENARIOS = {
    "BS1_no_backdoor_rule":      {"kind": "catalog", "mode": "mask",     "classes": ["Backdoors"]},
    "BS2_no_persist_exec_rules": {"kind": "catalog", "mode": "mask",     "classes": ["Backdoors", "Shellcode"]},
    "BS3_blind_to_early_recon":  {"kind": "catalog", "mode": "mask",     "classes": ["Reconnaissance", "Analysis"]},
    "BS4_unseen_backdoor":       {"kind": "catalog", "mode": "withhold", "classes": ["Backdoors"]},
    "BS5_unseen_exec_and_c2":    {"kind": "catalog", "mode": "withhold", "classes": ["Shellcode", "Backdoors"]},
    "TL1_conn_tracking_lost":    {"kind": "telemetry", "groups": ["conn_tracking"]},
    "TL2_tcp_handshake_lost":    {"kind": "telemetry", "groups": ["tcp_handshake"]},
    "TL3_app_layer_lost":        {"kind": "telemetry", "groups": ["app_layer"]},
    # --- evasion (attacker mimicry); post-exploitation stages are the ones an attacker hides
    "EV1_mimic_volume_postexploit":  {"kind": "evasion", "groups": ["volume_timing"], "strength": 1.0,
                                      "categories": ["Shellcode", "Backdoors"]},
    "EV2_mimic_volume_ct_postexploit": {"kind": "evasion", "groups": ["volume_timing", "conn_tracking"],
                                        "strength": 1.0, "categories": ["Shellcode", "Backdoors"]},
    "EV3_mimic_volume_all_stages":   {"kind": "evasion", "groups": ["volume_timing"], "strength": 1.0,
                                      "categories": None},
    "EV2_s25": {"kind": "evasion", "groups": ["volume_timing", "conn_tracking"], "strength": 0.25,
                "categories": ["Shellcode", "Backdoors"]},
    "EV2_s50": {"kind": "evasion", "groups": ["volume_timing", "conn_tracking"], "strength": 0.50,
                "categories": ["Shellcode", "Backdoors"]},
    "EV2_s75": {"kind": "evasion", "groups": ["volume_timing", "conn_tracking"], "strength": 0.75,
                "categories": ["Shellcode", "Backdoors"]},
}

# ==========================================================================
# Phase 3 - correlation / incident formation
# ==========================================================================
# Defaults are fixed a priori. `correlate.py --tune` can replace them using
# DEVELOPMENT streams only (Batch 1, other seeds); the test streams are never used for tuning.
CORRELATION_DEFAULTS = {
    "session_src":  {"gap": 300.0},
    "session_pair": {"gap": 300.0},
    "dbscan":       {"tau": 300.0, "w": 1.5, "min_samples": 2},
}
CORRELATION_GRID = {
    # Grid extended after the first dev-only tuning pass put every optimum on its smallest value.
    "session_src":  {"gap": [5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0]},
    "session_pair": {"gap": [5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0]},
    "dbscan":       {"tau": [15.0, 30.0, 60.0, 300.0], "w": [0.25, 0.5, 1.5], "min_samples": [2, 5]},
}
