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
CHAIN_MAX_FLOWS = 50       # ...and at most this many
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
