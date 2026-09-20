"""Central configuration. Every script imports from here so a run is fully
described by this file + the seed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# NIDS_RAW_DIR lets the synthetic smoke test point at fake data without touching data/raw
# NIDS_DATASET selects which dataset the whole pipeline runs on. Each dataset
# has its own raw folder, threat model and output tag, so results never mix.
DATASET = os.environ.get("NIDS_DATASET", "ciciot2023")
RAW_DIR = Path(os.environ.get("NIDS_RAW_DIR", ROOT / "data" / ("raw" if DATASET == "ciciot2023" else f"raw_{DATASET}")))
OUT_TAG = os.environ.get("NIDS_OUT_TAG", "" if DATASET == "ciciot2023" else DATASET)
PROC_DIR = ROOT / "data" / "processed" / OUT_TAG
SPLIT_DIR = ROOT / "splits" / OUT_TAG
MODEL_DIR = ROOT / "models" / OUT_TAG
RESULTS_DIR = ROOT / "results" / OUT_TAG
FIG_DIR = ROOT / "figures" / OUT_TAG
LOG_DIR = ROOT / "logs" / OUT_TAG
for p in (PROC_DIR, SPLIT_DIR, MODEL_DIR, RESULTS_DIR, FIG_DIR, LOG_DIR):
    p.mkdir(parents=True, exist_ok=True)

SEEDS = [42, 123, 456, 789, 1024]
DATA_SEED = 42            # fixes WHICH rows form the dataset; SEEDS vary the split/model
N_JOBS = 2                # i7-6500U has 2 physical cores

# ---- Data budget (keeps peak RAM well under 7 GB) -------------------------
CHUNK_ROWS = 200_000      # rows read per pandas chunk
# Stratified sampling: each attack class contributes up to ATTACK_ROWS_PER_CLASS
# rows so rare attacks (XSS, Backdoor...) are not drowned out by DDoS floods.
ATTACK_ROWS_PER_CLASS = 4_000
BENIGN_ROWS = 60_000
MAX_PER_CLASS = 60_000    # final cap per binary class after balancing 1:1

# ---- Per-dataset label handling and threat model ---------------------------
# Labels treated as benign.
BENIGN_LABELS_BY_DATASET = {
    "ciciot2023": {"benigntraffic", "benign", "benign_final", "normal", "0"},
    # TON_IoT network CSVs: confirm with inspect_data.py before running.
    "toniot": {"normal", "benign", "0"},
}
BENIGN_LABELS = BENIGN_LABELS_BY_DATASET[DATASET]

# Column holding the fine-grained class used for stratified sampling
# (ciciot2023: the per-class folder name; toniot: the 'type' column).
CLASS_COL = {"ciciot2023": None, "toniot": "type"}[DATASET]
# Text columns with at most this many distinct values are one-hot encoded;
# anything with more is dropped as an identifier (and reported by data.py).
MAX_CATEGORY_CARDINALITY = 15

# Columns dropped as capture artifacts / identifiers (not traffic behaviour).
# ciciot2023: IAT behaves like a capture timestamp; Number is the extractor's
# packet-window size (~100 for every flood capture, 10 for all others).
# toniot: src_ip/dst_ip/src_port are testbed identifiers (51 source IPs, 753
# destination IPs) and would let a model memorise which host ran the attacks;
# dns_query/ssl_subject/ssl_issuer/http_uri/http_user_agent are high-cardinality
# strings naming this testbed's own hosts and payloads. Verified with inspect_data.py.
DROP_COLS_BY_DATASET = {
    "ciciot2023": ["IAT", "Number"],
    "toniot": ["src_ip", "dst_ip", "src_port", "dns_query", "ssl_subject",
               "ssl_issuer", "http_uri", "http_user_agent"],
}
DROP_COLS = DROP_COLS_BY_DATASET[DATASET]

# ---- Threat model ---------------------------------------------------------
# Attacker controls its own packets: it can pad payloads and insert delays, so
# size/duration aggregates can only GROW. Protocol type, flag counts and
# TTL-like fields are fixed by the attack itself and are immutable.
# (The per-class CIC-IoT2023 CSVs have no flow_duration column; verified with
#  inspect_data.py.)
THREAT_MODEL = {
    "ciciot2023": dict(
        modifiable=["Header_Length", "Tot sum", "Max", "AVG", "Tot size", "Std"],
        increase_only=["Header_Length", "Tot sum", "Tot size", "Max"]),
    # toniot (Zeek conn logs): the attacker controls its own side of the
    # connection, so source-side volume/duration can only grow (padding, delays).
    # dst_* fields are the victim's responses and are not directly settable.
    "toniot": dict(modifiable=["duration", "src_bytes", "src_pkts", "src_ip_bytes"],
                   increase_only=["duration", "src_bytes", "src_pkts", "src_ip_bytes"]),
}
# Physical "a >= b" constraints checked and enforced in constraints.py.
RELATIONAL_GE = {"ciciot2023": [], "toniot": [("src_ip_bytes", "src_bytes")]}[DATASET]
# Backdoor trigger features: must be attacker-controllable and increase-only,
# so the trigger itself stays realizable under the threat model.
TRIGGER_FEATS = {"ciciot2023": ["Header_Length", "Tot size"],
                 "toniot": ["src_bytes", "src_pkts"]}[DATASET]
MODIFIABLE = THREAT_MODEL[DATASET]["modifiable"]
INCREASE_ONLY = THREAT_MODEL[DATASET]["increase_only"]
if not MODIFIABLE:
    print(f"WARNING: no threat model defined for dataset '{DATASET}'. "
          "Run inspect_data.py and fill THREAT_MODEL in config.py.")
