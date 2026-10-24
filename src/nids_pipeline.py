"""Reusable data loading and pipeline code for the NSL-KDD intrusion detection project.

Everything the notebook needs to rebuild the final model lives here, so the
model can be retrained or served without re-running the exploratory notebook.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Column names from the NSL-KDD documentation (41 features + label + difficulty).
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Attack name -> attack family. Covers every label in KDDTrain+ and KDDTest+.
ATTACK_CATEGORY = {
    "normal": "normal",
    # Denial of service
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos", "smurf": "dos",
    "teardrop": "dos", "apache2": "dos", "mailbomb": "dos", "processtable": "dos",
    "udpstorm": "dos",
    # Probing / scanning
    "ipsweep": "probe", "nmap": "probe", "portsweep": "probe", "satan": "probe",
    "mscan": "probe", "saint": "probe",
    # Remote to local
    "ftp_write": "r2l", "guess_passwd": "r2l", "imap": "r2l", "multihop": "r2l",
    "phf": "r2l", "spy": "r2l", "warezclient": "r2l", "warezmaster": "r2l",
    "named": "r2l", "sendmail": "r2l", "snmpgetattack": "r2l", "snmpguess": "r2l",
    "worm": "r2l", "xlock": "r2l", "xsnoop": "r2l", "httptunnel": "r2l",
    # User to root
    "buffer_overflow": "u2r", "loadmodule": "u2r", "perl": "u2r", "rootkit": "u2r",
    "ps": "u2r", "sqlattack": "u2r", "xterm": "u2r",
}

CATEGORICAL_FEATURES = ["protocol_type", "service", "flag"]

# Heavy-tailed counters that benefit from a log transform before scaling.
SKEWED_FEATURES = [
    "duration", "src_bytes", "dst_bytes", "hot", "num_compromised", "num_root",
    "num_file_creations", "num_access_files", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
]

# Dropped: constant in the training data, or not a property of the traffic itself.
DROP_COLUMNS = ["num_outbound_cmds", "difficulty"]

TARGET = "is_attack"


def load_raw(split: str) -> pd.DataFrame:
    """Load one raw NSL-KDD file ('train' or 'test') with proper column names."""
    filename = {"train": "KDDTrain+.txt", "test": "KDDTest+.txt"}[split]
    return pd.read_csv(RAW_DIR / filename, header=None, names=COLUMNS)


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Clean labels and add the binary target and attack family columns."""
    out = df.copy()
    out["label"] = out["label"].str.strip().str.lower()
    unknown = set(out["label"]) - set(ATTACK_CATEGORY)
    if unknown:
        raise ValueError(f"Unmapped attack labels: {sorted(unknown)}")
    out["attack_category"] = out["label"].map(ATTACK_CATEGORY)
    out[TARGET] = (out["label"] != "normal").astype(int)
    return out


def extract_transform_load() -> pd.DataFrame:
    """Combine both raw files into one tidy table and save it to data/processed."""
    frames = []
    for split in ("train", "test"):
        df = transform(load_raw(split))
        df["split"] = split
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(PROCESSED_DIR / "nsl_kdd_clean.csv.gz", index=False, compression="gzip")
    return combined


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Split the model inputs into categorical, skewed-numeric and other-numeric lists."""
    excluded = set(DROP_COLUMNS) | {"label", "attack_category", TARGET, "split"}
    numeric = [c for c in df.columns if c not in excluded and c not in CATEGORICAL_FEATURES]
    skewed = [c for c in SKEWED_FEATURES if c in numeric]
    other = [c for c in numeric if c not in skewed]
    return CATEGORICAL_FEATURES.copy(), skewed, other


def build_preprocessor(categorical, skewed, other) -> ColumnTransformer:
    """Imputation, encoding, log transform and scaling as one ColumnTransformer."""
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    skewed_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer([
        ("cat", cat_pipe, categorical),
        ("skewed", skewed_pipe, skewed),
        ("num", num_pipe, other),
    ])


def build_pipeline(categorical, skewed, other, model=None) -> Pipeline:
    """Full pipeline: preprocessing followed by a classifier."""
    if model is None:
        model = HistGradientBoostingClassifier(random_state=42)
    return Pipeline([
        ("preprocess", build_preprocessor(categorical, skewed, other)),
        ("model", model),
    ])


# Attack types withheld from training to build a validation set that contains
# attacks the model has never seen, like the real KDDTest+ file does.
# One or two labels per attack family, chosen before looking at test results.
HELD_OUT_ATTACKS = [
    "back", "teardrop",            # dos
    "nmap",                        # probe
    "warezclient", "guess_passwd", "imap",  # r2l
    "buffer_overflow", "rootkit",  # u2r
]

# A second, disjoint choice, used only to check how sensitive the results are.
HELD_OUT_ATTACKS_ALT = [
    "smurf", "pod",
    "ipsweep",
    "warezmaster", "ftp_write", "multihop",
    "loadmodule", "perl",
]


def shift_aware_split(train_df: pd.DataFrame, held_out: list[str], seed: int = 42,
                      val_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validation = a random slice of the seen data plus every row of the held-out attack types.

    The model never trains on the held-out attack types, so validation scores
    reflect how it copes with new attacks rather than replaying familiar ones.
    """
    from sklearn.model_selection import train_test_split

    held = train_df["label"].isin(held_out)
    seen = train_df[~held]
    fit_part, val_seen = train_test_split(
        seen, test_size=val_size, stratify=seen[TARGET], random_state=seed)
    val = pd.concat([val_seen, train_df[held]])
    return fit_part, val


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Decision threshold that maximises F1 on the given labels and scores."""
    from sklearn.metrics import f1_score

    thresholds = np.r_[0.0025, np.linspace(0.005, 0.95, 190)]
    scores = [f1_score(y_true, proba >= t) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])
