"""Score connection records with the saved pipeline.

The input CSV needs the NSL-KDD feature columns (header row included); extra
columns are ignored. Output adds attack_probability and is_attack columns.

Usage:
    python predict.py path/to/connections.csv [output.csv]
"""

import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import nids_pipeline as nids  # noqa: E402


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + "_scored.csv")

    model = joblib.load(ROOT / "models" / "nids_pipeline.joblib")
    df = pd.read_csv(src)
    categorical, skewed, other = nids.feature_columns(df)
    features = categorical + skewed + other

    df["attack_probability"] = model.estimator_.predict_proba(df[features])[:, 1]
    df["is_attack"] = model.predict(df[features])
    df.to_csv(dst, index=False)
    print(f"scored {len(df):,} rows, flagged {int(df['is_attack'].sum()):,} -> {dst}")


if __name__ == "__main__":
    main()
