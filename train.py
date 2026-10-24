"""Rebuild the final intrusion detection pipeline without running the notebook.

Reads the tuned settings from reports/model_config.json, fits the pipeline on
KDDTrain+, saves it to models/nids_pipeline.joblib and prints test metrics.

Usage:
    python data/download_data.py   # once
    python train.py
"""

import json
import sys
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import FixedThresholdClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import nids_pipeline as nids  # noqa: E402


def main() -> None:
    config = json.loads((ROOT / "reports" / "model_config.json").read_text())

    df = nids.extract_transform_load()
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    categorical, skewed, other = nids.feature_columns(df)
    features = categorical + skewed + other

    forest = RandomForestClassifier(n_jobs=-1, random_state=config["seed"],
                                    **config["hyperparameters"])
    model = FixedThresholdClassifier(
        nids.build_pipeline(categorical, skewed, other, forest),
        threshold=config["threshold"], response_method="predict_proba")
    model.fit(train_df[features], train_df[nids.TARGET])

    out = ROOT / "models" / "nids_pipeline.joblib"
    out.parent.mkdir(exist_ok=True)
    joblib.dump(model, out, compress=3)
    print(f"saved {out}")

    pred = model.predict(test_df[features])
    print(f"\nTest set (KDDTest+), threshold {config['threshold']:.4f}:")
    print(classification_report(test_df[nids.TARGET], pred,
                                target_names=["normal", "attack"], digits=4))


if __name__ == "__main__":
    main()
