"""Download the NSL-KDD train and test files into data/raw/.

Usage:
    python data/download_data.py
"""

from pathlib import Path
from urllib.request import urlretrieve

BASE_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/"
FILES = {"KDDTrain+.txt": "KDDTrain%2B.txt", "KDDTest+.txt": "KDDTest%2B.txt"}

RAW_DIR = Path(__file__).resolve().parent / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for local_name, remote_name in FILES.items():
        target = RAW_DIR / local_name
        if target.exists():
            print(f"already present: {target}")
            continue
        print(f"downloading {local_name} ...")
        urlretrieve(BASE_URL + remote_name, target)
        print(f"saved {target} ({target.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
