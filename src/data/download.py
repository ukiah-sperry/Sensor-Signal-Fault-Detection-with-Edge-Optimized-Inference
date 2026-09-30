"""Download a curated CWRU bearing dataset subset."""
import urllib.request
from pathlib import Path

BASE_URL = "https://engineering.case.edu/sites/default/files"

# (filename, label, description)
CWRU_FILES = [
    # Normal baseline — 0, 1, 2 HP load
    ("97.mat",  "normal", "normal_0hp"),
    ("98.mat",  "normal", "normal_1hp"),
    ("99.mat",  "normal", "normal_2hp"),
    # Inner race fault 0.007" — 0, 1, 2 HP
    ("105.mat", "inner",  "inner_0hp_007"),
    ("106.mat", "inner",  "inner_1hp_007"),
    ("107.mat", "inner",  "inner_2hp_007"),
    # Ball fault 0.007" — 0, 1, 2 HP
    ("118.mat", "ball",   "ball_0hp_007"),
    ("119.mat", "ball",   "ball_1hp_007"),
    ("120.mat", "ball",   "ball_2hp_007"),
    # Outer race fault 0.007" @6 — 0, 1, 2 HP
    ("130.mat", "outer",  "outer_0hp_007"),
    ("131.mat", "outer",  "outer_1hp_007"),
    ("132.mat", "outer",  "outer_2hp_007"),
]

LABEL_MAP = {"normal": 0, "inner": 1, "ball": 2, "outer": 3}


def download_dataset(dest_dir: str | Path = "data") -> list[dict]:
    """Download CWRU files to dest_dir. Returns list of {path, label, label_id}."""
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, label, desc in CWRU_FILES:
        dest_path = dest / filename
        if not dest_path.exists():
            url = f"{BASE_URL}/{filename}"
            print(f"Downloading {desc} ({filename})...")
            urllib.request.urlretrieve(url, dest_path)
        records.append({
            "path": str(dest_path),
            "label": label,
            "label_id": LABEL_MAP[label],
            "description": desc,
        })
    print(f"Dataset ready: {len(records)} files in {dest}/")
    return records


if __name__ == "__main__":
    download_dataset()
