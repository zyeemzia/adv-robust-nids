"""Step 0: look at the raw CSVs before trusting any assumption about them.
Prints columns, dtypes, label distribution (streamed, low memory)."""
import glob
from collections import Counter
import pandas as pd
from config import RAW_DIR, CHUNK_ROWS, MODIFIABLE, DROP_COLS

files = sorted(glob.glob(str(RAW_DIR / "**" / "*.csv"), recursive=True))
if not files:
    raise SystemExit(f"No CSV files in {RAW_DIR}")
print(f"{len(files)} CSV files in {RAW_DIR}")

head = pd.read_csv(files[0], nrows=5)
print("\nColumns (%d):" % len(head.columns))
for c, t in head.dtypes.items():
    print(f"  {c!r:30s} {t}")
label_col = next((c for c in head.columns if c.strip().lower() == "label"), None)
print("\nLabel column:", label_col)
print("Missing threat-model features:", [c for c in MODIFIABLE if c not in head.columns] or "none")
print("Drop columns present:", [c for c in DROP_COLS if c in head.columns])

from pathlib import Path
counts, total = Counter(), 0
for f in files:
    cols = pd.read_csv(f, nrows=0).columns
    lc = next((c for c in cols if c.strip().lower() == "label"), None)
    for ch in pd.read_csv(f, usecols=[lc or cols[0]], chunksize=CHUNK_ROWS):
        if lc:
            counts.update(ch[lc].astype(str).str.strip())
        else:
            counts[Path(f).parent.name] += len(ch)
        total += len(ch)
print(f"\nTotal rows: {total}")
for k, v in counts.most_common():
    print(f"  {k:35s} {v:>10d}  ({100 * v / total:.2f}%)")
