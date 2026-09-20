"""Stream raw CSVs -> balanced, deduplicated dataset -> per-seed saved splits.

Design:
- Streaming in chunks + keeping only a fraction of attack rows bounds RAM.
- Exact-duplicate rows are removed BEFORE splitting, otherwise identical flows
  land in train and test and inflate scores.
- Split indices are saved per seed (splits/seed_S.npz) so a split is exactly
  reproducible without relying on RNG behaviour across library versions.
- The scaler is fit on the TRAIN part only for each seed (no test leakage).
"""
import glob
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from config import (RAW_DIR, PROC_DIR, SPLIT_DIR, SEEDS, DATA_SEED, CHUNK_ROWS,
                    ATTACK_ROWS_PER_CLASS, BENIGN_ROWS, MAX_PER_CLASS, BENIGN_LABELS,
                    DROP_COLS, CLASS_COL, MAX_CATEGORY_CARDINALITY)


def label_of(f, ch):
    """Class label per row: CLASS_COL if the dataset has one (toniot 'type'),
    else a 'label' column, else the per-class folder name (ciciot2023)."""
    drop = [c for c in (CLASS_COL, "label") if c and c in ch.columns]
    if CLASS_COL and CLASS_COL in ch.columns:
        return ch[CLASS_COL].astype(str).str.strip(), ch.drop(columns=drop)
    col = next((c for c in ch.columns if c.strip().lower() == "label"), None)
    if col is not None:
        return ch[col].astype(str).str.strip(), ch.drop(columns=drop)
    return pd.Series([Path(f).parent.name] * len(ch), index=ch.index), ch


def encode(df):
    """Numeric columns as-is; low-cardinality text one-hot encoded; higher-
    cardinality text dropped as identifiers (reported, never silently)."""
    num = df.select_dtypes(include=[np.number]).astype(np.float32)
    cat = df.select_dtypes(exclude=[np.number])
    keep = [c for c in cat.columns if cat[c].nunique() <= MAX_CATEGORY_CARDINALITY]
    dropped = [c for c in cat.columns if c not in keep]
    if dropped:
        print("  dropped high-cardinality text columns:",
              {c: int(cat[c].nunique()) for c in dropped})
    if keep:
        print("  one-hot encoding:", {c: int(cat[c].nunique()) for c in keep})
        num = pd.concat([num, pd.get_dummies(cat[keep], prefix_sep="=").astype(np.float32)], axis=1)
    return num


def sample_per_class(df, rng):
    """Cap each class: BENIGN_ROWS for benign, ATTACK_ROWS_PER_CLASS per attack
    class, so rare attacks are not drowned out by the big flood classes."""
    out = []
    for cls, g in sorted(df.groupby("label_str")):
        cap = BENIGN_ROWS if cls.lower() in BENIGN_LABELS else ATTACK_ROWS_PER_CLASS
        if len(g) > cap:
            g = g.iloc[np.sort(rng.choice(len(g), cap, replace=False))]
        print(f"  {cls:25s} sampled={len(g)}")
        out.append(g)
    return pd.concat(out, ignore_index=True)


def stream_raw():
    """Reads data/raw[_<dataset>]: either per-class folders <Class>/*.csv
    (label = folder name) or files with a class column (CLASS_COL / 'label').
    Rows are sampled per class; with folders the class is known before reading,
    so the sampling probability is applied while streaming to bound RAM."""
    files = sorted(glob.glob(str(RAW_DIR / "**" / "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No CSV files in {RAW_DIR}")
    rng = np.random.default_rng(DATA_SEED)

    if CLASS_COL is None:                      # per-class folders: stream + subsample
        by_class = {}
        for f in files:
            by_class.setdefault(Path(f).parent.name, []).append(f)
        parts = []
        for cls, fs in sorted(by_class.items()):
            n_rows = sum(sum(1 for _ in open(f)) - 1 for f in fs)
            cap = BENIGN_ROWS if cls.lower() in BENIGN_LABELS else ATTACK_ROWS_PER_CLASS
            p = min(1.0, 1.3 * cap / n_rows)   # oversample a bit, trimmed exactly below
            got = []
            for f in fs:
                for ch in pd.read_csv(f, chunksize=CHUNK_ROWS, low_memory=False):
                    lab, ch = label_of(f, ch)
                    keep = rng.random(len(ch)) < p
                    ch = ch.loc[keep].copy()
                    ch["label_str"] = lab[keep].values
                    got.append(ch)
            g = pd.concat(got, ignore_index=True)
            if len(g) > cap:
                g = g.iloc[np.sort(rng.choice(len(g), cap, replace=False))]
            print(f"  {cls:25s} rows={n_rows:>9d} sampled={len(g)}")
            parts.append(g)
        df = pd.concat(parts, ignore_index=True)
    else:                                      # class column: read, then sample per class
        parts = []
        for f in files:
            print("reading", f)
            for ch in pd.read_csv(f, chunksize=CHUNK_ROWS, low_memory=False):
                lab, ch = label_of(f, ch)
                ch["label_str"] = lab.values
                parts.append(ch)
        df = sample_per_class(pd.concat(parts, ignore_index=True), rng)

    X = df.drop(columns=["label_str"] + [c for c in DROP_COLS if c in df.columns])
    X = encode(X).replace([np.inf, -np.inf], np.nan)
    ok = X.notna().all(axis=1).values
    return X[ok].reset_index(drop=True), df["label_str"].values[ok]


def main():
    X, lab = stream_raw()
    y = (~pd.Series(lab).str.lower().isin(BENIGN_LABELS)).astype(np.int8).values
    print(f"\nkept after streaming: benign={(y == 0).sum()} attack={(y == 1).sum()}")

    # Deduplicate on features+label
    key = pd.concat([X, pd.Series(y, name="__y")], axis=1)
    dup = key.duplicated().values
    X, y, lab = X[~dup].reset_index(drop=True), y[~dup], lab[~dup]
    print(f"removed {dup.sum()} exact duplicates")

    # Balance 1:1, capped
    rng = np.random.default_rng(DATA_SEED)
    b, a = np.where(y == 0)[0], np.where(y == 1)[0]
    n = min(len(b), len(a), MAX_PER_CLASS)
    idx = np.concatenate([rng.choice(b, n, replace=False), rng.choice(a, n, replace=False)])
    idx.sort()
    X, y, lab = X.iloc[idx].reset_index(drop=True), y[idx], lab[idx]
    print(f"balanced: {n} per class, {X.shape[1]} features")
    print("attack sub-types in final set:\n", pd.Series(lab[y == 1]).value_counts().to_string())

    np.savez_compressed(PROC_DIR / "dataset.npz", X=X.values.astype(np.float32), y=y,
                        label_str=lab.astype(str), feature_names=np.array(X.columns))

    for s in SEEDS:
        all_i = np.arange(len(y))
        tr, tmp = train_test_split(all_i, test_size=0.3, stratify=y, random_state=s)
        va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=s)
        np.savez(SPLIT_DIR / f"seed_{s}.npz", train=tr, val=va, test=te)
        scaler = MinMaxScaler().fit(X.values[tr])
        joblib.dump(scaler, PROC_DIR / f"scaler_{s}.joblib")
        print(f"seed {s}: train={len(tr)} val={len(va)} test={len(te)}")


if __name__ == "__main__":
    main()
