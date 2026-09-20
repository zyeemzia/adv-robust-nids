"""Generates FAKE flow data with a CIC-IoT2023-like schema, ONLY to verify the
pipeline runs. Never report numbers from this data."""
import numpy as np, pandas as pd, sys
from pathlib import Path
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
cols = ["flow_duration","Header_Length","Protocol Type","Duration","Rate","syn_flag_number",
        "ack_flag_number","Tot sum","Min","Max","AVG","Std","Tot size","IAT","Number"]
for f in range(2):
    n = 20000
    y = rng.random(n) < 0.8
    X = rng.gamma(2, 1, (n, len(cols))) + y[:, None] * rng.normal(0.8, 0.5, (n, len(cols)))
    df = pd.DataFrame(np.abs(X), columns=cols)
    df["Min"] = df[["Min","Max"]].min(1); df["Max"] = df[["Min","Max"]].max(1) + 0.1
    df["AVG"] = (df.Min + df.Max) / 2; df["Std"] = (df.Max - df.Min) / 4
    df["label"] = np.where(y, rng.choice(["DDoS-SYN_Flood","Recon-PortScan"], n), "BenignTraffic")
    df.to_csv(out / f"part-{f:05d}-synthetic.csv", index=False)
print("wrote synthetic CSVs to", out)
