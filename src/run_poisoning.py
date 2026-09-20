"""Label-flip poisoning: attacker relabels a fraction of malicious TRAIN flows as
benign (the realistic direction: hide attacks). Test set is always clean."""
import numpy as np
import pandas as pd
from common import load_seed, metrics
from config import SEEDS, RESULTS_DIR
from models import train_rf, train_xgb, train_mlp, mlp_predict

RATES = [0.0, 0.05, 0.1, 0.2, 0.3]


def main():
    rows = []
    for s in SEEDS:
        D = load_seed(s)
        Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
        rng = np.random.default_rng(s)
        att = np.where(ytr == 1)[0]
        for r in RATES:
            yp = ytr.copy()
            yp[rng.choice(att, int(r * len(att)), replace=False)] = 0
            for name, fit, pred in [("RF", train_rf, lambda m, X: m.predict(X)),
                                    ("XGB", train_xgb, lambda m, X: m.predict(X)),
                                    ("MLP", train_mlp, mlp_predict)]:
                rows.append(dict(seed=s, flip_rate=r, model=name, **metrics(yte, pred(fit(Xtr, yp, s), Xte))))
            print(f"seed {s} flip {r}: " + " ".join(f"{x['model']} recall={x['recall']:.3f}" for x in rows[-3:]))
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "poisoning_per_seed.csv", index=False)
    agg = df.drop(columns="seed").groupby(["flip_rate", "model"]).agg(["mean", "std"]).round(4)
    agg.to_csv(RESULTS_DIR / "poisoning.csv")
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
