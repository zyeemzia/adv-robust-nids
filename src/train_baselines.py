"""Train RF / XGB / MLP for every seed; report clean test metrics + leakage check."""
import pandas as pd
from common import load_seed, save_models, metrics
from config import SEEDS, RESULTS_DIR
from models import train_mlp, train_rf, train_xgb, mlp_proba, mlp_predict


def main():
    rows, imps = [], []
    for s in SEEDS:
        print(f"\n=== seed {s} ===")
        D = load_seed(s)
        Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
        rf = train_rf(Xtr, ytr, s)
        xgb = train_xgb(Xtr, ytr, s)
        mlp = train_mlp(Xtr, ytr, s)
        save_models(s, mlp, rf, xgb)
        for name, pred, score in [("RF", rf.predict(Xte), rf.predict_proba(Xte)[:, 1]),
                                  ("XGB", xgb.predict(Xte), xgb.predict_proba(Xte)[:, 1]),
                                  ("MLP", mlp_predict(mlp, Xte), mlp_proba(mlp, Xte))]:
            rows.append(dict(seed=s, model=name, **metrics(yte, pred, score)))
            print(rows[-1])
        imps.append(pd.Series(xgb.feature_importances_, index=D["feature_names"], name=s))

    imp = pd.concat(imps, axis=1).mean(1).sort_values(ascending=False)
    imp.to_csv(RESULTS_DIR / "xgb_feature_importance.csv", header=["importance"])
    print("\nTop-10 mean XGB importances:\n", imp.head(10).round(4).to_string())
    if imp.iloc[0] > 0.5:
        print(f"\n!! LEAKAGE WARNING: '{imp.index[0]}' has importance {imp.iloc[0]:.2f}. "
              "Consider adding it to DROP_COLS and rerunning data.py.")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "clean_metrics_per_seed.csv", index=False)
    agg = df.drop(columns="seed").groupby("model").agg(["mean", "std"]).round(4)
    agg.to_csv(RESULTS_DIR / "clean_metrics.csv")
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
