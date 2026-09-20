"""Evasion: unconstrained vs. constrained PGD (white-box on MLP, transfer to RF/XGB)
and masked HopSkipJump (black-box, decision-only, directly on XGB)."""
import os
import numpy as np
import pandas as pd
from art.attacks.evasion import ProjectedGradientDescent, HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from common import load_seed, load_models, art_mlp
from constraints import Constraints
from config import SEEDS, RESULTS_DIR
from models import mlp_predict

EPS = [0.01, 0.05, 0.1, 0.2]
N = int(os.environ.get("NIDS_N_ATTACK", 2000))
N_HSJ = int(os.environ.get("NIDS_N_HSJ", 50))     # HSJ is slow on 2 cores


def run_seed(s):
    D = load_seed(s)
    d = D["Xtr"].shape[1]
    mlp, rf, xgb = load_models(s, d)
    cons = Constraints(D["feature_names"], D["scaler"], D["Xtr"])
    preds = {"MLP": lambda X: mlp_predict(mlp, X), "RF": rf.predict, "XGB": xgb.predict}

    # attack only malicious test flows that every model currently detects
    mal = D["Xte"][D["yte"] == 1]
    det = np.ones(len(mal), bool)
    for f in preds.values():
        det &= f(mal) == 1
    rng = np.random.default_rng(s)
    X0 = mal[det][rng.choice(det.sum(), min(N, det.sum()), replace=False)]
    Y0 = np.eye(2)[np.ones(len(X0), int)]
    print(f"seed {s}: attacking {len(X0)} detected malicious flows")
    clf, rows = art_mlp(mlp, d), []

    def record(attack, eps, Xa, Xo, surrogate):
        ok, checks = cons.audit(Xa, Xo)
        for name, f in preds.items():
            ev = f(Xa) == 0
            rows.append(dict(seed=s, attack=attack, eps=eps, target=name,
                             setting="white-box" if name == surrogate else "transfer",
                             evasion_rate=ev.mean(), realizable_rate=ok.mean(),
                             realizable_evasion_rate=(ev & ok).mean(),
                             **{f"chk_{k}": v for k, v in checks.items()}))
        print(f"  {attack:18s} eps={eps}: realizable={ok.mean():.3f} evasion "
              + " ".join(f"{r['target']}={r['evasion_rate']:.3f}" for r in rows[-3:]))

    for eps in EPS:
        pgd = ProjectedGradientDescent(clf, norm=np.inf, eps=eps, eps_step=eps / 10, max_iter=50,
                                       num_random_init=1, batch_size=512, verbose=False)
        record("PGD-unconstrained", eps, pgd.generate(X0, y=Y0), X0, "MLP")
        record("PGD-constrained", eps, cons.project(pgd.generate(X0, y=Y0, mask=cons.mask), X0), X0, "MLP")

    bb = BlackBoxClassifier(lambda X: np.eye(2)[xgb.predict(X.astype(np.float32)).astype(int)],
                            input_shape=(d,), nb_classes=2, clip_values=(0.0, 1.0))
    Xh = X0[:N_HSJ]
    hsj = HopSkipJump(bb, targeted=False, norm=2, max_iter=10, max_eval=500, init_eval=50, verbose=False)
    Xa = hsj.generate(Xh, mask=np.tile(cons.mask, (len(Xh), 1)))
    record("HSJ-masked", "-", cons.project(Xa, Xh), Xh, "XGB")
    return rows


def main():
    rows = [r for s in SEEDS for r in run_seed(s)]
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "evasion_per_seed.csv", index=False)
    cols = ["evasion_rate", "realizable_rate", "realizable_evasion_rate"]
    agg = df.groupby(["attack", "eps", "target", "setting"])[cols].agg(["mean", "std"]).round(4)
    agg.to_csv(RESULTS_DIR / "evasion.csv")
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
