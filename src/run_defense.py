"""Constraint-aware adversarial training of the MLP, evaluated against an ADAPTIVE
attacker (constrained PGD recomputed against each model it is attacking)."""
import os
import numpy as np
import pandas as pd
from art.attacks.evasion import ProjectedGradientDescent
from common import load_seed, load_models, art_mlp, metrics
from constraints import Constraints
from config import SEEDS, RESULTS_DIR
from models import train_mlp, mlp_predict, mlp_proba

TRAIN_EPS, ROUNDS = 0.1, 2
N_ADV = int(os.environ.get("NIDS_N_ADV", 20000))
EVAL_EPS = [0.01, 0.05, 0.1, 0.2]


def constrained_pgd(m, X, cons, eps):
    pgd = ProjectedGradientDescent(art_mlp(m, X.shape[1]), norm=np.inf, eps=eps, eps_step=eps / 10,
                                   max_iter=50, batch_size=512, verbose=False)
    return cons.project(pgd.generate(X, y=np.eye(2)[np.ones(len(X), int)], mask=cons.mask), X)


def main():
    rows = []
    for s in SEEDS:
        D = load_seed(s)
        Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
        cons = Constraints(D["feature_names"], D["scaler"], Xtr)
        base, _, _ = load_models(s, Xtr.shape[1])
        rng = np.random.default_rng(s)
        m = base
        for r in range(ROUNDS):
            mal = Xtr[ytr == 1]
            mal = mal[rng.choice(len(mal), min(N_ADV, len(mal)), replace=False)]
            adv = constrained_pgd(m, mal, cons, TRAIN_EPS)
            m = train_mlp(np.vstack([Xtr, adv]), np.concatenate([ytr, np.ones(len(adv), int)]), s)
            print(f"seed {s}: adversarial training round {r + 1}/{ROUNDS} done")
        mal_te = Xte[yte == 1][:2000]
        for name, model in [("MLP (baseline)", base), ("MLP (adv-trained)", m)]:
            clean = metrics(yte, mlp_predict(model, Xte), mlp_proba(model, Xte))
            for eps in EVAL_EPS:
                Xa = constrained_pgd(model, mal_te, cons, eps)
                rows.append(dict(seed=s, model=name, eps=eps, clean_f1=clean["f1"],
                                 clean_fpr=float((mlp_predict(model, Xte[yte == 0]) == 1).mean()),
                                 robust_recall=float((mlp_predict(model, Xa) == 1).mean())))
            print(f"  {name}: clean_f1={clean['f1']:.4f} robust_recall@0.1="
                  f"{[r['robust_recall'] for r in rows if r['seed'] == s and r['model'] == name and r['eps'] == 0.1][0]:.4f}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "defense_per_seed.csv", index=False)
    agg = df.drop(columns="seed").groupby(["model", "eps"]).agg(["mean", "std"]).round(4)
    agg.to_csv(RESULTS_DIR / "defense.csv")
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
