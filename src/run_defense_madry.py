"""Madry-style (per-batch) constraint-aware adversarial training.

Differences from run_defense.py (2-round offline retraining, which failed):
- Adversarial examples are regenerated EVERY batch against the CURRENT weights,
  so the model never trains on stale attacks (min-max objective of Madry et al.).
- The PGD inner loop projects onto the threat-model constraints at every step.
- Loss = 0.5 * clean loss + 0.5 * adversarial loss on the batch's attack flows,
  which limits the clean-accuracy / false-positive cost.
- Models are compared at the SAME false-positive rate: each model's threshold is
  calibrated on the validation benign flows to match the baseline's val FPR at
  0.5. Without this, a model that flags everything looks perfectly "robust".
Evaluation reuses run_defense.constrained_pgd (same adaptive white-box attack)
plus a transfer attack from the baseline model, to detect gradient masking.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import load_seed, load_models, metrics
from constraints import Constraints
from config import SEEDS, RESULTS_DIR, MODEL_DIR
from models import MLP, set_seed, mlp_predict, mlp_proba
from run_defense import constrained_pgd, EVAL_EPS

TRAIN_EPS, PGD_STEPS, EPOCHS, BS = 0.1, 10, 15, 512


def pgd_batch(m, x, cons, eps, steps):
    """Untargeted constrained PGD on attack flows (label 1), projected every step."""
    mask = torch.tensor(cons.mask)
    x0 = x.numpy()
    xa = x + (torch.rand_like(x) * 2 - 1) * eps * mask
    xa = torch.tensor(cons.project(xa.clamp(0, 1).numpy(), x0))
    y = torch.ones(len(x), dtype=torch.long)
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = nn.functional.cross_entropy(m(xa), y)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() + (eps / 4) * g.sign() * mask
        xa = torch.max(torch.min(xa, x + eps), x - eps).clamp(0, 1)
        xa = torch.tensor(cons.project(xa.numpy(), x0))
    return xa.detach()


def train_madry(X, y, cons, seed):
    set_seed(seed)
    m = MLP(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt, yt = torch.tensor(X), torch.tensor(y, dtype=torch.long)
    for ep in range(EPOCHS):
        perm, tot = torch.randperm(len(Xt)), 0.0
        for i in range(0, len(Xt), BS):
            j = perm[i:i + BS]
            xb, yb = Xt[j], yt[j]
            mal = yb == 1
            m.eval()
            xadv = pgd_batch(m, xb[mal], cons, TRAIN_EPS, PGD_STEPS) if mal.any() else xb[:0]
            m.train()
            opt.zero_grad()
            loss = 0.5 * nn.functional.cross_entropy(m(xb), yb)
            if len(xadv):
                loss = loss + 0.5 * nn.functional.cross_entropy(m(xadv), torch.ones(len(xadv), dtype=torch.long))
            loss.backward(); opt.step()
            tot += loss.item() * len(j)
        print(f"  seed {seed} epoch {ep + 1}/{EPOCHS} loss={tot / len(Xt):.4f}", flush=True)
    m.eval()
    return m


def main():
    rows = []
    for s in SEEDS:
        D = load_seed(s)
        Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
        cons = Constraints(D["feature_names"], D["scaler"], Xtr)
        base, _, _ = load_models(s, Xtr.shape[1])
        madry = train_madry(Xtr, ytr, cons, s)
        torch.save(madry.state_dict(), MODEL_DIR / f"mlp_madry_{s}.pt")

        mal_te = Xte[yte == 1][:2000]
        ben_va = D["Xva"][D["yva"] == 0]
        target_fpr = float((mlp_proba(base, ben_va) >= 0.5).mean())
        for name, model in [("MLP (baseline)", base), ("MLP (Madry-AT)", madry)]:
            thr = float(np.quantile(mlp_proba(model, ben_va), 1 - target_fpr))
            pred = lambda X, mm=model, t=thr: (mlp_proba(mm, X) >= t).astype(int)
            clean = metrics(yte, pred(Xte), mlp_proba(model, Xte))
            fpr = float(pred(Xte[yte == 0]).mean())
            for eps in EVAL_EPS:
                adaptive = constrained_pgd(model, mal_te, cons, eps)       # white-box vs this model
                transfer = constrained_pgd(base, mal_te, cons, eps)        # crafted on baseline
                rows.append(dict(seed=s, model=name, eps=eps, threshold=thr, clean_f1=clean["f1"],
                                 clean_auc=clean["auc"], clean_fpr=fpr,
                                 clean_recall_subset=float(pred(mal_te).mean()),
                                 robust_recall=float(pred(adaptive).mean()),
                                 transfer_recall=float(pred(transfer).mean())))
            r = [x for x in rows if x["seed"] == s and x["model"] == name and x["eps"] == 0.1][0]
            print(f"  {name}: clean_f1={r['clean_f1']:.4f} fpr={fpr:.4f} "
                  f"robust@0.1={r['robust_recall']:.4f} transfer@0.1={r['transfer_recall']:.4f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "defense_madry_per_seed.csv", index=False)
    agg = df.drop(columns="seed").groupby(["model", "eps"]).agg(["mean", "std"]).round(4)
    agg.to_csv(RESULTS_DIR / "defense_madry.csv")
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
