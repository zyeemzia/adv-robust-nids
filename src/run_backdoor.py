"""Dirty-label backdoor poisoning + spectral-signature defense.

Attack: the attacker injects n = rate * |train| malicious training flows carrying
a trigger, labelled BENIGN. Goal: at test time, adding the trigger to any attack
flow makes the detector call it benign, while clean accuracy stays intact.
Trigger: config.TRIGGER_FEATS (ciciot2023: Header_Length + Tot size; toniot:
src_bytes + src_pkts) set to their train 99.5th percentile in raw units, then
projected with Constraints, so the trigger is realizable under the threat model
(all trigger features are attacker-controllable and increase-only).

Metrics (clean test set, per seed):
- clean F1 / recall (stealth: should stay close to the unpoisoned model)
- ASR = fraction of malicious test flows that the UNPOISONED model detects
  without the trigger but the evaluated model misses WITH the trigger.
  At rate 0 this is the control: pure evasion effect of the trigger itself.

Defense: spectral signatures (Tran et al., NeurIPS 2018). For training flows
labelled benign, take penultimate-layer MLP activations, centre them, score
each flow by its squared projection on the top singular vector, remove the top
1.5 * rate fraction (the poison budget the defender assumes), and retrain all
three models on the filtered data.
"""
import numpy as np
import pandas as pd
import torch
from common import load_seed, load_models, metrics
from constraints import Constraints
from config import SEEDS, RESULTS_DIR, TRIGGER_FEATS
from models import train_rf, train_xgb, train_mlp, mlp_predict

RATES = [0.0, 0.001, 0.005, 0.01, 0.02]
TRIGGER_Q = 0.995
REMOVE_FACTOR = 1.5
FITS = [("RF", train_rf, lambda m, X: m.predict(X)),
        ("XGB", train_xgb, lambda m, X: m.predict(X)),
        ("MLP", train_mlp, mlp_predict)]


def make_trigger(cons, Xtr):
    raw = cons.to_raw(Xtr)
    idx = [cons.i[f] for f in TRIGGER_FEATS]
    vals = np.quantile(raw[:, idx], TRIGGER_Q, axis=0)

    def apply(X):
        R = cons.to_raw(X)
        R[:, idx] = vals
        return cons.project(cons.to_scaled(R), X)      # increase-only etc. enforced
    return apply


def spectral_filter(Xtr, ytr, seed, remove_frac):
    """Return a keep-mask over training rows (only label-0 rows can be removed)."""
    m = train_mlp(Xtr, ytr, seed)
    feat = torch.nn.Sequential(*list(m.net.children())[:-1])     # penultimate activations
    ben = np.where(ytr == 0)[0]
    with torch.no_grad():
        A = feat(torch.tensor(Xtr[ben])).numpy()
    A = A - A.mean(0)
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    score = (A @ vt[0]) ** 2
    n_rm = int(remove_frac * len(Xtr))
    keep = np.ones(len(ytr), bool)
    if n_rm:
        keep[ben[np.argsort(score)[-n_rm:]]] = False
    return keep


def main():
    rows = []
    for s in SEEDS:
        D = load_seed(s)
        Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
        cons = Constraints(D["feature_names"], D["scaler"], Xtr)
        trig = make_trigger(cons, Xtr)
        rng = np.random.default_rng(s)
        mlp0, rf0, xgb0 = load_models(s, Xtr.shape[1])
        clean_pred = {"RF": rf0.predict, "XGB": xgb0.predict, "MLP": lambda X: mlp_predict(mlp0, X)}
        mal_te = Xte[yte == 1]
        mal_te_trig = trig(mal_te)
        audit_ok, _ = cons.audit(mal_te_trig, mal_te)
        mal_tr = Xtr[ytr == 1]

        for r in RATES:
            n_p = int(r * len(Xtr))
            P = trig(mal_tr[rng.choice(len(mal_tr), n_p, replace=False)]) if n_p else Xtr[:0]
            Xp = np.vstack([Xtr, P]).astype(np.float32)
            yp = np.concatenate([ytr, np.zeros(n_p, int)])
            is_poison = np.r_[np.zeros(len(Xtr), bool), np.ones(n_p, bool)]
            settings = [("none", np.ones(len(yp), bool))]
            if n_p:
                settings.append(("spectral", spectral_filter(Xp, yp, s, REMOVE_FACTOR * r)))
            for defense, keep in settings:
                caught = float((~keep & is_poison).sum() / max(n_p, 1))
                removed_clean = int((~keep & ~is_poison).sum())
                for name, fit, pred in FITS:
                    m = fit(Xp[keep], yp[keep], s)
                    base_det = clean_pred[name](mal_te) == 1
                    asr = float(((pred(m, mal_te_trig) == 0) & base_det).sum() / base_det.sum())
                    rows.append(dict(seed=s, rate=r, n_poison=n_p, defense=defense, model=name,
                                     poison_removed=caught, clean_removed=removed_clean,
                                     trigger_realizable=float(audit_ok.mean()), asr=asr,
                                     **metrics(yte, pred(m, Xte))))
                print(f"seed {s} rate {r} def={defense:8s} caught={caught:.2f} " +
                      " ".join(f"{x['model']}: ASR={x['asr']:.3f} F1={x['f1']:.3f}" for x in rows[-3:]),
                      flush=True)
        pd.DataFrame(rows).to_csv(RESULTS_DIR / "backdoor_per_seed.csv", index=False)

    df = pd.DataFrame(rows)
    agg = (df.drop(columns=["seed"]).groupby(["rate", "defense", "model"])
           [["asr", "f1", "recall", "poison_removed", "clean_removed"]].agg(["mean", "std"]).round(4))
    agg.to_csv(RESULTS_DIR / "backdoor.csv")
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
