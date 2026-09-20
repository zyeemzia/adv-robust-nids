"""HopSkipJump (decision-only black-box) with a larger sample than run_attacks.py.

Same attack settings as run_attacks.py (masked to modifiable features, projected
onto the constraints), but N_HSJ flows per seed instead of 50, and a pooled 95%
Wilson confidence interval so the XGB evasion estimate is statistically firm.
Targets: XGB (strongest clean model) and RF, each attacked directly.
"""
import os
import signal
import time
import numpy as np
import pandas as pd
from art.attacks.evasion import HopSkipJump
from art.estimators.classification import BlackBoxClassifier
from common import load_seed, load_models
from constraints import Constraints
from config import SEEDS, RESULTS_DIR
from models import mlp_predict

N_HSJ = int(os.environ.get("NIDS_N_HSJ", 300))
# RF prediction on HSJ's small query batches is ~20x slower than XGB on 2 cores
# (~15 s/flow vs ~0.7 s/flow), so RF gets a smaller sample.
N_PER_TARGET = {"XGB": N_HSJ, "RF": max(1, N_HSJ // 5)}
# HSJ occasionally never terminates on a flow (seen once: >5 h on one seed).
# Each flow gets a wall-clock budget; a timed-out flow counts as NOT evaded
# (conservative for the attacker) and is reported in the `timeouts` column.
FLOW_TIMEOUT_S = int(os.environ.get("NIDS_HSJ_TIMEOUT", 120))
PER_SEED_CSV = RESULTS_DIR / "hsj_per_seed.csv"


class FlowTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise FlowTimeout


def attack_flows(hsj, Xs, mask):
    """Attack one flow at a time under a timeout. Returns (X_adv, n_timeouts)."""
    signal.signal(signal.SIGALRM, _alarm)
    out, timeouts = Xs.copy(), 0
    for i in range(len(Xs)):
        signal.alarm(FLOW_TIMEOUT_S)
        try:
            out[i] = hsj.generate(Xs[i:i + 1], mask=mask[None, :])[0]
        except FlowTimeout:
            timeouts += 1          # keep the original flow -> not evaded
        finally:
            signal.alarm(0)
    return out, timeouts


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def main():
    # Resume: seeds already saved in PER_SEED_CSV are skipped, so a crash or
    # hang never loses completed seeds.
    rows = pd.read_csv(PER_SEED_CSV).to_dict("records") if PER_SEED_CSV.exists() else []
    done = {(r["seed"], r["target"]) for r in rows}
    for s in SEEDS:
        if all((s, t) in done for t in N_PER_TARGET):
            print(f"seed {s}: already done, skipping")
            continue
        D = load_seed(s)
        d = D["Xtr"].shape[1]
        mlp, rf, xgb = load_models(s, d)
        cons = Constraints(D["feature_names"], D["scaler"], D["Xtr"])
        preds = {"MLP": lambda X: mlp_predict(mlp, X), "RF": rf.predict, "XGB": xgb.predict}
        mal = D["Xte"][D["yte"] == 1]
        det = np.ones(len(mal), bool)
        for f in preds.values():
            det &= f(mal) == 1
        rng = np.random.default_rng(s)
        X0 = mal[det][rng.choice(det.sum(), min(N_HSJ, det.sum()), replace=False)]
        for tname, model in [("XGB", xgb), ("RF", rf)]:
            t = time.time()
            Xs = X0[:N_PER_TARGET[tname]]
            bb = BlackBoxClassifier(lambda X, m=model: np.eye(2)[m.predict(X.astype(np.float32)).astype(int)],
                                    input_shape=(d,), nb_classes=2, clip_values=(0.0, 1.0))
            hsj = HopSkipJump(bb, targeted=False, norm=2, max_iter=10, max_eval=500,
                              init_eval=50, verbose=False)
            if (s, tname) in done:
                continue
            Xa, n_to = attack_flows(hsj, Xs, cons.mask)
            Xa = cons.project(Xa, Xs)
            ok, _ = cons.audit(Xa, Xs)
            ev = model.predict(Xa) == 0
            rows.append(dict(seed=s, target=tname, n=len(Xs), timeouts=n_to, evaded=int((ev & ok).sum()),
                             realizable_rate=ok.mean(), realizable_evasion_rate=(ev & ok).mean(),
                             median_l2=float(np.median(np.linalg.norm(Xa - Xs, axis=1)[ev])) if ev.any() else np.nan))
            print(f"seed {s} HSJ->{tname}: evasion={rows[-1]['realizable_evasion_rate']:.3f} "
                  f"(n={len(Xs)}, timeouts={n_to}, {time.time() - t:.0f}s)", flush=True)
            pd.DataFrame(rows).to_csv(PER_SEED_CSV, index=False)

    df = pd.DataFrame(rows)
    out = []
    for tname, g in df.groupby("target"):
        k, n = g.evaded.sum(), g.n.sum()
        lo, hi = wilson(k, n)
        out.append(dict(target=tname, n_total=n, timeouts=int(g.timeouts.sum()), evaded=k, pooled_rate=k / n, ci95_low=lo, ci95_high=hi,
                        seed_mean=g.realizable_evasion_rate.mean(), seed_std=g.realizable_evasion_rate.std(),
                        realizable_rate=g.realizable_rate.mean(), median_l2=g.median_l2.median()))
    agg = pd.DataFrame(out).round(4)
    agg.to_csv(RESULTS_DIR / "hsj.csv", index=False)
    print("\n", agg.to_string(index=False))


if __name__ == "__main__":
    main()
