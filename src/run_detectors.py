"""Adversarial-example DETECTION (vs. robust classification) for the MLP NIDS.

Detectors (all unsupervised: no adversarial examples used to fit them):
- Feature squeezing (Xu et al., NDSS 2018), tabular version: squeezers reduce
  the bit depth of the [0,1]-scaled features; score = max L1 distance between
  the model's softmax on x and on squeeze(x). Median smoothing is image-only.
- Manda-style (after Wang et al., INFOCOM 2021): our reimplementation of its two
  stated ideas, NOT the authors' code or exact configuration:
    (a) manifold inconsistency: |p_MLP(x) - p_kNN(x)|, kNN fit on clean train data;
    (b) decision-boundary proximity: fraction of Gaussian-noised copies of x
        whose MLP label differs from the label of x.
  The two scores are combined by averaging their percentile ranks on clean val data.

Protocol per seed:
- X0 = malicious test flows the MLP detects; attacked with ART PGD against the MLP:
  unconstrained eps 0.1 (50 steps); constrained eps 0.2 with 3 steps (low-confidence
  examples near the boundary) and 50 steps (high-confidence examples).
- Detector threshold = 95th percentile of its score on clean VALIDATION flows
  (5% false-alarm budget, fixed before seeing any adversarial example).
- Reported: AUC (adversarial vs. clean test flows; auc_successful uses only the
  examples that actually evade the MLP), mean confidence of successful examples, TPR on all / successful
  adversarial examples, false-alarm rate on clean benign test flows, and the
  operational number: combined recall = P(MLP says attack OR detector flags).
Limitation: attacks are NOT adaptive to the detectors (see README).
"""
import os
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from art.attacks.evasion import ProjectedGradientDescent
from common import load_seed, load_models, art_mlp
from constraints import Constraints
from config import SEEDS, RESULTS_DIR, N_JOBS
from models import mlp_proba

N = int(os.environ.get("NIDS_N_DET", 1000))
N_CLEAN = 2000
BITS = [3, 4, 5]
KNN_K, KNN_FIT = 10, 20000
NOISE_SIGMA, NOISE_COPIES = 0.02, 20
FPR_BUDGET = 0.05


def squeeze_score(m, X):
    p = mlp_proba(m, X)
    return np.max([np.abs(p - mlp_proba(m, np.round(X * (2 ** b - 1)) / (2 ** b - 1))) for b in BITS], axis=0)


class MandaStyle:
    def __init__(self, m, Xtr, ytr, Xval, seed):
        self.m, self.rng = m, np.random.default_rng(seed)
        idx = self.rng.choice(len(Xtr), min(KNN_FIT, len(Xtr)), replace=False)
        self.knn = KNeighborsClassifier(KNN_K, n_jobs=N_JOBS).fit(Xtr[idx], ytr[idx])
        self.ref_inc = np.sort(self.inconsistency(Xval))
        self.ref_bnd = np.sort(self.boundary(Xval))

    def inconsistency(self, X):
        return np.abs(mlp_proba(self.m, X) - self.knn.predict_proba(X)[:, 1])

    def boundary(self, X):
        lab = mlp_proba(self.m, X) >= 0.5
        flips = np.zeros(len(X))
        for _ in range(NOISE_COPIES):
            Xn = np.clip(X + self.rng.normal(0, NOISE_SIGMA, X.shape), 0, 1).astype(np.float32)
            flips += (mlp_proba(self.m, Xn) >= 0.5) != lab
        return flips / NOISE_COPIES

    def score(self, X):
        r = lambda ref, v: np.searchsorted(ref, v, side="right") / len(ref)
        return 0.5 * (r(self.ref_inc, self.inconsistency(X)) + r(self.ref_bnd, self.boundary(X)))


def main():
    rows = []
    for s in SEEDS:
        D = load_seed(s)
        Xtr, ytr, Xva, Xte, yte = D["Xtr"], D["ytr"], D["Xva"], D["Xte"], D["yte"]
        d = Xtr.shape[1]
        mlp, _, _ = load_models(s, d)
        cons = Constraints(D["feature_names"], D["scaler"], Xtr)
        rng = np.random.default_rng(s)

        mal_idx = np.where((yte == 1) & (mlp_proba(mlp, Xte) >= 0.5))[0]
        a_idx = rng.choice(mal_idx, min(N, len(mal_idx)), replace=False)
        X0 = Xte[a_idx]
        rest = np.setdiff1d(np.arange(len(yte)), a_idx)
        c_idx = rng.choice(rest, min(N_CLEAN, len(rest)), replace=False)
        Xclean, ben = Xte[c_idx], Xte[yte == 0]

        clf, Y0 = art_mlp(mlp, d), np.eye(2)[np.ones(len(X0), int)]
        # Attack strength matters: few PGD steps give low-confidence examples just
        # across the boundary; many steps push deep into the benign region.
        def pgd(eps, iters, constrained):
            a = ProjectedGradientDescent(clf, norm=np.inf, eps=eps, eps_step=eps / 10, max_iter=iters,
                                         batch_size=512, verbose=False)
            return cons.project(a.generate(X0, y=Y0, mask=cons.mask), X0) if constrained else a.generate(X0, y=Y0)
        advs = {"PGD-unconstrained eps=0.1 iters=50": pgd(0.1, 50, False),
                "PGD-constrained eps=0.2 iters=3": pgd(0.2, 3, True),
                "PGD-constrained eps=0.2 iters=50": pgd(0.2, 50, True)}

        manda = MandaStyle(mlp, Xtr, ytr, Xva, s)
        dets = {"FeatureSqueezing": lambda X: squeeze_score(mlp, X), "Manda-style": manda.score}
        for dname, score in dets.items():
            thr = np.quantile(score(Xva), 1 - FPR_BUDGET)
            s_clean, s_ben = score(Xclean), score(ben)
            for aname, Xa in advs.items():
                s_adv = score(Xa)
                flag = s_adv > thr
                evaded = mlp_proba(mlp, Xa) < 0.5
                rows.append(dict(seed=s, detector=dname, attack=aname, n_adv=len(Xa),
                                 evasion_rate=evaded.mean(),
                                 auc=roc_auc_score(np.r_[np.zeros(len(s_clean)), np.ones(len(s_adv))],
                                                   np.r_[s_clean, s_adv]),
                                 auc_successful=roc_auc_score(np.r_[np.zeros(len(s_clean)), np.ones(evaded.sum())],
                                                              np.r_[s_clean, s_adv[evaded]]) if evaded.any() else np.nan,
                                 adv_confidence=float(np.abs(2 * mlp_proba(mlp, Xa[evaded]) - 1).mean()) if evaded.any() else np.nan,
                                 tpr_all=flag.mean(),
                                 tpr_successful=flag[evaded].mean() if evaded.any() else np.nan,
                                 false_alarm_benign=float((s_ben > thr).mean()),
                                 recall_mlp_only=float((~evaded).mean()),
                                 recall_combined=float((~evaded | flag).mean())))
                r = rows[-1]
                print(f"seed {s} {dname:16s} {aname:34s} AUC(succ)={r['auc_successful']:.3f} "
                      f"TPR(succ)={r['tpr_successful']:.3f} recall {r['recall_mlp_only']:.3f}"
                      f"->{r['recall_combined']:.3f} FA={r['false_alarm_benign']:.3f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "detectors_per_seed.csv", index=False)
    agg = df.drop(columns=["seed", "n_adv"]).groupby(["detector", "attack"]).agg(["mean", "std"]).round(4)
    agg.to_csv(RESULTS_DIR / "detectors.csv")
    print("\n", agg.to_string())


if __name__ == "__main__":
    main()
