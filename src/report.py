"""Aggregate results into results/RESULTS.md and figures (mean ± std over seeds)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from config import RESULTS_DIR, FIG_DIR, SEEDS


def pm(df, keys, cols):
    """mean ± std table; silently skips columns a results file does not have
    (e.g. a file written by an older version of the pipeline)."""
    cols = [c for c in cols if c in df.columns]
    keys = [k for k in keys if k in df.columns]
    g = df.groupby(keys)[cols]
    m, s = g.mean(), g.std().fillna(0)
    return (m.map(lambda v: f"{v:.4f}") + " ± " + s.map(lambda v: f"{v:.4f}")).reset_index()


out = [f"# Results\n\nMean ± std over seeds {SEEDS}.\n"]

p = RESULTS_DIR / "clean_metrics_per_seed.csv"
if p.exists():
    out.append("## Clean detection (test set)\n\n" + pm(pd.read_csv(p), ["model"],
               ["accuracy", "precision", "recall", "f1", "auc"]).to_markdown(index=False))

p = RESULTS_DIR / "evasion_per_seed.csv"
if p.exists():
    ev = pd.read_csv(p)
    out.append("## Evasion (fraction of detected malicious flows)\n\n" + pm(
        ev, ["attack", "eps", "target", "setting"],
        ["evasion_rate", "realizable_rate", "realizable_evasion_rate"]).to_markdown(index=False))
    pgd = ev[ev.attack.str.startswith("PGD")].copy()
    pgd["eps"] = pgd.eps.astype(float)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, col in zip(ax, ["evasion_rate", "realizable_evasion_rate"]):
        for (atk, tgt), g in pgd.groupby(["attack", "target"]):
            st = g.groupby("eps")[col].agg(["mean", "std"]).fillna(0)
            a.errorbar(st.index, st["mean"], yerr=st["std"], marker="o", capsize=3,
                       ls="-" if "unconstrained" in atk else "--", label=f"{atk} → {tgt}")
        a.set_title(col.replace("_", " ")); a.set_xlabel("ε (L∞, scaled)"); a.grid(alpha=.3)
    ax[0].set_ylabel("rate"); ax[1].legend(fontsize=7)
    plt.tight_layout(); plt.savefig(FIG_DIR / "evasion_vs_eps.png", dpi=150); plt.close()

p = RESULTS_DIR / "poisoning_per_seed.csv"
if p.exists():
    po = pd.read_csv(p)
    out.append("## Label-flip poisoning (clean test set)\n\n" + pm(
        po, ["flip_rate", "model"], ["precision", "recall", "f1"]).to_markdown(index=False))
    fig, a = plt.subplots(figsize=(5.5, 4))
    for mdl, g in po.groupby("model"):
        st = g.groupby("flip_rate")["recall"].agg(["mean", "std"])
        a.errorbar(st.index, st["mean"], yerr=st["std"], marker="o", capsize=3, label=mdl)
    a.set_xlabel("fraction of attack labels flipped"); a.set_ylabel("attack recall"); a.grid(alpha=.3); a.legend()
    plt.tight_layout(); plt.savefig(FIG_DIR / "poisoning_recall.png", dpi=150); plt.close()

p = RESULTS_DIR / "defense_per_seed.csv"
if p.exists():
    out.append("## Constraint-aware adversarial training (adaptive constrained PGD)\n\n" + pm(
        pd.read_csv(p), ["model", "eps"], ["clean_f1", "clean_fpr", "robust_recall"]).to_markdown(index=False))

p = RESULTS_DIR / "defense_madry_per_seed.csv"
if p.exists():
    out.append("## Madry-style per-batch adversarial training (equal-FPR thresholds)\n\n" + pm(
        pd.read_csv(p), ["model", "eps"], ["clean_f1", "clean_fpr", "robust_recall", "transfer_recall"]).to_markdown(index=False))

p = RESULTS_DIR / "hsj.csv"
if p.exists():
    out.append("## HopSkipJump, decision-only black-box (pooled over seeds, 95% Wilson CI)\n\n"
               + pd.read_csv(p).to_markdown(index=False))

p = RESULTS_DIR / "detectors_per_seed.csv"
if p.exists():
    out.append("## Adversarial-example detectors (5% false-alarm budget, non-adaptive attacks)\n\n" + pm(
        pd.read_csv(p), ["detector", "attack"],
        ["adv_confidence", "auc_successful", "tpr_successful", "false_alarm_benign",
         "recall_mlp_only", "recall_combined"]).to_markdown(index=False))

p = RESULTS_DIR / "backdoor_per_seed.csv"
if p.exists():
    out.append("## Backdoor poisoning and spectral-signature defense\n\n" + pm(
        pd.read_csv(p), ["rate", "defense", "model"],
        ["asr", "f1", "poison_removed", "clean_removed"]).to_markdown(index=False))

(RESULTS_DIR / "RESULTS.md").write_text("\n\n".join(out) + "\n")
print(f"Wrote {RESULTS_DIR / 'RESULTS.md'} and figures in {FIG_DIR}")
