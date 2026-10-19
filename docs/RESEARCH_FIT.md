# How this repository relates to specific research groups

This note maps findings in this repository to the published directions of the
groups I am applying to work with. Every claim links to a measured result in
`results/`; nothing here is aspirational. Descriptions of other people's work
are my reading of their published record and may be out of date.

---

## Adversarial ML for network intrusion detection

**Relevant results.** Unconstrained PGD evades the MLP on 91.3% of flows at
epsilon=0.2, and 0% of those adversarial examples survive the constraint audit
— every one modifies a feature the attacker does not control. Under the threat
model in `README.md`, constrained PGD still evades on 43.8 +/- 6.8%, and a
decision-only HopSkipJump attacker restricted to modifiable features evades
XGBoost on 43.2% (95% Wilson CI 40.7–45.7%). The same qualitative result holds
on TON_IoT under a different feature set and a different threat model.

This is the position argued by elShehaby & Matrawy (arXiv:2306.05494): measured
on both datasets, unconstrained feature-space attacks overstate the threat,
while constrained black-box attacks remain a genuine one. The repository
separates `evasion_rate` from `realizable_evasion_rate` everywhere so the two
cannot be confused.

**Where I would want to take it.** Learned rather than hand-specified
constraints; constrained attacks in the problem space (generating packets
rather than feature vectors) so realizability is proven by construction rather
than checked; and extending the audit to dynamic/streaming detectors, where
the realizability question is harder.

## Adversarial-example detection

**Relevant results.** `run_detectors.py` reimplements the two core ideas of
MANDA (Wang et al., INFOCOM 2021) — inconsistency between the NIDS decision and
a manifold model, plus decision-boundary proximity — and benchmarks it against
feature squeezing (Xu et al., NDSS 2018) at a 5% false-alarm budget. The
Manda-style detector holds 0.76–0.83 AUC across attack strengths, while feature
squeezing collapses to 0.26 AUC against high-confidence attacks. But at a 5%
false-alarm rate the detector flags only 4–27% of successful evasions, raising
attack recall from 0.524 to 0.570 — far less than Madry adversarial training
achieves (0.787 at epsilon=0.2).

My reading is that detection-based defenses are being evaluated under
conditions that flatter them, and that an equal-false-positive comparison
against robust training should be reported alongside any detector. This is a
reimplementation of two published ideas, not the authors' code, so the
comparison is indicative rather than definitive.

**Where I would want to take it.** Adaptive attackers that optimise against the
detector and the classifier jointly; contrastive representations as the
manifold model instead of my simpler one; and whether detection and robust
training compose or interfere.

## Backdoor and poisoning attacks

**Relevant results.** `run_backdoor.py` implements a dirty-label backdoor whose
trigger is projected into the same realizable space as the evasion attacks, so
the trigger corresponds to traffic an attacker could actually send. Poisoning
0.1% of the training set (83 flows) yields 92.5% attack success on XGBoost with
no measurable change in clean F1 — the backdoor is invisible to ordinary
validation. The spectral-signature defense (Tran et al., NeurIPS 2018) shows an
inverse relationship: it removes 5% of the poison at a 0.1% rate but 97% at a
2% rate, and even after removing 97% the surviving flows sustain 78.6% ASR.

The cross-dataset run produced the result I find most interesting. On TON_IoT
the backdoor takes hold in RF (0.92 ASR) and XGBoost (0.97) but not in the MLP
(0.003). Because spectral signatures are computed from MLP activations, the
defense removes 0% of the poison there: it inspects the one model that was
never compromised. A defense that reads a single model's internals cannot
protect a heterogeneous ensemble.

**Where I would want to take it.** Clean-label backdoors, which are stealthier
and which I did not evaluate; triggers in the physical/sensor domain rather
than in network features; and defenses that do not assume the defender knows
the poison rate (mine assumes it, removing 1.5x the rate).

## Robust training and the evaluation of robustness

**Relevant results.** Two rounds of offline adversarial retraining did not
improve robustness (0.466 against 0.517 robust recall at epsilon=0.1).
Per-batch Madry-style training, with constraints projected at every PGD step,
raised it to 0.789 +/- 0.022 at epsilon=0.1 and 0.787 at epsilon=0.2, at a
clean-F1 cost of 0.832 -> 0.826, compared at an equal false-positive rate of
0.132 calibrated on validation. A transfer attack from the baseline is weaker
against the robust model than its own white-box attack (0.880 against 0.789),
the ordering expected if the robustness is real rather than gradient masking.

I report the failed offline variant because the gap between the two is the
useful finding: where the constraint projection sits in the training loop
determines whether constraint-aware adversarial training works at all.
Robustness was measured with 50-step PGD and no random restarts; stronger
adaptive attacks would likely lower these numbers.

## Cyber-physical and sensor-domain security

**Not yet covered here.** This repository is network-feature only. The parts
that carry over to a physical-sensor setting are the constraint projection and
realizability audit (sensor signals have tighter physical constraints than
network features, so the method should apply more cleanly, not less) and the
realizable-trigger backdoor construction. Extending both to CPS sensor streams,
evaluated with PA%K and event-wise metrics rather than point-adjust F1, is the
subject of a separate repository in progress.

---

## Reading guide

| Interest | Start here |
|---|---|
| Constrained attacks, realizability | `src/constraints.py`, `results/evasion.csv` |
| Adversarial-example detection | `src/run_detectors.py`, `results/detectors.csv` |
| Backdoors and poisoning defenses | `src/run_backdoor.py`, `results/backdoor.csv` |
| Robust training and its evaluation | `src/run_defense_madry.py`, `results/defense_madry.csv` |
| Does any of it generalize? | `results/toniot/RESULTS.md` |
| What I think is wrong with this work | `README.md`, Limitations |
