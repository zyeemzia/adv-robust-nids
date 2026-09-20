# Working rules and plan

## Machine
Ubuntu, Intel i7-6500U (2 cores), 8 GB RAM, no GPU. Use n_jobs <= 2, CPU-only PyTorch.

## Rules
1. Never invent results, numbers, dataset facts, or citations. If unsure of a dataset's structure, inspect it first.
2. Complete runnable code, one file at a time, with its path.
3. After each file: exact command + expected output (describe format when numbers depend on data).
4. Seeds: 42, 123, 456, 789, 1024. Splits saved to `splits/` for reproducibility.
5. Memory under 7 GB: chunked reading, float32, subsampling.
6. Explain each design choice in 2-3 lines (interview-defensible).
7. Wait for real pasted outputs before the next step.

## Portfolio plan (Fall 2027 PhD applications)
Build order: 2 -> 3 -> 4 -> 1.
- **Project 2 (flagship, current):** Constraint-aware adversarial-robust NIDS. CIC-IoT2023 + TON_IoT. RF/XGBoost/MLP, IBM ART evasion (PGD, HopSkipJump) with constraint layer + realizability audit, label-flip/backdoor poisoning, adversarial training, feature squeezing, adversarial-example detector (vs. Manda). Targets: Ning Wang (USF), Tao Hou & Tao Wang (UNT), Ram Krishnan (UTSA).
- **Project 3:** Federated NIDS (Flower) under non-IID + model poisoning. Edge-IIoTset primary. FedAvg/FedProx/SCAFFOLD/FedNova, Krum/trimmed-mean defense. Target: Wang.
- **Project 4:** CPS sensor-attack detection (SWaT/WADI via iTrust request, HAI, ArduPilot SITL), PA%K/event-wise metrics (not point-adjust), adversarial + backdoor extension. Targets: Xiali Hei, Hyungsub Kim, Liqun Shan.
- **Project 1:** RL-adaptive honeypot (Cowrie + SB3/QR-DQN, CybORG comparison), modernizing MSc thesis.

## Pending actions
- File SUTD iTrust SWaT/WADI request (Week 1).
- Download ~10 CIC-IoT2023 CSV files into `data/raw/`.
