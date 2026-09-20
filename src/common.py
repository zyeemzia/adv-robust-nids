"""Shared helpers: per-seed data loading, model I/O, ART wrapper, metrics."""
import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from art.estimators.classification import PyTorchClassifier
from config import PROC_DIR, SPLIT_DIR, MODEL_DIR
from models import MLP


def load_seed(seed):
    z = np.load(PROC_DIR / "dataset.npz", allow_pickle=True)
    X, y = z["X"], z["y"].astype(int)
    sp = np.load(SPLIT_DIR / f"seed_{seed}.npz")
    sc = joblib.load(PROC_DIR / f"scaler_{seed}.joblib")
    f = lambda A: np.clip(sc.transform(A), 0, 1).astype(np.float32)
    D = dict(feature_names=[str(n) for n in z["feature_names"]], scaler=sc)
    for k, part in (("tr", "train"), ("va", "val"), ("te", "test")):
        D["X" + k], D["y" + k] = f(X[sp[part]]), y[sp[part]]
    return D


def save_models(seed, mlp, rf, xgb):
    torch.save(mlp.state_dict(), MODEL_DIR / f"mlp_{seed}.pt")
    joblib.dump(rf, MODEL_DIR / f"rf_{seed}.joblib")
    joblib.dump(xgb, MODEL_DIR / f"xgb_{seed}.joblib")


def load_models(seed, d):
    mlp = MLP(d)
    mlp.load_state_dict(torch.load(MODEL_DIR / f"mlp_{seed}.pt"))
    mlp.eval()
    return mlp, joblib.load(MODEL_DIR / f"rf_{seed}.joblib"), joblib.load(MODEL_DIR / f"xgb_{seed}.joblib")


def art_mlp(m, d):
    return PyTorchClassifier(model=m, loss=nn.CrossEntropyLoss(), input_shape=(d,),
                             nb_classes=2, clip_values=(0.0, 1.0))


def metrics(y, pred, score=None):
    out = dict(accuracy=accuracy_score(y, pred),
               precision=precision_score(y, pred, zero_division=0),
               recall=recall_score(y, pred, zero_division=0),
               f1=f1_score(y, pred, zero_division=0))
    if score is not None:
        out["auc"] = roc_auc_score(y, score)
    return out
