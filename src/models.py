"""Three detector families: bagged trees (RF), boosted trees (XGB), neural (MLP).
Trees are the strong tabular baselines; the MLP is differentiable, so it is the
white-box target for gradient attacks and the model we adversarially train."""
import random
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from config import N_JOBS

torch.set_num_threads(N_JOBS)


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


class MLP(nn.Module):
    def __init__(self, d, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))

    def forward(self, x):
        return self.net(x)


def train_mlp(X, y, seed, epochs=15, bs=512, lr=1e-3, verbose=False):
    set_seed(seed)
    m = MLP(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    for ep in range(epochs):
        m.train()
        perm, tot = torch.randperm(len(Xt)), 0.0
        for i in range(0, len(Xt), bs):
            j = perm[i:i + bs]
            opt.zero_grad()
            loss = loss_fn(m(Xt[j]), yt[j])
            loss.backward(); opt.step()
            tot += loss.item() * len(j)
        if verbose:
            print(f"    MLP epoch {ep + 1}/{epochs} loss={tot / len(Xt):.4f}")
    m.eval()
    return m


def mlp_proba(m, X):
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(X, dtype=torch.float32)), 1)[:, 1].numpy()


def mlp_predict(m, X):
    return (mlp_proba(m, X) >= 0.5).astype(int)


def train_rf(X, y, seed):
    return RandomForestClassifier(n_estimators=100, n_jobs=N_JOBS, random_state=seed).fit(X, y)


def train_xgb(X, y, seed):
    return XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.1, tree_method="hist",
                         n_jobs=N_JOBS, random_state=seed, eval_metric="logloss").fit(X, y)
