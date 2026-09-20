"""Threat-model constraints and realizability audit for flow features.

project(): maps an adversarial example back into the realizable space
           (immutable features restored, domain range, increase-only, relational).
audit():   per-example flag: does this example satisfy every constraint?
           Applied to UNconstrained attacks it measures how many "successful"
           evasions could never exist as real traffic.
"""
import numpy as np
from config import MODIFIABLE, INCREASE_ONLY, RELATIONAL_GE


class Constraints:
    def __init__(self, feature_names, scaler, Xtr_scaled):
        self.names = list(feature_names)
        self.scaler = scaler
        self.i = {n: k for k, n in enumerate(self.names)}
        missing = [n for n in MODIFIABLE if n not in self.i]
        if missing:
            print("WARNING: modifiable features not in data:", missing)
        self.mod = np.array([n in MODIFIABLE for n in self.names])
        self.inc = np.array([n in INCREASE_ONLY for n in self.names])
        self.mask = self.mod.astype(np.float32)
        raw = self.to_raw(Xtr_scaled)
        self.lo, self.hi = raw.min(0), raw.max(0)
        self.scale = np.maximum(self.hi - self.lo, 1e-9)
        self.has_rel = all(k in self.i for k in ("Min", "Max", "AVG", "Std"))
        # generic "a >= b" pairs from config (e.g. src_ip_bytes >= src_bytes)
        self.ge = [(self.i[a], self.i[b]) for a, b in RELATIONAL_GE if a in self.i and b in self.i]

    def to_raw(self, X):
        return self.scaler.inverse_transform(np.asarray(X, dtype=np.float64))

    def to_scaled(self, R):
        return np.clip(self.scaler.transform(R), 0, 1).astype(np.float32)

    def project(self, Xadv, Xorig):
        a, o = self.to_raw(Xadv), self.to_raw(Xorig)
        a[:, ~self.mod] = o[:, ~self.mod]                                   # immutable
        a[:, self.mod] = np.clip(a[:, self.mod], np.maximum(self.lo[self.mod], 0), self.hi[self.mod])
        a[:, self.inc] = np.maximum(a[:, self.inc], o[:, self.inc])         # increase-only
        if self.has_rel:                                                    # Min<=AVG<=Max, Std<=(Max-Min)/2
            mn, mx, av, sd = (self.i[k] for k in ("Min", "Max", "AVG", "Std"))
            a[:, mx] = np.maximum(a[:, mx], a[:, mn])
            a[:, av] = np.clip(a[:, av], np.minimum(a[:, mn], o[:, av]), np.maximum(a[:, mx], o[:, av]))
            a[:, sd] = np.minimum(a[:, sd], np.maximum((a[:, mx] - a[:, mn]) / 2, o[:, sd]))
        for ia, ib in self.ge:
            a[:, ia] = np.maximum(a[:, ia], a[:, ib])
        return self.to_scaled(a)

    def audit(self, Xadv, Xorig, tol=1e-3):
        a, o = self.to_raw(Xadv), self.to_raw(Xorig)
        rel = (a - o) / self.scale
        checks = {
            "immutable_unchanged": (np.abs(rel[:, ~self.mod]) <= tol).all(1),
            "increase_only": (rel[:, self.inc] >= -tol).all(1),
            "in_domain": (((a - self.lo) / self.scale >= -tol) & ((self.hi - a) / self.scale >= -tol)).all(1),
        }
        if self.has_rel:
            mn, mx, av, sd = (self.i[k] for k in ("Min", "Max", "AVG", "Std"))

            def viol(z):
                t = tol * self.scale[av]
                return ((z[:, av] < z[:, mn] - t) | (z[:, av] > z[:, mx] + t) |
                        (z[:, sd] > (z[:, mx] - z[:, mn]) / 2 + tol * self.scale[sd]))
            checks["relational"] = ~(viol(a) & ~viol(o))    # only NEW violations count
        if self.ge:
            new_viol = np.zeros(len(a), bool)
            for ia, ib in self.ge:
                t = tol * self.scale[ia]
                new_viol |= (a[:, ia] < a[:, ib] - t) & ~(o[:, ia] < o[:, ib] - t)
            checks["relational_ge"] = ~new_viol
        ok = np.all(np.stack(list(checks.values())), axis=0)
        return ok, {k: float(v.mean()) for k, v in checks.items()}
