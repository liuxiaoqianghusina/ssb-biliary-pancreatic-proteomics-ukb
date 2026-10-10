"""Substitution analysis repeated in participants with no major dietary change, stable weight and no diabetes
(the reverse-causation-restricted sample of S12). Output: results/substitution_restricted.csv"""
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from common import CACHE, RES, base_covariates, cox, follow_up_end, outcome_defs

warnings.filterwarnings("ignore")
d = pd.read_pickle(CACHE / "analysis.pkl").merge(pd.read_pickle(CACHE / "diet_change.pkl"), on="eid", how="left")
A = d[d.n_valid_recalls >= 1].copy()
A["start"] = A["first_recall_date"].fillna(A["baseline_date"])
A["age_start"] = A["age"] + (A["start"] - A["baseline_date"]).dt.days / 365.25
A = A[follow_up_end(A) > A["start"]]
A = A[~(A["Any_cancer"].notna() & (A["Any_cancer"] <= A["start"]))].reset_index(drop=True)
for b, cap in [("SSB", 6), ("ASB", 6), ("juice", 6), ("water", 6), ("tea", 10), ("coffee", 10)]:
    A[b] = A[b].clip(upper=cap)
mask = A["diet_change"].eq(0) & pd.to_numeric(A["p2306_i0"], errors="coerce").eq(0) & A["diabetes_sr"].fillna(0).eq(0)
Z = base_covariates(A)
X = A[["SSB", "ASB", "water", "tea", "coffee", "juice"]]
rows = []
for o in ["Gallstones", "AP", "AP_nonbiliary", "CP"]:
    elig, ev, T = outcome_defs(A, [o], A["start"])[o]
    _, m = cox(X, Z, elig & mask, ev, T, return_model=True)
    b, V = m.params_, m.variance_matrix_
    for src, alts in [("SSB", ["ASB", "water", "tea", "coffee"]), ("ASB", ["water", "tea", "coffee"])]:
        for alt in alts:
            est = b[alt] - b[src]
            se = np.sqrt(V.at[alt, alt] + V.at[src, src] - 2 * V.at[alt, src])
            rows.append({"outcome": o, "replace": src, "with": alt, "hr": np.exp(est), "lci": np.exp(est - 1.96 * se),
                         "uci": np.exp(est + 1.96 * se), "p": 2 * stats.norm.sf(abs(est / se)), "events": int(ev[elig & mask].sum())})
r = pd.DataFrame(rows)
r.to_csv(RES / "substitution_restricted.csv", index=False)
r["est"] = r.apply(lambda x: f"{x.hr:.2f} ({x.lci:.2f}-{x.uci:.2f})" + ("*" if x.p < 0.05 else ""), axis=1)
print(r.pivot_table(index=["replace", "with"], columns="outcome", values="est", aggfunc="first").to_string())
