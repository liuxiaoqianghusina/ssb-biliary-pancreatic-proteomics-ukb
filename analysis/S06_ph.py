"""Proportional-hazards checks (scaled Schoenfeld residual test, rank-transformed time) for the exposure terms of the
main intake models (Model 2) and signature models (Model 2). Output: results/ph_test.csv"""
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.statistics import proportional_hazard_test

from common import (BEVS, CACHE, RES, add_biomarkers, add_kidney, base_covariates, follow_up_end, outcome_defs)

warnings.filterwarnings("ignore")
OUTS = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "CP"]
d = pd.read_pickle(CACHE / "analysis.pkl")


def ph(X, Z, elig, ev, T, terms):
    f = pd.concat([X.reset_index(drop=True), Z.reset_index(drop=True)], axis=1)
    f["T"], f["E"] = np.asarray(T), np.asarray(ev)
    f = f[np.asarray(elig)]
    f = f.loc[:, (f.std() > 0) | f.columns.isin(["T", "E"])]
    m = CoxPHFitter(penalizer=1e-4).fit(f, "T", "E")
    r = proportional_hazard_test(m, f, time_transform="rank").summary
    return {t: r.loc[t, "p"] for t in terms}


rows = []
A = d[d.n_valid_recalls >= 1].copy()
A["start"] = A["first_recall_date"].fillna(A["baseline_date"])
A["age_start"] = A["age"] + (A["start"] - A["baseline_date"]).dt.days / 365.25
A = A[follow_up_end(A) > A["start"]]
A = A[~(A["Any_cancer"].notna() & (A["Any_cancer"] <= A["start"]))].reset_index(drop=True)
for b in BEVS:
    A[b] = A[b].clip(upper=6)
Z = base_covariates(A)
defs = outcome_defs(A, OUTS, A["start"])
for o in OUTS:
    p = ph(A[BEVS], Z, *defs[o], BEVS)
    rows += [{"analysis": "intake per serving (Model 2)", "outcome": o, "term": k, "p_PH": v} for k, v in p.items()]
    print(rows[-2:], flush=True)

sc = pd.read_pickle(CACHE / "signatures.pkl")
B = d.merge(sc, on="eid")
B = B[(B.consortium == 0) & ~B["train"]]
B = B[follow_up_end(B) > B["baseline_date"]]
B = B[~(B["Any_cancer"].notna() & (B["Any_cancer"] <= B["baseline_date"]))].reset_index(drop=True)
for s in ["sig_SSB", "sig_ASB"]:
    B[s] = (B[s] - B[s].mean()) / B[s].std()
ZB = add_kidney(add_biomarkers(base_covariates(B, diet=False), B), B)
defs = outcome_defs(B, OUTS, B["baseline_date"])
for o in OUTS:
    for s in ["sig_SSB", "sig_ASB"]:
        p = ph(B[[s]], ZB, *defs[o], [s])
        rows += [{"analysis": "signature per SD (Model 2)", "outcome": o, "term": k, "p_PH": v} for k, v in p.items()]
pd.DataFrame(rows).to_csv(RES / "ph_test.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
