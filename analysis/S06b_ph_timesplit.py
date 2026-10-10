"""Time-split HRs for the exposure terms with a proportional-hazards P <0.05 (S06_ph.py):
events in the first 6 years (follow-up censored at 6 years) and after 6 years (participants event-free and under
follow-up at 6 years, time restarted at 6 years). Output: results/ph_timesplit.csv"""
import warnings

import numpy as np
import pandas as pd

from common import (BEVS, CACHE, RES, add_biomarkers, add_kidney, base_covariates, cox, follow_up_end, outcome_defs)

warnings.filterwarnings("ignore")
CUT = 6.0
d = pd.read_pickle(CACHE / "analysis.pkl")
ph = pd.read_csv(RES / "ph_test.csv")
viol = ph[ph.p_PH < 0.05]
print(viol.to_string(index=False))


def split(X, Z, elig, ev, T, term):
    rows = []
    e1 = ev.where(T <= CUT, 0)
    r1 = cox(X, Z, elig, e1, T.clip(upper=CUT), [term])
    late = elig & (T > CUT)
    r2 = cox(X, Z, late, ev, T - CUT, [term])
    for lab, r in [(f"0–{CUT:g} years", r1), (f">{CUT:g} years", r2)]:
        rows.append({"period": lab, "events": r["events"], **{k: r[term][k] for k in ("hr", "lci", "uci", "p")}})
    return rows


out = []
A = d[d.n_valid_recalls >= 1].copy()
A["start"] = A["first_recall_date"].fillna(A["baseline_date"])
A["age_start"] = A["age"] + (A["start"] - A["baseline_date"]).dt.days / 365.25
A = A[follow_up_end(A) > A["start"]]
A = A[~(A["Any_cancer"].notna() & (A["Any_cancer"] <= A["start"]))].reset_index(drop=True)
for b in BEVS:
    A[b] = A[b].clip(upper=6)
sc = pd.read_pickle(CACHE / "signatures.pkl")
B = d.merge(sc, on="eid")
B = B[(B.consortium == 0) & ~B["train"]]
B = B[follow_up_end(B) > B["baseline_date"]]
B = B[~(B["Any_cancer"].notna() & (B["Any_cancer"] <= B["baseline_date"]))].reset_index(drop=True)
for s in ["sig_SSB", "sig_ASB"]:
    B[s] = (B[s] - B[s].mean()) / B[s].std()
for _, v in viol.iterrows():
    if v.analysis.startswith("intake"):
        elig, ev, T = outcome_defs(A, [v.outcome], A["start"])[v.outcome]
        rows = split(A[BEVS], base_covariates(A), elig, ev, T, v.term)
    else:
        elig, ev, T = outcome_defs(B, [v.outcome], B["baseline_date"])[v.outcome]
        Z = add_kidney(add_biomarkers(base_covariates(B, diet=False), B), B)
        rows = split(B[[v.term]], Z, elig, ev, T, v.term)
    out += [{"analysis": v.analysis, "outcome": v.outcome, "term": v.term, "p_PH": v.p_PH, **r} for r in rows]
out = pd.DataFrame(out)
out.to_csv(RES / "ph_timesplit.csv", index=False)
print(out.round(3).to_string(index=False))
