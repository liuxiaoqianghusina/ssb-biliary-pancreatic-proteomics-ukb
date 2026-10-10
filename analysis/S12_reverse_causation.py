"""Reverse-causation analyses (main concern for ASB: people with obesity or metabolic disease switching to diet drinks).

Restrictions (baseline touchscreen): no major dietary change in the past 5 years (field 1538 = 0), stable weight in the
past year (field 2306 = 0), no diabetes, BMI <30 kg/m2, applied cumulatively.
Habitual consumption (participants with >=2 valid recalls): for each beverage, never (consumed in no recall),
occasional (some recalls) or habitual (every recall); both beverages in the same Model 2.
Signature analyses (independent sample, Model 2) use the same baseline restrictions.
Outputs: results/reverse_causation_intake.csv, results/reverse_causation_signature.csv, cache/recall_consistency.pkl
"""
import warnings

import numpy as np
import pandas as pd

from common import (BEVS, CACHE, PILOT, RES, UKB, add_biomarkers, add_kidney, base_covariates, cox, follow_up_end,
                    outcome_defs)

warnings.filterwarnings("ignore")
OUTS = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "CP", "Pancreatitis_any"]


def servings(s):
    s = pd.to_numeric(s, errors="coerce")
    return s.replace({555: 0.5, 444: 0.25, 111: 0.5, 200: 2, 300: 3, 400: 4, 500: 5, 600: 6}).where(lambda x: x >= 0)


# ---------------------------------------------------------------- extra fields
if not (CACHE / "diet_change.pkl").exists():
    x = pd.concat(pd.read_csv(UKB / "UKB619657_assessment_center_2025Nov.csv", usecols=["eid", "p1538_i0"], dtype=str,
                              chunksize=100000), ignore_index=True)
    x["eid"] = x["eid"].astype(int)
    x["diet_change"] = pd.to_numeric(x["p1538_i0"], errors="coerce")
    x[["eid", "diet_change"]].to_pickle(CACHE / "diet_change.pkl")
if not (CACHE / "recall_consistency.pkl").exists():
    diet = pd.read_pickle(PILOT / "diet_raw.pkl")
    diet = diet[["eid"] + [f"p26002_i{i}" for i in range(5)] + [f"p100026_i{i}" for i in range(5)]]
    sex = pd.read_pickle(CACHE / "analysis.pkl")[["eid", "sex"]]
    diet = diet.merge(sex, on="eid", how="left")
    items = pd.read_pickle(PILOT / "items_raw.pkl").set_index("eid").reindex(diet["eid"]).reset_index()
    n_valid = np.zeros(len(diet)); n_ssb = np.zeros(len(diet)); n_asb = np.zeros(len(diet))
    for i in range(5):
        e = diet[f"p26002_i{i}"]
        kcal = e / 4.184
        ok = (e.notna() & diet[f"p100026_i{i}"].isna() & (kcal >= 500) & (kcal <= np.where(diet["sex"] == 1, 4200, 3500))).to_numpy()
        ssb = (servings(items[f"p100170_i{i}"]).fillna(0) + servings(items[f"p100180_i{i}"]).fillna(0)).to_numpy() > 0
        asb = servings(items[f"p100160_i{i}"]).fillna(0).to_numpy() > 0
        n_valid += ok; n_ssb += ok & ssb; n_asb += ok & asb
    pd.DataFrame({"eid": diet["eid"], "n_valid": n_valid, "n_ssb": n_ssb, "n_asb": n_asb}).to_pickle(CACHE / "recall_consistency.pkl")
    del diet, items

d = (pd.read_pickle(CACHE / "analysis.pkl").merge(pd.read_pickle(CACHE / "diet_change.pkl"), on="eid", how="left")
     .merge(pd.read_pickle(CACHE / "recall_consistency.pkl"), on="eid", how="left"))
d["weight_change"] = pd.to_numeric(d["p2306_i0"], errors="coerce")


def restrictions(df):
    r1 = df["diet_change"].eq(0)
    r2 = r1 & df["weight_change"].eq(0)
    r3 = r2 & df["diabetes_sr"].fillna(0).eq(0)
    r4 = r3 & (df["bmi"] < 30)
    return [("All participants (main)", pd.Series(True, index=df.index)),
            ("No major dietary change in past 5 years", r1),
            ("+ stable weight in past year", r2),
            ("+ no diabetes", r3),
            ("+ BMI <30 kg/m2", r4)]


# ---------------------------------------------------------------- intake
A = d[d.n_valid_recalls >= 1].copy()
A["start"] = A["first_recall_date"].fillna(A["baseline_date"])
A["age_start"] = A["age"] + (A["start"] - A["baseline_date"]).dt.days / 365.25
A = A[follow_up_end(A) > A["start"]]
A = A[~(A["Any_cancer"].notna() & (A["Any_cancer"] <= A["start"]))].reset_index(drop=True)
for b in BEVS:
    A[b] = A[b].clip(upper=6)
Z = base_covariates(A)
defs = outcome_defs(A, OUTS, A["start"])
rows = []
for lab, mask in restrictions(A):
    print(lab, int(mask.sum()), flush=True)
    for o in OUTS:
        elig, ev, T = defs[o]
        r = cox(A[BEVS], Z, elig & mask, ev, T)
        rows.append({"analysis": "per serving/day", "restriction": lab, "outcome": o, "n": r["n"], "events": r["events"],
                     **{f"{b}_{k}": r[b][k] for b in BEVS for k in ("hr", "lci", "uci", "p")}})
# habitual consumption among participants with >=2 valid recalls
H = A[A["n_valid"] >= 2].copy()
for b, col in [("SSB", "n_ssb"), ("ASB", "n_asb")]:
    H[f"{b}_pattern"] = np.where(H[col] == 0, "never", np.where(H[col] == H["n_valid"], "habitual", "occasional"))
XH = pd.get_dummies(H[["SSB_pattern", "ASB_pattern"]], dtype=float).drop(columns=["SSB_pattern_never", "ASB_pattern_never"])
ZH = Z.loc[H.index].reset_index(drop=True)
XH = XH.reset_index(drop=True)
Hr = H.reset_index(drop=True)
defsH = outcome_defs(Hr, OUTS, Hr["start"])
for lab, mask in restrictions(Hr)[:1] + restrictions(Hr)[2:3]:
    for o in OUTS:
        elig, ev, T = defsH[o]
        r = cox(XH, ZH, elig & mask, ev, T)
        for b in BEVS:
            for lv in ("occasional", "habitual"):
                k = f"{b}_pattern_{lv}"
                rows.append({"analysis": f"consumption pattern, >=2 recalls ({lv} vs never)", "restriction": lab,
                             "outcome": o, "n": r["n"], "events": r["events"], "exposure": b,
                             **{f"{b}_{kk}": r[k][kk] for kk in ("hr", "lci", "uci", "p")}})
pat_n = {b: Hr[f"{b}_pattern"].value_counts().to_dict() for b in BEVS}
print("pattern counts:", pat_n, flush=True)
ri = pd.DataFrame(rows)
ri.to_csv(RES / "reverse_causation_intake.csv", index=False)

# ---------------------------------------------------------------- signatures
sc = pd.read_pickle(CACHE / "signatures.pkl")
B = d.merge(sc, on="eid")
B = B[(B.consortium == 0) & ~B["train"]]
B = B[follow_up_end(B) > B["baseline_date"]]
B = B[~(B["Any_cancer"].notna() & (B["Any_cancer"] <= B["baseline_date"]))].reset_index(drop=True)
for s in ["sig_SSB", "sig_ASB"]:
    B[s] = (B[s] - B[s].mean()) / B[s].std()
ZB = add_kidney(add_biomarkers(base_covariates(B, diet=False), B), B)
defsB = outcome_defs(B, OUTS, B["baseline_date"])
rows = []
for lab, mask in restrictions(B):
    for o in OUTS:
        elig, ev, T = defsB[o]
        for s in ["sig_SSB", "sig_ASB"]:
            r = cox(B[[s]], ZB, elig & mask, ev, T)
            rows.append({"restriction": lab, "outcome": o, "signature": s, "n": r["n"], "events": r["events"],
                         **{k: r[s][k] for k in ("hr", "lci", "uci", "p")}})
rs = pd.DataFrame(rows)
rs.to_csv(RES / "reverse_causation_signature.csv", index=False)

# ---------------------------------------------------------------- print
pd.set_option("display.width", 250)
f = lambda h, l, u, p: f"{h:.2f} ({l:.2f}-{u:.2f})" + ("*" if p < 0.05 else "")
x = ri[ri.analysis == "per serving/day"].copy()
for b in BEVS:
    x[b] = [f(*v) for v in x[[f"{b}_hr", f"{b}_lci", f"{b}_uci", f"{b}_p"]].to_numpy()]
print(x.pivot_table(index="outcome", columns="restriction", values=["SSB", "ASB"], aggfunc="first").to_string())
print(x.pivot_table(index="outcome", columns="restriction", values="events", aggfunc="first").to_string())
y = ri[ri.analysis != "per serving/day"].copy()
y["est"] = [f(*v) if not np.isnan(v[0]) else "" for v in
            np.array([[r[f"{r.exposure}_hr"], r[f"{r.exposure}_lci"], r[f"{r.exposure}_uci"], r[f"{r.exposure}_p"]] for _, r in y.iterrows()])]
print(y.pivot_table(index=["outcome", "exposure"], columns=["restriction", "analysis"], values="est", aggfunc="first").to_string())
rs["est"] = [f(*v) for v in rs[["hr", "lci", "uci", "p"]].to_numpy()]
print(rs.pivot_table(index=["outcome", "signature"], columns="restriction", values="est", aggfunc="first").to_string())
print(rs.pivot_table(index="outcome", columns="restriction", values="events", aggfunc="first").to_string())
