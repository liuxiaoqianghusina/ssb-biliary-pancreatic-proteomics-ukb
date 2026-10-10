"""Proteomic signatures of SSB/ASB and incident biliary-pancreatic disease in the independent Olink sample.

Sample: QC-passed, randomly selected Olink participants outside the signature training set (incl. those without
dietary data), no cancer at baseline; time zero = baseline visit. Signatures standardised within this sample.
  M1 = age, sex, ethnicity, Townsend, education, smoking, alcohol frequency, physical activity, BMI, diabetes,
       fasting time, centre
  M2 = M1 + triglycerides, HbA1c, CRP, GGT, cystatin C, creatinine
  M3 = M2 + BMI proteomic signature
Outputs (results/): flow_signature.csv, signature_main.csv, signature_tertiles.csv, signature_sensitivity.csv,
                    signature_protein_cox.csv
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from common import (CACHE, DIETPROJ, NEGATIVE, POSITIVE, PRIMARY, RES, SECONDARY, SENS_OUT, add_biomarkers, add_kidney,
                    base_covariates, cox, extended, follow_up_end, outcome_defs)

warnings.filterwarnings("ignore")
SIGS = ["sig_SSB", "sig_ASB", "rsig_SSB", "rsig_ASB"]
sc = pd.read_pickle(CACHE / "signatures.pkl")
B = pd.read_pickle(CACHE / "analysis.pkl").merge(sc, on="eid")
flow = {"QC-passed Olink participants": len(B)}
B = B[B["consortium"] == 0]
flow["Randomly selected (non-consortium)"] = len(B)
B = B[~B["train"]]
flow["Outside the signature training set"] = len(B)
B = B[follow_up_end(B) > B["baseline_date"]]
B = B[~(B["Any_cancer"].notna() & (B["Any_cancer"] <= B["baseline_date"]))].reset_index(drop=True)
flow["No cancer at baseline (analytic sample)"] = len(B)
flow["... of whom with >=1 valid 24h recall (test set)"] = int(B["test"].sum())
pd.Series(flow).to_csv(RES / "flow_signature.csv", header=["n"])
print(pd.Series(flow).to_string(), flush=True)
for s in SIGS + ["sig_BMI"]:
    B[s] = (B[s] - B[s].mean()) / B[s].std()

M1 = base_covariates(B, diet=False)
M2 = add_kidney(add_biomarkers(M1, B), B)
M3 = M2.assign(sig_BMI=B["sig_BMI"].to_numpy())
M1e = M1.join(extended(B))
M2u = M2.assign(urate=B["urate"].fillna(B["urate"].median()).to_numpy(), ppi=B["ppi"].fillna(0).to_numpy())
prev_gs = (B["Gallstones"].notna() & (B["Gallstones"] <= B["baseline_date"])) | B["sr_gallstones"].fillna(0).eq(1)

OUTS = PRIMARY + SECONDARY + SENS_OUT + POSITIVE + NEGATIVE
FULL = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "Cholecystitis", "CP", "Pancreatitis_any"]
start = B["baseline_date"]
DEFS = outcome_defs(B, OUTS, start)
DEFS_LAG = outcome_defs(B, FULL, start, lag_years=2)
DEFS_CGS = outcome_defs(B, ["AP_nonbiliary", "CP"], start, censor_at="Gallstones")
for s in ["sig_SSB", "sig_ASB"]:
    t = pd.qcut(B[s], 3, labels=["T1", "T2", "T3"])
    B[f"{s}_T"] = t
    B[f"{s}_Tmed"] = t.map(B.groupby(t, observed=True)[s].median()).astype(float)


def rows_for(name, analysis, model, r, terms):
    return [{"outcome": name, "analysis": analysis, "model": model, "signature": t, "n": r["n"], "events": r["events"],
             **{k: r[t][k] for k in ("hr", "lci", "uci", "p")}} for t in terms]


def run(name):
    out = {"main": [], "tert": [], "sens": []}
    elig, ev, T = DEFS[name]
    for mname, Z in [("M1", M1), ("M2", M2), ("M3", M3)]:
        for s in SIGS:
            out["main"] += rows_for(name, "per SD", mname, cox(B[[s]], Z, elig, ev, T), [s])
        out["main"] += rows_for(name, "per SD, mutually adjusted", mname,
                                cox(B[["sig_SSB", "sig_ASB"]], Z, elig, ev, T), ["sig_SSB", "sig_ASB"])
    for s in ["sig_SSB", "sig_ASB"]:
        D = pd.get_dummies(B[f"{s}_T"], drop_first=True, dtype=float).add_prefix(f"{s}_")
        for mname, Z in [("M1", M1), ("M2", M2)]:
            r = cox(D, Z, elig, ev, T)
            rt = cox(B[[f"{s}_Tmed"]], Z, elig, ev, T)
            for c in D.columns:
                out["tert"].append({"outcome": name, "model": mname, "signature": s, "tertile": c.split("_")[-1],
                                    "n": r["n"], "events": r["events"], **{k: r[c][k] for k in ("hr", "lci", "uci", "p")},
                                    "p_trend": rt[f"{s}_Tmed"]["p"]})
    if name not in FULL:
        return out
    S = out["sens"]
    two = ["sig_SSB", "sig_ASB"]
    for s in two:
        S += rows_for(name, "Main (M2)", "M2", cox(B[[s]], M2, elig, ev, T), [s])
        el, e_l, T_l = DEFS_LAG[name]
        S += rows_for(name, "Excluding the first 2 years of follow-up", "M2", cox(B[[s]], M2, el, e_l, T_l), [s])
        S += rows_for(name, "Excluding baseline diabetes", "M2",
                      cox(B[[s]], M2, elig & B["diabetes_sr"].fillna(0).eq(0), ev, T), [s])
        S += rows_for(name, "+ gallstone-related confounders", "M1+", cox(B[[s]], M1e, elig, ev, T), [s])
        S += rows_for(name, "+ urate and PPI use", "M2+", cox(B[[s]], M2u, elig, ev, T), [s])
        S += rows_for(name, "Participants without dietary data only", "M2",
                      cox(B[[s]], M2, elig & ~B["test"], ev, T), [s])
        if name not in ("Gallstones", "Cholecystitis"):
            S += rows_for(name, "Excluding prevalent gallstones", "M2", cox(B[[s]], M2, elig & ~prev_gs, ev, T), [s])
        if name in DEFS_CGS:
            eg, e_g, T_g = DEFS_CGS[name]
            S += rows_for(name, "Censored at incident gallstones", "M2", cox(B[[s]], M2, eg, e_g, T_g), [s])
    return out


res = Parallel(n_jobs=10)(delayed(run)(o) for o in OUTS)
for k, f in {"main": "signature_main", "tert": "signature_tertiles", "sens": "signature_sensitivity"}.items():
    pd.DataFrame([r for o in res for r in o[k]]).to_csv(RES / f"{f}.csv", index=False)

# ---------------------------------------------------------------- individual signature proteins
P = pd.read_pickle(DIETPROJ / "cache" / "olink_qc_plateadj.pkl")
prots = sorted(set(pd.read_csv(RES / "signature_weights_sig_SSB.csv", index_col=0).index)
               | set(pd.read_csv(RES / "signature_weights_sig_ASB.csv", index_col=0).index))
Pm = P.loc[B["eid"], prots].reset_index(drop=True)
PO = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "CP"]


def prot_cox(pr):
    rows = []
    for name in PO:
        r = cox(Pm[[pr]], M1, *DEFS[name])
        rows.append({"protein": pr, "outcome": name, "events": r["events"], **{k: r[pr][k] for k in ("hr", "lci", "uci", "p")}})
    return rows


pc = pd.DataFrame([r for rr in Parallel(n_jobs=10)(delayed(prot_cox)(p) for p in prots) for r in rr])
pc["fdr"] = pc.groupby("outcome")["p"].transform(lambda p: multipletests(p, method="fdr_bh")[1])
w = {k: pd.read_csv(RES / f"signature_weights_sig_{k}.csv", index_col=0)["weight"] for k in ("SSB", "ASB")}
pc["weight_SSB"], pc["weight_ASB"] = pc["protein"].map(w["SSB"]), pc["protein"].map(w["ASB"])
pc.to_csv(RES / "signature_protein_cox.csv", index=False)

# ---------------------------------------------------------------- print
pd.set_option("display.width", 250)
m = pd.read_csv(RES / "signature_main.csv")
m["est"] = m["hr"].map("{:.2f}".format) + " (" + m["lci"].map("{:.2f}".format) + "-" + m["uci"].map("{:.2f}".format) + ")" + \
    np.where(m["p"] < 0.05, "*", "")
print(m[m.analysis == "per SD"].pivot_table(index="outcome", columns=["signature", "model"], values="est", aggfunc="first")
      .reindex(OUTS).to_string())
print(m[m.analysis != "per SD"].pivot_table(index="outcome", columns=["signature", "model"], values="est", aggfunc="first")
      .reindex(OUTS).to_string())
print(pc[pc.fdr < 0.05].groupby("outcome").size().to_string())
