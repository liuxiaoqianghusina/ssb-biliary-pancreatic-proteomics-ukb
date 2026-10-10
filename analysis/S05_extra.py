"""Supporting analyses.
(1) SSB/ASB and baseline blood biomarkers (TG, HbA1c, CRP, GGT, ALT, urate, cystatin C): percentage difference
    per serving/day (log-transformed markers; HbA1c in mmol/mol), M2 covariates, mutually adjusted.
(2) E-values for the main intake and signature associations.
(3) Heterogeneity between gallstone-related and non-gallstone acute pancreatitis (z test on log HRs).
(4) Fine-Gray input (death as a competing event) for S05b_finegray.R.
(5) Pathway enrichment (g:Profiler, GO:BP/KEGG/Reactome, background = all 2,919 measured proteins).
Outputs: results/biomarkers.csv, evalues.csv, heterogeneity_ap.csv, enrichment_*.csv; cache/finegray_*.csv
"""
import json
import subprocess
import warnings

import numpy as np
import pandas as pd

from common import (BEVS, CACHE, DIETPROJ, RES, base_covariates, add_biomarkers, add_kidney, follow_up_end, ols_many,
                    outcome_defs)

warnings.filterwarnings("ignore")
d = pd.read_pickle(CACHE / "analysis.pkl")
A = d[d.n_valid_recalls >= 1].copy()
A["start"] = A["first_recall_date"].fillna(A["baseline_date"])
A["age_start"] = A["age"] + (A["start"] - A["baseline_date"]).dt.days / 365.25
A = A[follow_up_end(A) > A["start"]]
A = A[~(A["Any_cancer"].notna() & (A["Any_cancer"] <= A["start"]))].reset_index(drop=True)
for b in BEVS:
    A[b] = A[b].clip(upper=6)

# ---------------------------------------------------------------- (1) biomarkers
Z = base_covariates(A)
Z["fasting_h"] = A["fasting_h"].fillna(A["fasting_h"].median())
MARK = {"tg": ("Triglycerides", True), "hba1c": ("HbA1c", True), "crp": ("C-reactive protein", True),
        "ggt": ("Gamma-glutamyltransferase", True), "alt": ("Alanine aminotransferase", True),
        "urate": ("Urate", True), "cysc": ("Cystatin C", True)}
rows = []
for c, (lab, logt) in MARK.items():
    y = np.log(A[c].clip(lower=1e-3)) if logt else A[c]
    ok = y.notna().to_numpy()
    for b, other in [("SSB", "ASB"), ("ASB", "SSB")]:
        for adj, Zm in [("with BMI", Z), ("without BMI", Z.drop(columns=["bmi"]))]:
            Zb = Zm.assign(other=A[other]).to_numpy()[ok]
            beta, se, p = ols_many(A[b].to_numpy()[ok], Zb, y.to_numpy()[ok, None])
            rows.append({"biomarker": lab, "exposure": b, "adjustment": adj, "n": int(ok.sum()),
                         "pct": 100 * (np.exp(beta[0]) - 1), "lci": 100 * (np.exp(beta[0] - 1.96 * se[0]) - 1),
                         "uci": 100 * (np.exp(beta[0] + 1.96 * se[0]) - 1), "p": p[0]})
bm = pd.DataFrame(rows)
bm.to_csv(RES / "biomarkers.csv", index=False)
print(bm[bm.adjustment == "with BMI"].round(3).to_string(index=False), flush=True)


# ---------------------------------------------------------------- (2) E-values
def evalue(hr):
    hr = hr if hr >= 1 else 1 / hr
    return hr + np.sqrt(hr * (hr - 1))


ev = []
im = pd.read_csv(RES / "intake_main.csv")
im = im[im.analysis == "M2"]
for _, r in im.iterrows():
    for b in BEVS:
        hr, lo, hi = r[f"{b}_hr"], r[f"{b}_lci"], r[f"{b}_uci"]
        bound = lo if hr >= 1 else hi
        ev.append({"source": "intake per serving (M2)", "outcome": r.outcome, "exposure": b, "hr": hr,
                   "e_value": evalue(hr), "e_value_ci": 1.0 if (lo <= 1 <= hi) else evalue(bound)})
sm = pd.read_csv(RES / "signature_main.csv")
sm = sm[(sm.analysis == "per SD") & (sm.model == "M2") & sm.signature.isin(["sig_SSB", "sig_ASB"])]
for _, r in sm.iterrows():
    bound = r.lci if r.hr >= 1 else r.uci
    ev.append({"source": "signature per SD (M2)", "outcome": r.outcome, "exposure": r.signature, "hr": r.hr,
               "e_value": evalue(r.hr), "e_value_ci": 1.0 if (r.lci <= 1 <= r.uci) else evalue(bound)})
pd.DataFrame(ev).to_csv(RES / "evalues.csv", index=False)

# ---------------------------------------------------------------- (3) heterogeneity biliary vs non-biliary AP
het = []
se_ci = lambda lo, hi: (np.log(hi) - np.log(lo)) / (2 * 1.96)
for src, tab in [("intake per serving (M2)", im), ("signature per SD (M2)", sm),
                 ("signature per SD, mutually adjusted (M2)",
                  (lambda t: t[(t.analysis == "per SD, mutually adjusted") & (t.model == "M2")])(pd.read_csv(RES / "signature_main.csv")))]:
    for x in (BEVS if src.startswith("intake") else ["sig_SSB", "sig_ASB"]):
        if src.startswith("intake"):
            g = tab.set_index("outcome")
            a, b_ = g.loc["AP_biliary"], g.loc["AP_nonbiliary"]
            la, lb = np.log(a[f"{x}_hr"]), np.log(b_[f"{x}_hr"])
            sa, sb = se_ci(a[f"{x}_lci"], a[f"{x}_uci"]), se_ci(b_[f"{x}_lci"], b_[f"{x}_uci"])
        else:
            g = tab[tab.signature == x].set_index("outcome")
            a, b_ = g.loc["AP_biliary"], g.loc["AP_nonbiliary"]
            la, lb = np.log(a.hr), np.log(b_.hr)
            sa, sb = se_ci(a.lci, a.uci), se_ci(b_.lci, b_.uci)
        z = (la - lb) / np.sqrt(sa ** 2 + sb ** 2)
        from scipy import stats
        het.append({"analysis": src, "exposure": x, "hr_biliary": np.exp(la), "hr_nonbiliary": np.exp(lb),
                    "ratio_of_hr": np.exp(la - lb), "p_heterogeneity": 2 * stats.norm.sf(abs(z))})
het = pd.DataFrame(het)
het.to_csv(RES / "heterogeneity_ap.csv", index=False)
print(het.round(3).to_string(index=False), flush=True)

# ---------------------------------------------------------------- (4) Fine-Gray input
FG = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "CP"]


def fg_frame(df, Zm, xcols, start):
    out = pd.concat([df[xcols].reset_index(drop=True), Zm.reset_index(drop=True).add_prefix("z_")], axis=1)
    defs = outcome_defs(df, FG, start)
    end = follow_up_end(df)
    died = df["death_date"].notna() & (df["death_date"] <= end)
    for k, (elig, e, T) in defs.items():
        out[f"elig_{k}"] = elig.astype(int).to_numpy()
        out[f"T_{k}"] = T.to_numpy()
        status = np.where(e == 1, 1, np.where(died & (df["death_date"] - start).dt.days.div(365.25).sub(T).abs().lt(0.01), 2, 0))
        out[f"S_{k}"] = status
    return out


zA = base_covariates(A)
zA.columns = [c.replace(" ", "_").replace("-", "m").replace(".", "_") for c in zA.columns]
fg_frame(A, zA, BEVS, A["start"]).to_csv(CACHE / "finegray_intake.csv", index=False)
sc = pd.read_pickle(CACHE / "signatures.pkl")
B = d.merge(sc, on="eid")
B = B[(B.consortium == 0) & ~B["train"]]
B = B[follow_up_end(B) > B["baseline_date"]]
B = B[~(B["Any_cancer"].notna() & (B["Any_cancer"] <= B["baseline_date"]))].reset_index(drop=True)
for s in ["sig_SSB", "sig_ASB", "sig_BMI"]:
    B[s] = (B[s] - B[s].mean()) / B[s].std()
zB = add_kidney(add_biomarkers(base_covariates(B, diet=False), B), B)
zB.columns = [c.replace(" ", "_").replace("-", "m").replace(".", "_") for c in zB.columns]
fg_frame(B, zB, ["sig_SSB", "sig_ASB"], B["baseline_date"]).to_csv(CACHE / "finegray_signature.csv", index=False)
print("Fine-Gray inputs written", flush=True)

# ---------------------------------------------------------------- (5) enrichment (g:Profiler)
assoc = pd.read_csv(RES / "protein_assoc.csv")
assoc = assoc[assoc.model == "main"]
bg = sorted(assoc.protein.unique())
sets = {"SSB_FDR": assoc[(assoc.exposure == "SSB") & (assoc.fdr < 0.05)].protein.tolist(),
        "ASB_signature": pd.read_csv(RES / "signature_weights_sig_ASB.csv", index_col=0).index.tolist(),
        "SSB_signature": pd.read_csv(RES / "signature_weights_sig_SSB.csv", index_col=0).index.tolist()}
up = lambda s: [g.upper().split("_")[0] for g in s]
for name, genes in sets.items():
    body = {"organism": "hsapiens", "query": up(genes), "sources": ["GO:BP", "KEGG", "REAC"],
            "user_threshold": 0.05, "significance_threshold_method": "fdr", "domain_scope": "custom",
            "background": up(bg), "no_evidences": True}
    r = subprocess.run(["curl", "-s", "--max-time", "120", "-H", "Content-Type: application/json", "-d", json.dumps(body),
                        "https://biit.cs.ut.ee/gprofiler/api/gost/profile/"], capture_output=True, text=True)
    try:
        res = json.loads(r.stdout)["result"]
        e = pd.DataFrame([{"source": x["source"], "term_id": x["native"], "term": x["name"], "p_adj": x["p_value"],
                           "term_size": x["term_size"], "overlap": x["intersection_size"], "query_size": x["query_size"]}
                          for x in res])
        e.to_csv(RES / f"enrichment_{name}.csv", index=False)
        print(name, len(genes), "genes;", len(e), "enriched terms;",
              "; ".join(e.sort_values("p_adj").head(6)["term"]) if len(e) else "", flush=True)
    except Exception as ex:
        print(name, "enrichment failed:", ex, r.stdout[:200])
