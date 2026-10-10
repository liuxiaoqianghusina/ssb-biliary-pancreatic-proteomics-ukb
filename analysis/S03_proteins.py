"""Plasma proteins associated with SSB and ASB, and their proteomic signatures.

Sample: randomly selected (non-consortium), QC-passed Olink participants with >=1 valid recall (n~17,000).
(1) Protein-wise OLS of each protein (z-scored, plate-adjusted) on SSB and ASB (servings/day, mutually adjusted),
    adjusted for M2 covariates + fasting time; sensitivity without BMI. FDR (Benjamini-Hochberg).
(2) Replication of the proteins highlighted by Zhao et al. (Nat Food 2025, Fig. 4).
(3) Elastic-net signatures (same 70/30 split as the diet-proteome project, seed 2026): sig_SSB, sig_ASB on intake;
    residual signatures rsig_* on intake residualised (training-set OLS) on demographic, lifestyle, adiposity, liver,
    glycaemic, lipid, kidney and inflammatory covariates, so that they are not adiposity/metabolic signatures.
Outputs: results/protein_assoc.csv, results/natfood_replication.csv, results/signature_performance.csv,
         results/signature_weights_*.csv, cache/signatures.pkl
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNetCV
from statsmodels.stats.multitest import multipletests

from common import BEVS, CACHE, DIETPROJ, RES, base_covariates, ols_many

P = pd.read_pickle(DIETPROJ / "cache" / "olink_qc_plateadj.pkl")
old = pd.read_pickle(DIETPROJ / "cache" / "signatures.pkl").set_index("eid").loc[P.index]
o = pd.read_pickle(CACHE / "analysis.pkl").set_index("eid").loc[P.index].reset_index()
for b in BEVS:
    o[b] = o[b].clip(upper=6)
train, test = old["train"].to_numpy(), old["test"].to_numpy()
diet = ((o.consortium == 0) & (o.n_valid_recalls >= 1)).to_numpy()
ind = (o.consortium == 0).to_numpy() & ~train
assert not (train & ~diet).any()
X = P.to_numpy()
print("Olink", X.shape, "diet sample", diet.sum(), "train", train.sum(), "test", test.sum(), "independent", ind.sum())

# ---------------------------------------------------------------- (1) protein-wise associations
Z = base_covariates(o, diet=True)
Z["fasting_h"] = o["fasting_h"].fillna(o["fasting_h"].median())
rows = []
for label, Zm in [("main", Z), ("without BMI", Z.drop(columns=["bmi"]))]:
    for b, other in [("SSB", "ASB"), ("ASB", "SSB")]:
        Zb = Zm.assign(other=o[other]).to_numpy()[diet]
        beta, se, p = ols_many(o[b].to_numpy()[diet], Zb, X[diet])
        rows.append(pd.DataFrame({"model": label, "exposure": b, "protein": P.columns, "beta": beta, "se": se, "p": p,
                                  "fdr": multipletests(p, method="fdr_bh")[1]}))
assoc = pd.concat(rows, ignore_index=True)
assoc.to_csv(RES / "protein_assoc.csv", index=False)
for (mdl, b), a in assoc.groupby(["model", "exposure"]):
    a = a.sort_values("p")
    print(f"{mdl:12s} {b}: FDR<0.05 n={int((a.fdr < 0.05).sum())} (up {int(((a.fdr < 0.05) & (a.beta > 0)).sum())}); top:",
          ", ".join(f"{r.protein}({r.beta:+.3f})" for r in a.head(12).itertuples()))

# ---------------------------------------------------------------- (2) replication of Nat Food Fig. 4 proteins
NF = {"SSB": ["hgf", "fgf21", "il1rn", "mmp9", "pglyrp1", "osm", "plaur", "fabp4", "ccl16", "pgf", "clec4d", "ces1",
              "prss8", "stc1", "itga5", "ism1", "sele", "tgfa", "inhbc", "f9"],
      "ASB": ["oxt", "adgrg1", "adm", "ghrl", "chi3l1", "adamts8", "cpm", "lilrb4", "hgf", "lgals9", "cd300c", "fgf21",
              "tff1", "fabp4", "lbr", "il1rn", "obp2b", "milr1", "tnfrsf10b", "gpkow"]}
rep = assoc[assoc.model == "main"].set_index(["exposure", "protein"])
rr = []
for b, prots in NF.items():
    for pr in prots:
        if (b, pr) in rep.index:
            r = rep.loc[(b, pr)]
            rr.append({"exposure": b, "protein": pr, "beta": r.beta, "p": r.p, "fdr": r.fdr})
        else:
            rr.append({"exposure": b, "protein": pr, "beta": np.nan, "p": np.nan, "fdr": np.nan})
rr = pd.DataFrame(rr)
rr.to_csv(RES / "natfood_replication.csv", index=False)
for b, g in rr.groupby("exposure"):
    print(f"Nat Food {b} proteins measured {g.beta.notna().sum()}/{len(g)}; P<0.05 {int((g.p < 0.05).sum())}; "
          f"FDR<0.05 {int((g.fdr < 0.05).sum())}")

# ---------------------------------------------------------------- (3) signatures
C = base_covariates(o, diet=True)
for c in ["fasting_h", "waist", "alt", "ggt", "hba1c", "tg", "cysc", "creat", "crp"]:
    v = o[c] if c in ("fasting_h", "waist", "hba1c") else np.log(o[c].clip(lower=1e-3))
    C[c] = v.fillna(v.median()).to_numpy()
Cm = np.column_stack([np.ones(len(o)), C.to_numpy()])


def fit_signature(y):
    ok = train & np.isfinite(y)
    en = ElasticNetCV(l1_ratio=0.5, alphas=20, cv=5, max_iter=5000, n_jobs=-1, random_state=1).fit(X[ok], y[ok])
    pred = en.predict(X)
    return (pred - pred.mean()) / pred.std(), en


scores = pd.DataFrame({"eid": o["eid"], "train": train, "test": test, "independent": ind,
                       "sig_BMI": old["sig_BMI"].to_numpy()})
perf = []
for b in BEVS:
    y = o[b].to_numpy()
    for kind in ("sig", "rsig"):
        if kind == "rsig":
            okt = train & np.isfinite(y)
            beta = np.linalg.lstsq(Cm[okt], y[okt], rcond=None)[0]
            target = y - Cm @ beta
        else:
            target = y
        s, en = fit_signature(target)
        scores[f"{kind}_{b}"] = s
        w = pd.Series(en.coef_, index=P.columns)
        w[w != 0].sort_values().to_csv(RES / f"signature_weights_{kind}_{b}.csv", header=["weight"])
        tt = test & np.isfinite(target)
        perf.append({"signature": f"{kind}_{b}", "target": "intake" if kind == "sig" else "residualised intake",
                     "n_train": int((train & np.isfinite(target)).sum()), "n_test": int(tt.sum()),
                     "n_proteins": int((w != 0).sum()), "alpha": en.alpha_,
                     "pearson_r_test": stats.pearsonr(s[tt], target[tt])[0],
                     "spearman_r_test": stats.spearmanr(s[tt], y[tt])[0],
                     "r_with_intake_test": stats.pearsonr(s[tt], y[tt])[0]})
        print(perf[-1], flush=True)
perf = pd.DataFrame(perf)
cor = scores.loc[ind, ["sig_SSB", "sig_ASB", "rsig_SSB", "rsig_ASB", "sig_BMI"]].corr()
for k in perf.signature:
    perf.loc[perf.signature == k, "r_with_BMI_signature"] = cor.at[k, "sig_BMI"]
perf.to_csv(RES / "signature_performance.csv", index=False)
cor.to_csv(RES / "signature_correlations.csv")
scores.to_pickle(CACHE / "signatures.pkl")
print(cor.round(2).to_string())
