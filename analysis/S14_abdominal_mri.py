"""Sweetened beverage intake, proteomic signatures and abdominal MRI organ fat (imaging visit 2015-2020).

Phenotypes (UK Biobank abdominal organ composition, instance 2): pancreas PDFF (21090) and liver PDFF (21088),
log-transformed and reported as % difference; visceral (21085) and abdominal subcutaneous (21086) fat volume and
pancreas volume (21087) in litres. For comparability, the figure uses phenotypes standardised to SD units.
Intake: participants with >=1 valid recall; SSB and ASB mutually adjusted; models
  A = Model 2 (incl. baseline BMI) + years from first recall to imaging + imaging centre      [main]
  B = A + BMI at the imaging visit
  C = A without BMI
repeated in participants with no major dietary change, stable weight and no diabetes.
Signatures: QC-passed non-consortium Olink participants outside the training set (main) and all non-consortium
participants (sensitivity); covariates as signature Model 1 + years from baseline to imaging + imaging centre,
with the same BMI variants; each signature alone and both together.
Inputs: UK Biobank assessment-centre extract (cached to cache/abdominal_mri.pkl) and cache/diet_change.pkl from S12.
Outputs: results/mri_intake.csv, results/mri_signature.csv, results/mri_flow.csv
"""
import warnings

import numpy as np
import pandas as pd

from common import BEVS, CACHE, RES, UKB, base_covariates, ols_many

warnings.filterwarnings("ignore")
PHENO = {"Pancreas PDFF": ("p21090_i2", True), "Liver PDFF": ("p21088_i2", True),
         "Visceral fat volume": ("p21085_i2", False), "Abdominal subcutaneous fat volume": ("p21086_i2", False),
         "Pancreas volume": ("p21087_i2", False)}
SCALE = {"Pancreas volume": (1000, "mL")}        # litres -> mL for pancreas volume

# ---------------------------------------------------------------- extra fields
if not (CACHE / "abdominal_mri.pkl").exists():
    cols = (["eid", "p53_i2", "p53_i3", "p54_i2", "p21001_i2"]
            + [f"p{f}_i{i}" for f in (21080, 21085, 21086, 21087, 21088, 21089, 21090, 21091) for i in (2, 3)])
    x = pd.concat(pd.read_csv(UKB / "UKB619657_assessment_center_2025Nov.csv", usecols=cols, dtype=str,
                              chunksize=100000), ignore_index=True)
    x["eid"] = x["eid"].astype(int)
    for c in cols[1:]:
        if c not in ("p53_i2", "p53_i3"):
            x[c] = pd.to_numeric(x[c], errors="coerce")
    x.to_pickle(CACHE / "abdominal_mri.pkl")
if not (CACHE / "diet_change.pkl").exists():
    raise SystemExit("Run S12_reverse_causation.py first (writes cache/diet_change.pkl)")

d = (pd.read_pickle(CACHE / "analysis.pkl").merge(pd.read_pickle(CACHE / "abdominal_mri.pkl"), on="eid", how="left")
     .merge(pd.read_pickle(CACHE / "diet_change.pkl"), on="eid", how="left"))
d["img_date"] = pd.to_datetime(d["p53_i2"], errors="coerce")
d["stable"] = d["diet_change"].eq(0) & pd.to_numeric(d["p2306_i0"], errors="coerce").eq(0) & d["diabetes_sr"].fillna(0).eq(0)


def design(df, Z, start):
    Z = Z.copy()
    gap = (df["img_date"] - start).dt.days / 365.25
    Z["gap_years"] = gap.fillna(gap.median()).to_numpy()
    Z = Z.join(pd.get_dummies(df["p54_i2"].fillna(-1).astype(int), prefix="imgcentre", drop_first=True, dtype=float)
               .set_axis(Z.index))
    return Z


def models(df, Z):
    bmi_img = df["p21001_i2"]
    return {"A: Model 2 (baseline BMI)": Z,
            "B: + BMI at imaging": Z.assign(bmi_img=bmi_img.fillna(bmi_img.median()).to_numpy()),
            "C: without BMI": Z.drop(columns=["bmi"])}


def fit(x, Zm, y, ok, log):
    b, se, p = ols_many(x[ok], Zm[ok], y[ok, None])
    b, se, p = b[0], se[0], p[0]
    if log:
        return 100 * (np.exp(b) - 1), 100 * (np.exp(b - 1.96 * se) - 1), 100 * (np.exp(b + 1.96 * se) - 1), p
    return b, b - 1.96 * se, b + 1.96 * se, p


flow = {}
# ---------------------------------------------------------------- intake
A = d[(d.n_valid_recalls >= 1) & d.img_date.notna()].copy().reset_index(drop=True)
for b in BEVS:
    A[b] = A[b].clip(upper=6)
A["start"] = A["first_recall_date"].fillna(A["baseline_date"])
A["age_start"] = A["age"] + (A["start"] - A["baseline_date"]).dt.days / 365.25
A = A[A["img_date"] > A["start"]].reset_index(drop=True)
Z0 = design(A, base_covariates(A), A["start"])
rows = []
for name, (col, log) in PHENO.items():
    raw = A[col] * SCALE.get(name, (1, ""))[0]
    y = np.log(raw.clip(lower=0.1)) if log else raw
    ysd = (y - y.mean()) / y.std()
    flow[f"intake sample, {name}"] = int(y.notna().sum())
    for sample, mask in [("All participants", np.ones(len(A), bool)),
                         ("No dietary change, stable weight, no diabetes", A["stable"].to_numpy())]:
        ok = y.notna().to_numpy() & mask
        for mname, Zm in models(A, Z0).items():
            for b, other in [("SSB", "ASB"), ("ASB", "SSB")]:
                Zb = Zm.assign(other=A[other].to_numpy()).to_numpy()
                est, lo, hi, p = fit(A[b].to_numpy(), Zb, y.to_numpy(), ok, log)
                sd, slo, shi, _ = fit(A[b].to_numpy(), Zb, ysd.to_numpy(), ok, False)
                rows.append({"phenotype": name, "scale": "% difference" if log else SCALE.get(name, (1, "litres"))[1], "sample": sample,
                             "model": mname, "exposure": b, "n": int(ok.sum()), "estimate": est, "lci": lo, "uci": hi,
                             "p": p, "sd_estimate": sd, "sd_lci": slo, "sd_uci": shi})
ri = pd.DataFrame(rows)
ri.to_csv(RES / "mri_intake.csv", index=False)

# ---------------------------------------------------------------- signatures
sc = pd.read_pickle(CACHE / "signatures.pkl")
B = d.merge(sc, on="eid")
B = B[(B.consortium == 0) & B.img_date.notna()].reset_index(drop=True)
for s in ["sig_SSB", "sig_ASB"]:
    B[s] = (B[s] - B[s].mean()) / B[s].std()
Zs = design(B, base_covariates(B, diet=False), B["baseline_date"])
rows = []
for name, (col, log) in PHENO.items():
    raw = B[col] * SCALE.get(name, (1, ""))[0]
    y = np.log(raw.clip(lower=0.1)) if log else raw
    ysd = (y - y.mean()) / y.std()
    for sample, mask in [("Independent sample", (~B["train"]).to_numpy()), ("All non-consortium", np.ones(len(B), bool))]:
        ok = y.notna().to_numpy() & mask
        flow[f"signature sample ({sample}), {name}"] = int(ok.sum())
        for mname, Zm in models(B, Zs).items():
            for s, other in [("sig_SSB", None), ("sig_ASB", None), ("sig_SSB", "sig_ASB"), ("sig_ASB", "sig_SSB")]:
                Zb = (Zm.assign(other=B[other].to_numpy()) if other else Zm).to_numpy()
                est, lo, hi, p = fit(B[s].to_numpy(), Zb, y.to_numpy(), ok, log)
                sd, slo, shi, _ = fit(B[s].to_numpy(), Zb, ysd.to_numpy(), ok, False)
                rows.append({"phenotype": name, "scale": "% difference" if log else SCALE.get(name, (1, "litres"))[1], "sample": sample,
                             "model": mname, "signature": s, "mutual": other is not None, "n": int(ok.sum()),
                             "estimate": est, "lci": lo, "uci": hi, "p": p, "sd_estimate": sd, "sd_lci": slo,
                             "sd_uci": shi})
rs = pd.DataFrame(rows)
rs.to_csv(RES / "mri_signature.csv", index=False)
pd.Series(flow).to_csv(RES / "mri_flow.csv", header=["n"])

# ---------------------------------------------------------------- print
pd.set_option("display.width", 250)
f = lambda r: f"{r.estimate:.2f} ({r.lci:.2f} to {r.uci:.2f})" + ("*" if r.p < 0.05 else "")
ri["est"] = ri.apply(f, axis=1)
print(ri.pivot_table(index=["phenotype", "exposure"], columns=["sample", "model"], values="est", aggfunc="first").to_string())
rs["est"] = rs.apply(f, axis=1)
x = rs[rs["sample"] == "Independent sample"]
print(x.pivot_table(index=["phenotype", "signature"], columns=["mutual", "model"], values="est", aggfunc="first").to_string())
print(pd.Series(flow).to_string())
