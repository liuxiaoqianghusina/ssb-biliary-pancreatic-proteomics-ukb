"""Two-sample cis-pQTL Mendelian randomization for the signature proteins associated with biliary or pancreatic
disease (FDR <0.05 in S04, Model 1) -> FinnGen R12 cholelithiasis, cholecystitis, acute and chronic pancreatitis.

Exposure: conditionally independent pQTL signals from the UKB-PPP discovery cohort (Sun et al., Nature 2023,
Supplementary Table 16): cis signals with -log10 P >= 7.3; within a protein, signals kept if r2 < 0.01 with the
signals already chosen (LD reported in the same table). ST16 effects are per SD of the standardised genotype and are
converted to per-allele effects by dividing by sqrt(2p(1-p)). Effect allele = ALT (hg19 variant ID).
Outcomes: FinnGen R12 summary statistics (no UK Biobank overlap), matched by rsID with allele checks; palindromic
variants with ALT frequency 0.42-0.58 dropped.
Estimators: Wald ratio (1 SNP); IVW (multiplicative random effects if Cochran's Q indicates heterogeneity);
MR-Egger and weighted median with >= 3 SNPs. FDR across all protein-outcome tests (primary estimate).
Outputs: results/cis_mr_instruments.csv, cis_mr_harmonised.csv, cis_mr_results.csv
"""
import os
import re
import subprocess
from io import StringIO
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from common import DIETPROJ, RES

MR = DIETPROJ / "data" / "mr"
OUTCOMES = {"K11_CHOLELITH": "Cholelithiasis", "K11_CHOLECYST": "Cholecystitis", "K11_ACUTPANC": "Acute pancreatitis",
            "K11_CHRONPANC": "Chronic pancreatitis"}
if os.environ.get("MR_OUTS"):                   # e.g. MR_OUTS=K11_CHOLELITH,K11_ACUTPANC for a partial run
    OUTCOMES = {k: v for k, v in OUTCOMES.items() if k in os.environ["MR_OUTS"].split(",")}
R2_MAX = 0.01
N_PPP = 34557   # UKB-PPP European discovery cohort (for approximate F statistics)

# ---------------------------------------------------------------- candidate proteins
pc = pd.read_csv(RES / "signature_protein_cox.csv")
cand = pc[(pc.fdr < 0.05) & pc.outcome.isin(["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "CP"])]
genes = sorted({p.split("_")[0].upper() for p in cand.protein})
print(len(genes), "candidate proteins:", ", ".join(genes), flush=True)

# ---------------------------------------------------------------- ST16 instruments
cache = MR / "ukbppp" / "ST16_all.csv"
if not cache.exists():
    wb = openpyxl.load_workbook(MR / "ukbppp" / "Sun2023_SupplementaryTables.xlsx", read_only=True)
    rows = list(wb["ST16"].iter_rows(min_row=6, values_only=True))
    st = pd.DataFrame([r[:16] for r in rows if r[0]], columns=[
        "pid", "vid", "rsid", "pip", "logbf", "beta", "se", "mlogp", "altfreq", "cistrans", "region", "ld",
        "cs_size", "cs_ids", "cs_rsids", "cs_pips"])
    st.to_csv(cache, index=False)
st = pd.read_csv(cache)
st["protein"] = st.pid.str.split(":").str[0]
st = st[st.protein.isin(genes) & (st.cistrans == "cis") & (st.mlogp >= 7.3) & st.rsid.astype(str).str.startswith("rs")].copy()
st[["chr", "pos19", "a0", "a1"]] = st.vid.str.split(":", expand=True).iloc[:, :4]
scale = np.sqrt(2 * st.altfreq * (1 - st.altfreq))
st["beta"], st["se"] = st.beta / scale, st.se / scale
keep = []
for p, g in st.sort_values("mlogp", ascending=False).groupby("protein", sort=False):
    chosen = []
    for _, r in g.iterrows():
        ld = dict((m.group(1), float(m.group(2))) for m in re.finditer(r"([\w:]+:imp:v1) \(([\d.eE-]+)\)", str(r.ld)))
        if all(ld.get(c.vid, 0) < R2_MAX for c in chosen):
            chosen.append(r)
    keep += chosen
inst = pd.DataFrame(keep)
inst["F"] = (inst.beta / inst.se) ** 2
inst.to_csv(RES / "cis_mr_instruments.csv", index=False)
print(f"instruments: {len(inst)} cis signals for {inst.protein.nunique()} of {len(genes)} proteins; "
      f"no cis-pQTL: {sorted(set(genes) - set(inst.protein))}", flush=True)


def outcome_rows(code, rsids):
    """Stream the FinnGen file once and keep rows whose rsids column (5th) contains an instrument rsID (awk hash)."""
    f = MR / "finngen_r12" / f"finngen_R12_{code}.gz"
    pat = MR / f"rsids_ssb_{code}.txt"
    pat.write_text("\n".join(rsids) + "\n")
    awk = ("NR==FNR{a[$1];next} FNR==1{print;next} "
           "{n=split($5,r,\",\"); for(i=1;i<=n;i++) if(r[i] in a){print; break}}")
    cmd = f"gzcat '{f}' | LC_ALL=C awk -F'\\t' '{awk}' '{pat}' -"
    return pd.read_csv(StringIO(subprocess.run(["bash", "-c", cmd], capture_output=True, text=True).stdout), sep="\t")


def harmonise(inst, o):
    rows = []
    o = o.assign(rs_list=o.rsids.astype(str).str.split(","))
    lookup = {rs: i for i, xs in zip(o.index, o.rs_list) for rs in xs}
    for _, r in inst.iterrows():
        if r.rsid not in lookup:
            continue
        m = o.loc[lookup[r.rsid]]
        if m.alt == r.a1 and m.ref == r.a0:
            b, af = m.beta, m.af_alt
        elif m.alt == r.a0 and m.ref == r.a1:
            b, af = -m.beta, 1 - m.af_alt
        else:
            continue
        if {r.a0, r.a1} in ({"A", "T"}, {"C", "G"}) and 0.42 <= r.altfreq <= 0.58:
            continue
        rows.append({"protein": r.protein, "rsid": r.rsid, "ea": r.a1, "oa": r.a0, "eaf_exp": r.altfreq, "eaf_out": af,
                     "pip": r.pip, "beta_exp": r.beta, "se_exp": r.se, "F": r.F, "beta_out": b, "se_out": m.sebeta,
                     "p_out": m.pval})
    return pd.DataFrame(rows)


def mr(h):
    bx, by, sx, sy = h.beta_exp.to_numpy(), h.beta_out.to_numpy(), h.se_exp.to_numpy(), h.se_out.to_numpy()
    ratio, rse = by / bx, sy / np.abs(bx)
    if len(h) == 1:
        return [{"method": "Wald ratio", "nsnp": 1, "b": ratio[0], "se": rse[0]}]
    out = []
    w = 1 / rse ** 2
    b = np.sum(w * ratio) / np.sum(w)
    q = np.sum(w * (ratio - b) ** 2)
    phi = max(1, q / (len(h) - 1))
    out.append({"method": "IVW", "nsnp": len(h), "b": b, "se": np.sqrt(1 / np.sum(w)) * np.sqrt(phi), "Q": q,
                "Q_p": stats.chi2.sf(q, len(h) - 1)})
    if len(h) >= 3:
        sign = np.sign(bx)
        X = np.column_stack([np.ones(len(h)), np.abs(bx)])
        W = np.diag(1 / sy ** 2)
        beta = np.linalg.solve(X.T @ W @ X, X.T @ W @ (by * sign))
        resid = by * sign - X @ beta
        s2 = max(1, (resid ** 2 / sy ** 2).sum() / (len(h) - 2))
        cov = np.linalg.inv(X.T @ W @ X) * s2
        out.append({"method": "MR-Egger", "nsnp": len(h), "b": beta[1], "se": np.sqrt(cov[1, 1]),
                    "egger_intercept": beta[0],
                    "egger_p": 2 * stats.t.sf(abs(beta[0] / np.sqrt(cov[0, 0])), len(h) - 2)})
        o = np.argsort(ratio)
        wr = 1 / rse[o] ** 2
        wm = np.interp(0.5, (np.cumsum(wr) - wr / 2) / wr.sum(), ratio[o])
        rng = np.random.default_rng(1)
        boots = []
        for _ in range(1000):
            rb = rng.normal(ratio, rse); ob = np.argsort(rb); wb = 1 / rse[ob] ** 2
            boots.append(np.interp(0.5, (np.cumsum(wb) - wb / 2) / wb.sum(), rb[ob]))
        out.append({"method": "Weighted median", "nsnp": len(h), "b": wm, "se": np.std(boots)})
    return out


from concurrent.futures import ThreadPoolExecutor
avail = {c: l for c, l in OUTCOMES.items() if (MR / "finngen_r12" / f"finngen_R12_{c}.gz").exists()}
with ThreadPoolExecutor(max_workers=4) as ex:
    raw = dict(zip(avail, ex.map(lambda c: outcome_rows(c, inst.rsid.tolist()), avail)))
rows, snps = [], []
for code, label in avail.items():
    h = harmonise(inst, raw[code])
    h["outcome"] = label
    snps.append(h)
    for p, g in h.groupby("protein"):
        for r in mr(g):
            r.update(protein=p, outcome=label, OR=np.exp(r["b"]), lci=np.exp(r["b"] - 1.96 * r["se"]),
                     uci=np.exp(r["b"] + 1.96 * r["se"]), p=2 * stats.norm.sf(abs(r["b"] / r["se"])))
            rows.append(r)
    print("done", label, len(h), "SNPs harmonised", flush=True)
res = pd.DataFrame(rows)
prim = res.method.isin(["Wald ratio", "IVW"])
res.loc[prim, "fdr"] = multipletests(res.loc[prim, "p"], method="fdr_bh")[1]
pd.concat(snps).to_csv(RES / "cis_mr_harmonised.csv", index=False)
res.to_csv(RES / "cis_mr_results.csv", index=False)

# observational direction (Model 1, independent sample) for comparison
obs_map = {"Cholelithiasis": "Gallstones", "Acute pancreatitis": "AP", "Chronic pancreatitis": "CP", "Cholecystitis": None}
pc["gene"] = pc.protein.str.split("_").str[0].str.upper()
obs = pc.set_index(["gene", "outcome"])["hr"]
pr = res[prim].copy()
pr["obs_HR"] = [obs.get((g, obs_map[o]), np.nan) if obs_map[o] else np.nan for g, o in zip(pr.protein, pr.outcome)]
pd.set_option("display.width", 220)
print(pr.sort_values("p")[["protein", "outcome", "method", "nsnp", "OR", "lci", "uci", "p", "fdr", "obs_HR"]]
      .round(4).head(30).to_string(index=False))
print("tests:", int(prim.sum()), "nominal P<0.05:", int((pr.p < 0.05).sum()), "FDR<0.05:", int((pr.fdr < 0.05).sum()))
