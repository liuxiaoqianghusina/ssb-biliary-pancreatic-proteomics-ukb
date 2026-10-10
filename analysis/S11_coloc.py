"""Colocalisation and full-region MR for the proteins with FDR-significant cis-pQTL MR (S10): MENT, CPB1, IL6R.

Exposure: UKB-PPP discovery cohort (European) REGENIE summary statistics downloaded from Synapse (GENPOS hg38,
BETA per ALLELE1, protein in SD units); variants with INFO >= 0.8. Outcomes: FinnGen R12 (hg38; effect allele = alt)
cholelithiasis, cholecystitis, acute and chronic pancreatitis. Region: gene start +/- 1 Mb (Ensembl GRCh38).
coloc.abf: prior variances 0.15^2 (protein) and 0.2^2 (log OR); p1 = p2 = 1e-4, p12 = 1e-5 (sensitivity 5e-6).
MR: Wald ratio for the lead cis-pQTL (overall and MAF >= 1%) and IVW over the conditionally independent cis signals
(ST16, -log10 P >= 7.3) using marginal estimates from the regional data.
Outputs: results/coloc_results.csv, results/cis_mr_region.csv
"""
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from scipy import stats

from common import DIETPROJ, RES

MR = DIETPROJ / "data" / "mr"
OUT = {"K11_CHOLELITH": "Cholelithiasis", "K11_CHOLECYST": "Cholecystitis", "K11_ACUTPANC": "Acute pancreatitis",
       "K11_CHRONPANC": "Chronic pancreatitis"}
PROT = {"MENT": (1, 151047746, "MENT_chr1"), "CPB1": (3, 148791102, "CPB1_chr3"), "IL6R": (1, 154405141, "IL6R_chr1")}
FLANK = 1_000_000
REG = MR / "finngen_regions_ssb"
REG.mkdir(exist_ok=True)


def cache_region(code):
    f = REG / f"{code}.tsv"
    if f.exists():
        return
    cond = " || ".join(f'($1=="{c}" && $2>={p - FLANK} && $2<={p + FLANK})' for c, p, _ in PROT.values())
    src = MR / "finngen_r12" / f"finngen_R12_{code}.gz"
    subprocess.run(["bash", "-c", f"gzcat '{src}' | LC_ALL=C awk -F'\\t' 'NR==1 || {cond}' > '{f}.tmp' && mv '{f}.tmp' '{f}'"],
                   check=True)


with ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(cache_region, OUT))
print("FinnGen regions cached", flush=True)


def logabf(beta, se, W):
    r = W / (W + se ** 2)
    return 0.5 * (np.log(1 - r) + r * (beta / se) ** 2)


def lsum(x):
    return np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))


def coloc_abf(l1, l2, p1=1e-4, p2=1e-4, p12=1e-5):
    a1, a2, a12 = lsum(l1), lsum(l2), lsum(l1 + l2)
    d = a12 - (a1 + a2)
    lH3 = np.log(p1) + np.log(p2) + a1 + a2 + np.log(-np.expm1(d))
    lH = np.array([0.0, np.log(p1) + a1, np.log(p2) + a2, lH3, np.log(p12) + a12])
    return np.exp(lH - lsum(lH))


st16 = pd.read_csv(MR / "ukbppp" / "ST16_all.csv")
st16["protein"] = st16.pid.str.split(":").str[0]
rows_c, rows_m = [], []
for prot, (chrom, start, folder) in PROT.items():
    f = next((MR / "ukbppp" / folder).rglob("*.gz"))
    p = pd.read_csv(f, sep=" ", usecols=["GENPOS", "ID", "ALLELE0", "ALLELE1", "A1FREQ", "INFO", "N", "BETA", "SE", "LOG10P"])
    p = p[(p.GENPOS >= start - FLANK) & (p.GENPOS <= start + FLANK) & (p.INFO >= 0.8)].copy()
    p["maf"] = np.minimum(p.A1FREQ, 1 - p.A1FREQ)
    lead = p.loc[p.LOG10P.idxmax()]
    print(f"{prot}: {len(p)} variants; lead {lead.ID} -log10P {lead.LOG10P:.1f}", flush=True)
    ind_ids = st16[(st16.protein == prot) & (st16.cistrans == "cis") & (st16.mlogp >= 7.3)].vid.tolist()
    for code, label in OUT.items():
        o = pd.read_csv(REG / f"{code}.tsv", sep="\t")
        o = o[(o["#chrom"].astype(str) == str(chrom)) & (o.pos >= start - FLANK) & (o.pos <= start + FLANK)]
        m = p.merge(o, left_on="GENPOS", right_on="pos")
        same = (m.ALLELE1 == m.alt) & (m.ALLELE0 == m.ref)
        flip = (m.ALLELE1 == m.ref) & (m.ALLELE0 == m.alt)
        m = m[same | flip].copy()
        m["beta_out"] = np.where(same[same | flip], m.beta, -m.beta)
        m["se_out"] = m.sebeta
        l1 = logabf(m.BETA.to_numpy(), m.SE.to_numpy(), 0.15 ** 2)
        l2 = logabf(m.beta_out.to_numpy(), m.se_out.to_numpy(), 0.2 ** 2)
        post = np.exp(l1 + l2 - lsum(l1 + l2))
        top = m.iloc[int(np.argmax(post))]
        for p12 in (1e-5, 5e-6):
            pp = coloc_abf(l1, l2, p12=p12)
            rows_c.append({"protein": prot, "outcome": label, "p12": p12, "nsnps": len(m),
                           **{f"PP.H{i}": pp[i] for i in range(5)}, "top_shared_variant": top.ID,
                           "top_shared_rsid": top.rsids, "SNP.PP.H4": post.max(),
                           "outcome_min_p_in_region": o.pval.min(), "protein_lead": lead.ID})

        def wald(r, lab):
            b, se = r.beta_out / r.BETA, r.se_out / abs(r.BETA)
            return {"protein": prot, "outcome": label, "instrument": lab, "variant": r.ID, "rsid": r.rsids,
                    "eaf": r.A1FREQ, "F": (r.BETA / r.SE) ** 2, "nsnp": 1, "b": b, "se": se}
        common = m[(m.maf >= 0.01) & (m.LOG10P >= 7.3)]
        if len(common):
            rows_m.append(wald(common.loc[common.LOG10P.idxmax()], "Lead common cis-pQTL (MAF >= 1%)"))
        ld = m.loc[m.LOG10P.idxmax()]
        if ld.LOG10P >= 7.3 and ld.maf < 0.01:
            rows_m.append(wald(ld, "Lead cis-pQTL overall (rare)"))
        ind = m[m.ID.isin(ind_ids)]
        if len(ind) >= 2:
            ratio, rse = ind.beta_out / ind.BETA, ind.se_out / ind.BETA.abs()
            w = 1 / rse ** 2
            b = (w * ratio).sum() / w.sum()
            q = (w * (ratio - b) ** 2).sum()
            rows_m.append({"protein": prot, "outcome": label, "instrument": "IVW, independent cis signals (marginal)",
                           "variant": ";".join(ind.ID), "rsid": ";".join(ind.rsids.astype(str)), "nsnp": len(ind), "b": b,
                           "se": np.sqrt(1 / w.sum()) * np.sqrt(max(1, q / (len(ind) - 1))), "Q": q,
                           "Q_p": stats.chi2.sf(q, len(ind) - 1)})
rc, rm = pd.DataFrame(rows_c), pd.DataFrame(rows_m)
rm["OR"], rm["lci"], rm["uci"] = np.exp(rm.b), np.exp(rm.b - 1.96 * rm.se), np.exp(rm.b + 1.96 * rm.se)
rm["p"] = 2 * stats.norm.sf((rm.b / rm.se).abs())
rc.to_csv(RES / "coloc_results.csv", index=False)
rm.to_csv(RES / "cis_mr_region.csv", index=False)
pd.set_option("display.width", 250)
print(rc[rc.p12 == 1e-5][["protein", "outcome", "nsnps", "PP.H0", "PP.H1", "PP.H2", "PP.H3", "PP.H4", "top_shared_rsid",
                          "SNP.PP.H4", "outcome_min_p_in_region"]].round(4).to_string(index=False))
print(rc[rc.p12 == 5e-6][["protein", "outcome", "PP.H3", "PP.H4"]].round(4).to_string(index=False))
print(rm[["protein", "outcome", "instrument", "rsid", "nsnp", "OR", "lci", "uci", "p"]].round(4).to_string(index=False))
