"""Sensitivity colocalisation in a narrow window (+/- 100 kb) around the cis-pQTL instrument used in S10, to reduce
the influence of strong independent cholelithiasis loci elsewhere in the +/- 1 Mb region (TM4SF4 near CPB1, CERS2 near
MENT, ADAR near IL6R), which violate the single-causal-variant assumption of coloc.abf.
Output: results/coloc_window.csv"""
import numpy as np
import pandas as pd

from common import DIETPROJ, RES

MR = DIETPROJ / "data" / "mr"
REG = MR / "finngen_regions_ssb"
OUT = {"K11_CHOLELITH": "Cholelithiasis", "K11_CHOLECYST": "Cholecystitis", "K11_ACUTPANC": "Acute pancreatitis",
       "K11_CHRONPANC": "Chronic pancreatitis"}
PROT = {"MENT": (1, 151047746, "MENT_chr1"), "CPB1": (3, 148791102, "CPB1_chr3"), "IL6R": (1, 154405141, "IL6R_chr1")}
WIN = 100_000


def logabf(beta, se, W):
    r = W / (W + se ** 2)
    return 0.5 * (np.log(1 - r) + r * (beta / se) ** 2)


def lsum(x):
    return np.max(x) + np.log(np.sum(np.exp(x - np.max(x))))


def coloc_abf(l1, l2, p1=1e-4, p2=1e-4, p12=1e-5):
    a1, a2, a12 = lsum(l1), lsum(l2), lsum(l1 + l2)
    lH3 = np.log(p1) + np.log(p2) + a1 + a2 + np.log(-np.expm1(a12 - (a1 + a2)))
    lH = np.array([0.0, np.log(p1) + a1, np.log(p2) + a2, lH3, np.log(p12) + a12])
    return np.exp(lH - lsum(lH))


inst = pd.read_csv(RES / "cis_mr_harmonised.csv")
rows = []
for prot, (chrom, start, folder) in PROT.items():
    p = pd.read_csv(next((MR / "ukbppp" / folder).rglob("*.gz")), sep=" ",
                    usecols=["GENPOS", "ID", "ALLELE0", "ALLELE1", "A1FREQ", "INFO", "BETA", "SE", "LOG10P"])
    p = p[(p.GENPOS >= start - 1_000_000) & (p.GENPOS <= start + 1_000_000) & (p.INFO >= 0.8)]
    snp = inst[(inst.protein == prot) & (inst.outcome == "Cholelithiasis")].sort_values("p_out").rsid.iloc[0]
    for code, label in OUT.items():
        o = pd.read_csv(REG / f"{code}.tsv", sep="\t")
        o = o[o["#chrom"].astype(str) == str(chrom)]
        m = p.merge(o, left_on="GENPOS", right_on="pos")
        same = (m.ALLELE1 == m.alt) & (m.ALLELE0 == m.ref)
        flip = (m.ALLELE1 == m.ref) & (m.ALLELE0 == m.alt)
        m = m[same | flip].copy()
        m["beta_out"] = np.where(same[same | flip], m.beta, -m.beta)
        centre = m[m.rsids.astype(str).str.split(",").apply(lambda x: snp in x)]
        if centre.empty:
            continue
        c = int(centre.pos.iloc[0])
        w = m[(m.pos >= c - WIN) & (m.pos <= c + WIN)]
        l1 = logabf(w.BETA.to_numpy(), w.SE.to_numpy(), 0.15 ** 2)
        l2 = logabf(w.beta_out.to_numpy(), w.sebeta.to_numpy(), 0.2 ** 2)
        pp = coloc_abf(l1, l2)
        post = np.exp(l1 + l2 - lsum(l1 + l2))
        rows.append({"protein": prot, "outcome": label, "window_centre": f"{snp} (chr{chrom}:{c})", "nsnps": len(w),
                     **{f"PP.H{i}": pp[i] for i in range(5)}, "top_shared_rsid": w.iloc[int(np.argmax(post))].rsids,
                     "outcome_min_p_in_window": w.pval.min(), "protein_min_log10p_in_window": w.LOG10P.max()})
r = pd.DataFrame(rows)
r.to_csv(RES / "coloc_window.csv", index=False)
pd.set_option("display.width", 220)
print(r.round(4).to_string(index=False))
