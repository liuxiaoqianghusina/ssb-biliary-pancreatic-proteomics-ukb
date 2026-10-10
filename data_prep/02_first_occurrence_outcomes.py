"""Re-define the 65 screen outcomes with UKB First Occurrence fields (primary care + hospital + self-report + death).

First Occurrence (FO) is coded at the 3-character ICD-10 level. Outcomes that need 4-character codes use the
closest FO parent where it is dominated by the target condition (noted below), or stay hospital-only (UTI).
FO special dates (1900-01-01, 1901-01-01, 1902-02-02, 1903-03-03, 1909-09-09) fall before baseline and therefore
count as prevalent; 2037-07-07 (date in the future) is set to missing.
Cancers keep the cancer registry; death keeps the death registry; Dementia/PD algorithmic outcomes are kept as well.
Output: cache/fo_outcomes.pkl (same column names as icd_outcomes.pkl except two renamed outcomes),
        results/outcome_definitions_fo.csv
"""
import os
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
C = Path(os.environ.get("SCREEN_DIR", HERE / "cache"))
RES = HERE / "results"
D = Path(os.environ.get("UKB_DATA_DIR", "../ukb_data"))
t0 = time.time()

# outcome: (FO 3-character codes, extra hospital ICD-10 prefixes, note)
DEF = {
    "GERD": (["K21"], [], ""), "Gallstones": (["K80"], [], ""), "Cholecystitis": (["K81"], [], ""),
    "IBD": (["K50", "K51"], [], ""), "Crohn": (["K50"], [], ""), "UC": (["K51"], [], ""),
    "Acute_pancreatitis": (["K85"], [], ""),
    "Chronic_pancreatitis": (["K86"], [], "FO K86 = other diseases of pancreas (mainly chronic pancreatitis)"),
    "Diverticular": (["K57"], [], ""), "IBS": (["K58"], [], ""),
    "PUD": (["K25", "K26", "K27"], [], ""), "Gastritis": (["K29"], [], ""),
    "Constipation": (["K59"], [], "FO K59 = other functional intestinal disorders (mainly constipation)"),
    "Haemorrhoids": (["K64", "I84"], [], ""), "Appendicitis": (["K35", "K36", "K37"], [], ""),
    "MASLD": (["K76"], ["K758"], "FO K76 = other diseases of liver (mainly fatty liver) + hospital K75.8"),
    "Cirrhosis": (["K74"], ["K703", "K717"], "FO K74 + hospital K70.3/K71.7"),
    "T2D": (["E11"], [], ""), "CKD": (["N18"], [], ""), "Kidney_stones": (["N20", "N21", "N22", "N23"], [], ""),
    "Gout": (["M10"], [], ""), "Obesity_dx": (["E66"], [], ""),
    "Hyperuricaemia": (["E79"], [], "FO E79 = purine/pyrimidine metabolism disorders"),
    "Hypertension": (["I10", "I11", "I12", "I13", "I15"], [], ""),
    "CHD": (["I20", "I21", "I22", "I23", "I24", "I25"], [], ""), "AF": (["I48"], [], ""), "HF": (["I50"], [], ""),
    "Stroke_isch": (["I63"], [], ""), "Stroke_any": (["I60", "I61", "I62", "I63", "I64"], [], ""),
    "VTE": (["I26", "I80", "I81", "I82"], [], ""), "PAD": (["I70"], ["I739"], "FO I70 + hospital I73.9"),
    "Dementia_icd": (["F00", "F01", "F02", "F03", "G30"], [], ""), "PD_icd": (["G20"], [], ""),
    "Depression_icd": (["F32", "F33"], [], ""), "Anxiety_icd": (["F40", "F41"], [], ""),
    "Sleep_disorders": (["G47"], [], "FO G47 = all sleep disorders (was hospital G47.3 apnoea)"),
    "Epilepsy": (["G40"], [], ""), "Osteoporosis": (["M80", "M81"], [], ""),
    "OA": (["M15", "M16", "M17", "M18", "M19"], [], ""), "RA": (["M05", "M06"], [], ""),
    "COPD": (["J43", "J44"], [], ""), "Asthma": (["J45", "J46"], [], ""),
    "Pneumonia": (["J12", "J13", "J14", "J15", "J16", "J17", "J18"], [], ""), "Glaucoma": (["H40"], [], ""),
    "Retinal_disorders": (["H35"], [], "FO H35 = other retinal disorders (mainly AMD; was hospital H35.3)"),
    "Cataract": (["H25", "H26"], [], ""), "Psoriasis": (["L40"], [], ""),
    "UTI": ([], ["N390"], "hospital N39.0 only (FO N39 includes incontinence)"),
    "Sepsis": (["A40", "A41"], [], ""), "Anaemia": (["D50", "D51", "D52", "D53", "D62", "D63", "D64"], [], ""),
    "Hearing_loss": (["H90", "H91"], [], ""), "Periodontal": (["K05"], [], ""), "Dental_caries": (["K02"], [], ""),
}

dd = pd.read_csv(D / "app45551_20251118060954.dataset.data_dictionary.csv", usecols=["name", "title"])
fo = dd[dd.title.str.match(r"Date [A-Z]\d\d first reported", na=False)].copy()
fo["code"] = fo.title.str.extract(r"Date ([A-Z]\d\d) first")
code2field = dict(zip(fo.code, fo.name))
need = sorted({code2field[c] for v in DEF.values() for c in v[0]})
missing = [c for v in DEF.values() for c in v[0] if c not in code2field]
assert not missing, missing

f = D / "UKB619657_health_outcomes_2025Nov.csv"
raw = pd.concat(pd.read_csv(f, usecols=["eid"] + need, dtype=str, chunksize=50000), ignore_index=True)
raw["eid"] = raw["eid"].astype(int)
print("FO fields read", raw.shape, f"{time.time()-t0:.0f}s", flush=True)
dates = raw[need].apply(pd.to_datetime, errors="coerce")
dates = dates.mask(dates == pd.Timestamp("2037-07-07"))

hosp = pd.read_pickle(C / "icd_outcomes.pkl")      # for cancers, death, and the hospital 4-character codes
hosp = hosp.set_index("eid").reindex(raw["eid"]).reset_index()

# hospital 4-character codes: rebuild from the same ICD extraction as 01_build.py
o = pd.read_pickle(Path(os.environ.get("UKB_EXTRACT_DIR", "../ukb_extracts")) / "outcomes.pkl")
o["eid"] = o["eid"].astype(int)
o = o.set_index("eid").reindex(raw["eid"]).reset_index()
date_cols = sorted([c for c in o.columns if c.startswith("p41280_a")], key=lambda x: int(x.split("_a")[1]))
codes = o["p41270"].fillna("").str.split("|").explode().to_frame("code")
codes["pos"] = codes.groupby(level=0).cumcount()
codes = codes[(codes["code"] != "") & (codes["pos"] < len(date_cols))]
dmat = o[date_cols].apply(pd.to_datetime, errors="coerce").to_numpy()
codes["date"] = dmat[codes.index.to_numpy(), codes["pos"].to_numpy()]
print("hospital codes", len(codes), f"{time.time()-t0:.0f}s", flush=True)

ev = pd.DataFrame({"eid": raw["eid"]})
rows = []
for k, (fo_codes, hosp_pref, note) in DEF.items():
    parts = [dates[code2field[c]] for c in fo_codes]
    if hosp_pref:
        hit = codes[codes["code"].str.startswith(tuple(hosp_pref))]
        parts.append(hit.groupby(level=0)["date"].min().reindex(ev.index))
    ev[k] = pd.concat(parts, axis=1).min(axis=1)
    rows.append({"outcome": k, "FO codes": " ".join(fo_codes), "extra hospital codes": " ".join(hosp_pref),
                 "note": note, "ever recorded": int(ev[k].notna().sum())})
for k in ["CRC", "Liver_cancer", "Pancreatic_cancer", "Kidney_cancer", "Breast_cancer", "Endometrial_cancer",
          "Oesophageal_cancer", "Gastric_cancer", "Any_cancer", "death_date"]:
    ev[k] = hosp[k].to_numpy()
    rows.append({"outcome": k, "FO codes": "", "extra hospital codes": "", "note": "cancer/death registry (unchanged)",
                 "ever recorded": int(ev[k].notna().sum())})
ev.to_pickle(C / "fo_outcomes.pkl")
pd.DataFrame(rows).to_csv(RES / "outcome_definitions_fo.csv", index=False)
print(pd.DataFrame(rows)[["outcome", "ever recorded"]].to_string(index=False))
print("done", f"{time.time()-t0:.0f}s")
