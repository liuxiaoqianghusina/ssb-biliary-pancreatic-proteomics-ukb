"""Sweetened-beverage exposures and a broad set of hospital-diagnosed outcomes for the outcome screen.

Exposures (Oxford WebQ, valid recalls only; same validity rule as the diet-proteome project):
  SSB  = fizzy drink (100170) + squash (100180)          servings/day, mean over valid recalls
  ASB  = low-calorie drink (100160)
  plus fruit juice, water, tea, coffee for later substitution work.
Time zero for intake analyses = date of the first valid recall (field 105010).
Outcomes: first hospital ICD-10 date (41270/41280), cancer registry for cancers, death date.
Outputs: cache/beverages.pkl, cache/icd_outcomes.pkl, cache/extra_cov.pkl
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
C = Path(os.environ.get("SCREEN_DIR", HERE / "cache"))
PC = Path(os.environ.get("UKB_EXTRACT_DIR", "../ukb_extracts"))
D = Path(os.environ.get("UKB_DATA_DIR", "../ukb_data"))
t0 = time.time()


def servings(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace({555: 0.5, 444: 0.25, 111: 0.5, 200: 2, 300: 3, 400: 4, 500: 5, 600: 6})
    return s.where(s >= 0)


def header(path):
    with open(path) as f:
        return [c.strip().strip('"') for c in f.readline().rstrip("\n").split(",")]


def read_cols(path, cols, dtype=str):
    hdr = header(path)
    use = [c for c in cols if c in hdr]
    return pd.concat(pd.read_csv(path, usecols=use, dtype=dtype, chunksize=50000), ignore_index=True)


# ---------------------------------------------------------------- beverages
BEV = {"SSB": [100170, 100180], "ASB": [100160], "juice": [100190, 100200, 100210],
       "water": [100150], "tea": [100400, 100410, 100420, 100430, 100440],
       "coffee": [100250, 100270, 100290, 100300, 100310, 100330]}
diet = pd.read_pickle(PC / "diet_raw.pkl")
keep = ["eid"] + [f"p26002_i{i}" for i in range(5)] + [f"p100026_i{i}" for i in range(5)]
diet = diet[[c for c in keep if c in diet.columns]]
pop = pd.read_pickle(PC / "pop.pkl")[["eid", "p31"]]
diet = diet.merge(pop, on="eid", how="left")
items = pd.read_pickle(PC / "items_raw.pkl").set_index("eid").reindex(diet["eid"]).reset_index()
dates = read_cols(D / "UKB619657_online_followup_2025Nov.csv", ["eid"] + [f"p105010_i{i}" for i in range(5)])
dates["eid"] = dates["eid"].astype(int)
dates = dates.set_index("eid").reindex(diet["eid"]).reset_index()
print("loaded diet/items/dates", f"{time.time()-t0:.0f}s", flush=True)

rows = []
for i in range(5):
    e = diet[f"p26002_i{i}"]
    kcal = e / 4.184
    hi = np.where(diet["p31"] == 1, 4200, 3500)
    ok = (e.notna() & diet[f"p100026_i{i}"].isna() & (kcal >= 500) & (kcal <= hi)).to_numpy()
    r = pd.DataFrame({"eid": diet["eid"], "inst": i,
                      "date": pd.to_datetime(dates[f"p105010_i{i}"], errors="coerce")})
    for k, fields in BEV.items():
        cols = [f"p{f}_i{i}" for f in fields if f"p{f}_i{i}" in items]
        r[k] = sum(servings(items[c]).fillna(0) for c in cols)
    rows.append(r[ok])
rec = pd.concat(rows, ignore_index=True)
bev = rec.groupby("eid")[list(BEV)].mean()
bev["n_valid_recalls"] = rec.groupby("eid").size()
bev["first_recall_date"] = rec.groupby("eid")["date"].min()
bev = bev.reset_index()
bev.to_pickle(C / "beverages.pkl")
print(bev.describe().round(2).T.to_string(), flush=True)
print("SSB consumers (>0):", (bev.SSB > 0).mean().round(3), " ASB consumers:", (bev.ASB > 0).mean().round(3))

# ---------------------------------------------------------------- outcomes (hospital ICD-10 + cancer registry)
ICD = {  # outcome: ICD-10 prefixes (no dot)
    # digestive
    "GERD": ["K21"], "Gallstones": ["K80"], "Cholecystitis": ["K81"], "IBD": ["K50", "K51"],
    "Crohn": ["K50"], "UC": ["K51"], "Acute_pancreatitis": ["K85"], "Chronic_pancreatitis": ["K860", "K861"],
    "Diverticular": ["K57"], "IBS": ["K58"], "PUD": ["K25", "K26", "K27"], "Gastritis": ["K29"],
    "Constipation": ["K590"], "Haemorrhoids": ["K64", "I84"], "Appendicitis": ["K35", "K36", "K37"],
    "MASLD": ["K760", "K758"], "Cirrhosis": ["K703", "K717", "K74"],
    # kidney / metabolic
    "T2D": ["E11"], "CKD": ["N18"], "Kidney_stones": ["N20", "N21", "N22", "N23"], "Gout": ["M10"],
    "Obesity_dx": ["E66"], "Hyperuricaemia": ["E790"],
    # cardiovascular
    "Hypertension": ["I10", "I11", "I12", "I13", "I15"], "CHD": ["I20", "I21", "I22", "I23", "I24", "I25"],
    "AF": ["I48"], "HF": ["I50"], "Stroke_isch": ["I63"], "Stroke_any": ["I60", "I61", "I62", "I63", "I64"],
    "VTE": ["I26", "I80", "I81", "I82"], "PAD": ["I70", "I739"],
    # neuro / psych
    "Dementia_icd": ["F00", "F01", "F02", "F03", "G30"], "PD_icd": ["G20"], "Depression_icd": ["F32", "F33"],
    "Anxiety_icd": ["F40", "F41"], "Sleep_apnoea": ["G473"], "Epilepsy": ["G40"],
    # musculoskeletal / other
    "Osteoporosis": ["M80", "M81"], "OA": ["M15", "M16", "M17", "M18", "M19"], "RA": ["M05", "M06"],
    "COPD": ["J43", "J44"], "Asthma": ["J45", "J46"], "Pneumonia": ["J12", "J13", "J14", "J15", "J16", "J17", "J18"],
    "Glaucoma": ["H40"], "AMD": ["H353"], "Cataract": ["H25", "H26"], "Psoriasis": ["L40"],
    "UTI": ["N390"], "Sepsis": ["A40", "A41"], "Anaemia": ["D50", "D51", "D52", "D53", "D62", "D63", "D64"],
    "Hearing_loss": ["H90", "H91"], "Periodontal": ["K05"], "Dental_caries": ["K02"],
}
CANCER = {"CRC": ["C18", "C19", "C20"], "Liver_cancer": ["C22"], "Pancreatic_cancer": ["C25"],
          "Kidney_cancer": ["C64"], "Breast_cancer": ["C50"], "Endometrial_cancer": ["C54"],
          "Oesophageal_cancer": ["C15"], "Gastric_cancer": ["C16"], "Any_cancer": ["C"]}

o = pd.read_pickle(PC / "outcomes.pkl")
o["eid"] = o["eid"].astype(int)
date_cols = sorted([c for c in o.columns if c.startswith("p41280_a")], key=lambda x: int(x.split("_a")[1]))
codes = o["p41270"].fillna("").str.split("|").explode().to_frame("code")
codes["pos"] = codes.groupby(level=0).cumcount()
codes = codes[codes["code"] != ""]
dmat = o[date_cols].apply(pd.to_datetime, errors="coerce")
dmat.columns = range(len(date_cols))
idx = codes.index.to_numpy()
pos = codes["pos"].to_numpy()
valid = pos < len(date_cols)
codes = codes[valid]
codes["date"] = dmat.to_numpy()[idx[valid], pos[valid]]
print("ICD rows:", len(codes), f"{time.time()-t0:.0f}s", flush=True)

ev = pd.DataFrame({"eid": o["eid"]})
for k, pref in ICD.items():
    hit = codes[codes["code"].str.startswith(tuple(pref))]
    ev[k] = hit.groupby(level=0)["date"].min().reindex(o.index)

can_rows = []
for c in [c for c in o.columns if c.startswith("p40006_i")]:
    inst = c.split("_i")[1]
    can_rows.append(pd.DataFrame({"code": o[c].fillna(""), "date": pd.to_datetime(o[f"p40005_i{inst}"], errors="coerce")},
                                 index=o.index))
can = pd.concat(can_rows)
can = can[can["code"] != ""]
for k, pref in CANCER.items():
    hit = can[can["code"].str.startswith(tuple(pref))]
    if k == "Any_cancer":
        hit = hit[~hit["code"].str.startswith("C44")]
    ev[k] = hit.groupby(level=0)["date"].min().reindex(o.index)
ev["death_date"] = pd.to_datetime(o["p40000_i0"], errors="coerce")
ev.to_pickle(C / "icd_outcomes.pkl")
print("outcomes", ev.shape, f"{time.time()-t0:.0f}s", flush=True)

# ---------------------------------------------------------------- extra covariates: diabetes, self-reported illness
ac = D / "UKB619657_assessment_center_2025Nov.csv"
hdr = header(ac)
cols = ["eid", "p2443_i0"] + [c for c in hdr if c.startswith("p20002_i0_a")]
x = read_cols(ac, cols)
x["eid"] = x["eid"].astype(int)
sr = x[[c for c in x.columns if c.startswith("p20002_i0_a")]]
out = pd.DataFrame({"eid": x["eid"], "diabetes_sr": pd.to_numeric(x["p2443_i0"], errors="coerce").eq(1).astype(int)})
SR = {"sr_diabetes": ["1220", "1222", "1223"], "sr_gallstones": ["1162"], "sr_kidney_stones": ["1197"],
      "sr_gout": ["1466"], "sr_ibd": ["1461", "1462", "1463"], "sr_gord": ["1138"], "sr_ckd": ["1192", "1193", "1194"],
      "sr_af": ["1471", "1483"], "sr_hf": ["1076"], "sr_chd": ["1074", "1075"], "sr_stroke": ["1081", "1583"],
      "sr_hypertension": ["1065", "1072"], "sr_osteoporosis": ["1309"], "sr_oa": ["1465"], "sr_copd": ["1112", "1113"],
      "sr_asthma": ["1111"], "sr_depression": ["1286"], "sr_pd": ["1262"], "sr_dementia": ["1263"],
      "sr_liver": ["1158", "1604"], "sr_pancreatitis": ["1164"], "sr_ibs": ["1154"],
      "sr_psoriasis": ["1453"], "sr_ra": ["1464"], "sr_glaucoma": ["1277"], "sr_cataract": ["1278"],
      "sr_diverticular": ["1458"], "sr_pud": ["1142", "1457"]}
for k, vals in SR.items():
    out[k] = sr.isin(vals).any(axis=1).astype(int)
out.to_pickle(C / "extra_cov.pkl")
print("extra covariates", out.shape, out.filter(like="sr_").mean().round(4).to_dict(), f"{time.time()-t0:.0f}s")
