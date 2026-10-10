"""Person-level analysis table.

Exposures: SSB (fizzy drink 100170 + squash 100180) and ASB (low-calorie drink 100160), servings/day averaged over
valid Oxford WebQ recalls (built in the screening folder, 01_build.py); other beverages for substitution.
Recalled alcohol (26030, g/day) and total sugars (26011, g/day) are averaged over the same valid recalls.
Outcomes:
  AP       First Occurrence K85 (primary care, hospital, self-report, death)        [primary]
  AP_hosp  hospital K85 only                                                         [sensitivity]
  K851     hospital K85.1 (biliary AP code) - used with gallstone records to split gallstone-related vs other AP
  AP_alcohol hospital K85.2
  CP       hospital K86.0/K86.1 (chronic pancreatitis)                               [secondary]
  CP_fo    First Occurrence K86 (other diseases of pancreas)                         [sensitivity]
  Pancreatic cancer (registry C25); MASLD, cirrhosis, dental caries (positive controls); 5 negative-control outcomes.
  Gallstones First Occurrence K80 [primary]; GS_hosp (hospital K80), GS_or_chole (K80 or OPCS J18 cholecystectomy),
  Cholecystectomy (J18) [sensitivity]; Cholecystitis (First Occurrence K81) [secondary].
Gallstone-related confounders (HRT, OCP, parity, medications, weight change) come from the diet-proteome project
(F10a_extract_extra.py, cache/extra_tier1.pkl).
Output: cache/analysis.pkl
"""
import numpy as np
import pandas as pd

from common import CACHE, DIETPROJ, PILOT, SCREEN

keep = ["eid", "baseline_date", "age", "sex", "white", "townsend", "bmi", "smoking", "alcohol_freq", "met_min",
        "degree", "fasting_h", "centre", "lost_date", "energy_kj", "n_recalls", "hPDI", "uPDI", "UPF", "fibre",
        "alt", "ast", "creat", "crp", "cysc", "hba1c", "waist", "ggt", "tg", "olink_qc", "consortium",
        "Gallstones", "NC_cataract", "NC_inguinal_hernia", "NC_hearing_loss", "NC_carpal_tunnel", "NC_varicose_veins"]
d = pd.read_pickle(DIETPROJ / "cache" / "person.pkl")[keep]

bev = pd.read_pickle(SCREEN / "beverages.pkl")
fo = pd.read_pickle(SCREEN / "fo_outcomes.pkl").rename(columns={"Acute_pancreatitis": "AP", "Chronic_pancreatitis": "CP_fo"})
hosp = pd.read_pickle(SCREEN / "icd_outcomes.pkl")[["eid", "Acute_pancreatitis", "Gallstones"]].rename(
    columns={"Acute_pancreatitis": "AP_hosp", "Gallstones": "GS_hosp"})
d = (d.merge(bev, on="eid", how="left")
     .merge(fo[["eid", "AP", "CP_fo", "Cholecystitis", "MASLD", "Cirrhosis", "Dental_caries", "Pancreatic_cancer",
                "Any_cancer", "death_date"]], on="eid", how="left")
     .merge(hosp, on="eid", how="left")
     .merge(pd.read_pickle(SCREEN / "extra_cov.pkl"), on="eid", how="left")
     .merge(pd.read_pickle(SCREEN / "urate_ppi.pkl"), on="eid", how="left")
     .merge(pd.read_pickle(DIETPROJ / "cache" / "extra_tier1.pkl").drop(columns=["p53_i2", "p53_i3"]), on="eid", how="left"))
# gallstone definitions: First Occurrence K80 (main), K80 or cholecystectomy (OPCS J18), hospital K80, cholecystectomy
d["Cholecystectomy"] = d["cholecystectomy_date"]
d["GS_or_chole"] = d[["Gallstones", "cholecystectomy_date"]].min(axis=1)

# ---------------------------------------------------------------- hospital 4-character pancreatitis codes
o = pd.read_pickle(PILOT / "outcomes.pkl")
o["eid"] = o["eid"].astype(int)
date_cols = sorted([c for c in o.columns if c.startswith("p41280_a")], key=lambda x: int(x.split("_a")[1]))
codes = o["p41270"].fillna("").str.split("|").explode().to_frame("code")
codes["pos"] = codes.groupby(level=0).cumcount()
codes = codes[codes["code"].str.match(r"K8[56]") & (codes["pos"] < len(date_cols))]
dmat = o[date_cols].to_numpy()
codes["date"] = pd.to_datetime(pd.Series(dmat[codes.index.to_numpy(), codes["pos"].to_numpy()]), errors="coerce").to_numpy()
codes["eid"] = o["eid"].to_numpy()[codes.index.to_numpy()]
del o, dmat


def first(prefixes, name):
    h = codes[codes["code"].str.startswith(tuple(prefixes))]
    return h.groupby("eid")["date"].min().rename(name)


hc = pd.concat([first(["K851"], "K851"), first(["K852"], "AP_alcohol"), first(["K860", "K861"], "CP")], axis=1)
d = d.merge(hc.reset_index().rename(columns={"index": "eid"}), on="eid", how="left")
d["Pancreatitis_any"] = d[["AP", "CP"]].min(axis=1)
print("hospital codes:", hc.notna().sum().to_dict())

# ---------------------------------------------------------------- recalled alcohol and total sugars (valid recalls)
raw = pd.read_pickle(PILOT / "diet_raw.pkl")
cols = ["eid"] + [f"p{f}_i{i}" for f in (26002, 26030, 26011, 100026) for i in range(5)]
raw = raw[[c for c in cols if c in raw.columns]].merge(d[["eid", "sex"]], on="eid", how="left")
acc = {k: [] for k in ("alcohol_g", "sugars_g")}
valid_all = []
for i in range(5):
    e = raw[f"p26002_i{i}"]
    kcal = e / 4.184
    hi = np.where(raw["sex"] == 1, 4200, 3500)
    ok = e.notna() & raw[f"p100026_i{i}"].isna() & (kcal >= 500) & (kcal <= hi)
    acc["alcohol_g"].append(raw[f"p26030_i{i}"].where(ok))
    acc["sugars_g"].append(raw[f"p26011_i{i}"].where(ok))
nut = pd.DataFrame({"eid": raw["eid"], **{k: pd.concat(v, axis=1).mean(axis=1) for k, v in acc.items()}})
d = d.merge(nut, on="eid", how="left")
del raw

d.to_pickle(CACHE / "analysis.pkl")
r = d[d.n_valid_recalls >= 1]
print("rows", d.shape, "with recall", len(r))
print(r[["SSB", "ASB", "juice", "water", "tea", "coffee", "alcohol_g", "sugars_g"]].describe().round(2).T.to_string())
print("events ever recorded:", d[["AP", "AP_hosp", "K851", "AP_alcohol", "CP", "CP_fo", "Pancreatitis_any",
                                  "Pancreatic_cancer", "Gallstones", "GS_hosp", "GS_or_chole",
                                  "Cholecystectomy", "Cholecystitis"]].notna().sum().to_dict())
