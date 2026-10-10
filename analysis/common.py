"""Shared paths, covariate builders and model helpers: sweetened beverages, plasma proteome and biliary-pancreatic disease."""
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from scipy import stats

# Directories are configured with environment variables (see README):
#   SSB_PROJECT_DIR   this project (cache/, results/, figures/, manuscript/ are created inside)
#   DIET_PROJECT_DIR  companion diet-proteome project (person.pkl, plate-adjusted Olink matrix, signature split, BMI signature)
#   UKB_EXTRACT_DIR   cached raw UK Biobank extracts (diet recalls, items, outcomes)
#   SCREEN_DIR        cache written by data_prep/ (beverages, outcomes, covariates)
#   UKB_DATA_DIR      UK Biobank dataset files (CSV extracts and data dictionary)
PROJ = Path(os.environ.get("SSB_PROJECT_DIR", "."))
CACHE, RES, FIG = PROJ / "cache", PROJ / "results", PROJ / "figures"
for _d in (CACHE, RES, FIG):
    _d.mkdir(exist_ok=True)
DIETPROJ = Path(os.environ.get("DIET_PROJECT_DIR", "../diet-proteome"))
PILOT = Path(os.environ.get("UKB_EXTRACT_DIR", "../ukb_extracts"))
SCREEN = Path(os.environ.get("SCREEN_DIR", "./data_prep_cache"))
UKB = Path(os.environ.get("UKB_DATA_DIR", "../ukb_data"))
CENSOR = pd.Timestamp("2022-12-31")      # hospital inpatient records incomplete from 2023
SEED = 2026
WINDOW = pd.Timedelta(days=90)           # gallstone diagnosis up to 90 days after AP -> gallstone-related AP

BEVS = ["SSB", "ASB"]
PRIMARY = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary"]
SECONDARY = ["Cholecystitis", "CP", "Pancreatitis_any", "AP_alcohol", "Pancreatic_cancer"]
POSITIVE = ["MASLD", "Cirrhosis", "Dental_caries"]
NEGATIVE = ["NC_cataract", "NC_inguinal_hernia", "NC_hearing_loss", "NC_carpal_tunnel", "NC_varicose_veins"]
SENS_OUT = ["GS_hosp", "GS_or_chole", "Cholecystectomy", "AP_hosp", "CP_fo"]
GB = ["Gallstones", "GS_hosp", "GS_or_chole", "Cholecystectomy", "Cholecystitis"]   # need an intact gallbladder
OUT_LABEL = {
    "Gallstones": "Gallstone disease", "GS_hosp": "Gallstone disease (hospital records only)",
    "GS_or_chole": "Gallstone disease or cholecystectomy", "Cholecystectomy": "Cholecystectomy",
    "Cholecystitis": "Cholecystitis",
    "AP": "Acute pancreatitis", "AP_nonbiliary": "Non-gallstone acute pancreatitis",
    "AP_biliary": "Gallstone-related acute pancreatitis", "AP_alcohol": "Alcohol-coded acute pancreatitis (K85.2)",
    "CP": "Chronic pancreatitis", "Pancreatitis_any": "Any pancreatitis", "Pancreatic_cancer": "Pancreatic cancer",
    "AP_hosp": "Acute pancreatitis (hospital records only)", "CP_fo": "Other pancreatic disease (First Occurrence K86)",
    "MASLD": "MASLD", "Cirrhosis": "Liver fibrosis/cirrhosis", "Dental_caries": "Dental caries",
    "NC_cataract": "Cataract", "NC_inguinal_hernia": "Inguinal hernia", "NC_hearing_loss": "Hearing loss",
    "NC_carpal_tunnel": "Upper-limb mononeuropathy", "NC_varicose_veins": "Varicose veins"}
# self-reported conditions at baseline that define prevalent cases (extra_cov.pkl flags)
SR = {"Gallstones": "sr_gallstones", "GS_hosp": "sr_gallstones", "GS_or_chole": "sr_gallstones",
      "Cholecystectomy": "sr_gallstones", "Cholecystitis": "sr_gallstones", "AP": "sr_pancreatitis", "AP_nonbiliary": "sr_pancreatitis", "AP_biliary": "sr_pancreatitis",
      "AP_alcohol": "sr_pancreatitis", "CP": "sr_pancreatitis", "Pancreatitis_any": "sr_pancreatitis",
      "AP_hosp": "sr_pancreatitis", "CP_fo": "sr_pancreatitis", "MASLD": "sr_liver", "Cirrhosis": "sr_liver",
      "NC_cataract": "sr_cataract"}


def base_covariates(d, diet=True, alcohol_g=True):
    """Age, sex, ethnicity, Townsend, education, smoking, alcohol frequency, physical activity, BMI, diabetes,
    assessment centre (+ energy, number of recalls and recalled alcohol g/day when diet=True;
    + fasting time when diet=False, i.e. for blood-based models)."""
    X = pd.DataFrame(index=d.index)
    cont = ["age_start" if "age_start" in d else "age", "townsend", "bmi"]
    cont += (["energy_kj", "n_recalls"] + (["alcohol_g"] if alcohol_g else [])) if diet else ["fasting_h"]
    for c in cont:
        X[c] = d[c].fillna(d[c].median())
    lm = np.log1p(d["met_min"])
    X["log_met"], X["met_missing"] = lm.fillna(lm.median()), d["met_min"].isna().astype(float)
    for c in ["sex", "white", "degree"]:
        X[c] = d[c].fillna(-1)
    X["diabetes"] = d["diabetes_sr"].fillna(0)
    for c in ["smoking", "alcohol_freq", "centre"]:
        X = X.join(pd.get_dummies(d[c].fillna(-1).astype(int), prefix=c, drop_first=True, dtype=float))
    return X.astype(float)


def model1(d, diet=True):
    X = pd.DataFrame(index=d.index)
    X["age"] = d["age_start" if "age_start" in d else "age"]
    X["sex"] = d["sex"]
    if diet:
        X["energy_kj"], X["n_recalls"] = d["energy_kj"], d["n_recalls"]
    X = X.join(pd.get_dummies(d["centre"].fillna(-1).astype(int), prefix="centre", drop_first=True, dtype=float))
    return X.fillna(X.median()).astype(float)


def add_biomarkers(X, d):
    """Metabolic/inflammatory markers on the SSB/ASB-pancreatitis pathway: triglycerides, HbA1c, CRP, GGT."""
    X = X.copy()
    for c, f in [("tg", np.log), ("hba1c", None), ("crp", np.log), ("ggt", np.log)]:
        v = d[c].clip(lower=1e-3)
        v = f(v) if f else v
        X[c] = v.fillna(v.median()).to_numpy()
    return X


def add_kidney(X, d):
    X = X.copy()
    for c in ["cysc", "creat"]:
        v = np.log(d[c].clip(lower=1e-3))
        X[f"log_{c}"] = v.fillna(v.median()).to_numpy()
    return X


def extended(d):
    """Gallstone-related confounders: HRT, oral contraceptives, parity (women; 0 for men, absorbed by sex),
    cholesterol/blood-pressure medication, insulin, weight change in the past year."""
    num = lambda s: pd.to_numeric(s, errors="coerce")
    has = lambda s, c: s.fillna("").astype(str).str.contains(rf"(?:^|\D){c}(?:\D|$)")
    X = pd.DataFrame(index=d.index)
    women = d["sex"] == 0
    for name, col in [("hrt", "p2814_i0"), ("ocp", "p2784_i0")]:
        v = num(d[col]).where(lambda s: s.isin([0, 1]))
        X[name] = np.where(women, v.fillna(-1), 0)
    births = num(d["p2734_i0"]).where(lambda s: s >= 0).clip(upper=3)
    X["parity"] = np.where(women, births.fillna(-1), 0)
    med = d["p6153_i0"].where(women, d["p6177_i0"])
    X["chol_med"], X["bp_med"], X["insulin"] = [has(med, c).astype(float) for c in (1, 2, 3)]
    X["weight_change"] = num(d["p2306_i0"]).where(lambda s: s.isin([0, 2, 3])).fillna(-1)
    out = pd.DataFrame(index=d.index)
    for c in ["hrt", "ocp", "parity", "weight_change"]:
        out = out.join(pd.get_dummies(X[c].astype(int), prefix=c, drop_first=True, dtype=float))
    return out.join(X[["chol_med", "bp_med", "insulin"]]).astype(float)


def follow_up_end(d):
    return pd.concat([d["death_date"], d["lost_date"]], axis=1).min(axis=1).clip(upper=CENSOR).fillna(CENSOR)


def outcome_defs(d, names, start, lag_years=0, censor_at=None):
    """{name: (eligible mask, event indicator, time in years)}; prevalent cases (date <= start or self-report) excluded.
    Gallstone-related vs non-gallstone AP: gallstone record (FO K80, hospital K80) on/before AP + 90 days, or K85.1.
    censor_at: column of dates at which follow-up is censored (e.g. incident gallstones)."""
    end = follow_up_end(d)
    if censor_at is not None:
        c = d[censor_at]
        end = end.where(~(c.notna() & (c > start) & (c < end)), c)
    out = {}
    for name in names:
        if name in ("AP_nonbiliary", "AP_biliary"):
            ap = d["AP"]
            prev = ap.notna() & (ap <= start)
            inc = ap.notna() & (ap > start) & (ap <= end)
            biliary = inc & (d["K851"].notna() | (d["Gallstones"].notna() & (d["Gallstones"] <= ap + WINDOW)))
            ev = biliary if name == "AP_biliary" else inc & ~biliary
            stop = ap.where(ev, end)
        else:
            s = d[name]
            prev = s.notna() & (s <= start)
            ev = s.notna() & (s > start) & (s <= end)
            stop = s.where(ev, end)
        if name in SR:
            prev = prev | d[SR[name]].fillna(0).eq(1)
        if name in GB:
            prev = prev | (d["cholecystectomy_date"].notna() & (d["cholecystectomy_date"] <= start))
        T = (stop - start).dt.days / 365.25
        elig = ~prev & (T > 0)
        if lag_years:
            early = ev & (T <= lag_years)
            elig = elig & ~early
        out[name] = (elig, ev.astype(int), T)
    return out


def cox(X, Z, elig, ev, T, terms=None, strata=None, return_model=False):
    """Cox model of exposure columns X (DataFrame) adjusted for Z; returns {term: (hr, lci, uci, p, logHR, se)}."""
    terms = terms or list(X.columns)
    f = pd.concat([X.reset_index(drop=True), Z.reset_index(drop=True)], axis=1)
    f["T"], f["E"] = np.asarray(T), np.asarray(ev)
    m = np.asarray(elig) & np.isfinite(f[list(X.columns)]).all(axis=1).to_numpy()
    f = f[m]
    f = f.loc[:, (f.std() > 0) | f.columns.isin(["T", "E"])]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cph = CoxPHFitter(penalizer=1e-4).fit(f, "T", "E")
    if any("failed to converge" in str(x.message) for x in w):     # Newton-Raphson overshoot: refit with smaller steps
        cph = CoxPHFitter(penalizer=1e-4).fit(f, "T", "E", fit_options={"step_size": 0.2, "max_steps": 500})
    s = cph.summary
    res = {"n": int(len(f)), "events": int(f["E"].sum())}
    for t in terms:
        res[t] = dict(hr=np.exp(s.at[t, "coef"]), lci=np.exp(s.at[t, "coef lower 95%"]),
                      uci=np.exp(s.at[t, "coef upper 95%"]), p=s.at[t, "p"], logHR=s.at[t, "coef"], se=s.at[t, "se(coef)"])
    if return_model:
        return res, cph
    return res


def ols_many(x, Z, Y):
    X = np.column_stack([np.ones(len(x)), x, Z])
    XtX_inv = np.linalg.pinv(X.T @ X)
    B = XtX_inv @ X.T @ Y
    resid = Y - X @ B
    dfree = X.shape[0] - np.linalg.matrix_rank(X)
    se = np.sqrt((resid ** 2).sum(axis=0) / dfree * XtX_inv[1, 1])
    p = 2 * stats.t.sf(np.abs(B[1] / se), dfree)
    return B[1], se, p


def fmt(r):
    return f"{r['hr']:.2f} ({r['lci']:.2f}–{r['uci']:.2f})"


def fmt_p(p):
    if p < 0.001:
        return f"{p:.1e}"
    return f"{p:.3f}"
