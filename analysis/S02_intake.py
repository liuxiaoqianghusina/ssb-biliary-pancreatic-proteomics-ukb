"""Sweetened beverage intake and incident biliary-pancreatic disease in the whole 24h-recall cohort.

Time zero = first valid recall; participants with cancer before time zero excluded (as in Zhao et al., Nat Food 2025).
SSB and ASB (servings/day, truncated at 6) enter the same model (mutual adjustment).
  M1 = age, sex, energy, number of recalls, centre
  M2 = M1 + ethnicity, Townsend, education, smoking, alcohol frequency, recalled alcohol (g/day), physical activity,
       BMI, diabetes                                                               [main]
  M3 = M2 + triglycerides, HbA1c, CRP, GGT (possible mediators)
Outputs (results/): flow_intake.csv, intake_main.csv, intake_categories.csv, intake_spline.csv,
                    intake_sensitivity.csv, intake_subgroups.csv, intake_substitution.csv
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

from common import (BEVS, CACHE, NEGATIVE, POSITIVE, PRIMARY, RES, SECONDARY, SENS_OUT, add_biomarkers, base_covariates,
                    cox, extended, follow_up_end, model1, outcome_defs)

warnings.filterwarnings("ignore")
d = pd.read_pickle(CACHE / "analysis.pkl")
flow = {}
A = d[d.n_valid_recalls >= 1].copy()
flow["Participants with >=1 valid 24h recall"] = len(A)
A["start"] = A["first_recall_date"].fillna(A["baseline_date"])
A["age_start"] = A["age"] + (A["start"] - A["baseline_date"]).dt.days / 365.25
A = A[follow_up_end(A) > A["start"]]
flow["Alive and under follow-up at first recall"] = len(A)
A = A[~(A["Any_cancer"].notna() & (A["Any_cancer"] <= A["start"]))].reset_index(drop=True)
flow["No cancer before first recall (analytic cohort)"] = len(A)
pd.Series(flow).to_csv(RES / "flow_intake.csv", header=["n"])
print(pd.Series(flow).to_string(), flush=True)

for b, cap in [("SSB", 6), ("ASB", 6), ("juice", 6), ("water", 6), ("tea", 10), ("coffee", 10)]:
    A[b] = A[b].clip(upper=cap)
CAT_LABELS = ["0", ">0 to <1", ">=1"]
for b in BEVS:
    A[f"{b}_cat"] = pd.cut(A[b], [-0.01, 0, 0.999, 99], labels=CAT_LABELS)
    med = A.groupby(f"{b}_cat", observed=True)[b].median()
    A[f"{b}_med"] = A[f"{b}_cat"].map(med).astype(float)

Z1, Z2 = model1(A), base_covariates(A)
Z3 = add_biomarkers(Z2, A)
Z2e = Z2.join(extended(A))
Z2d = Z2.assign(hPDI=A["hPDI"].fillna(A["hPDI"].median()), fibre=A["fibre"].fillna(A["fibre"].median()),
                juice=A["juice"], tea=A["tea"], coffee=A["coffee"], water=A["water"])
cc = A[["townsend", "bmi", "met_min", "white", "degree", "smoking", "alcohol_freq"]].notna().all(axis=1)
prev_gs = (A["Gallstones"].notna() & (A["Gallstones"] <= A["start"])) | A["sr_gallstones"].fillna(0).eq(1)
XB = A[BEVS]
XC = pd.get_dummies(A[[f"{b}_cat" for b in BEVS]], drop_first=True, dtype=float)
XC.columns = [c.replace(">=1", "ge1").replace(">0 to <1", "lt1") for c in XC.columns]
XT = A[[f"{b}_med" for b in BEVS]]

OUTS = PRIMARY + SECONDARY + SENS_OUT + POSITIVE + NEGATIVE
FULL = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "Cholecystitis", "CP", "Pancreatitis_any"]
DEFS = outcome_defs(A, OUTS, A["start"])
DEFS_LAG = outcome_defs(A, FULL, A["start"], lag_years=2)
DEFS_CGS = outcome_defs(A, ["AP_nonbiliary", "CP"], A["start"], censor_at="Gallstones")


# ---------------------------------------------------------------- restricted cubic spline (Harrell)
def rcs_basis(x, knots):
    k = np.asarray(knots, float)
    x = np.asarray(x, float)
    cols = [x]
    norm = (k[-1] - k[0]) ** 2
    for j in range(len(k) - 2):
        t = (np.clip(x - k[j], 0, None) ** 3
             - np.clip(x - k[-2], 0, None) ** 3 * (k[-1] - k[j]) / (k[-1] - k[-2])
             + np.clip(x - k[-1], 0, None) ** 3 * (k[-2] - k[j]) / (k[-1] - k[-2])) / norm
        cols.append(t)
    return np.column_stack(cols)


KNOTS = [0, 0.5, 1, 2.5]
GRID = np.round(np.arange(0, 4.01, 0.1), 2)


def spline(name, elig, ev, T):
    rows, tests = [], []
    for b, other in [("SSB", "ASB"), ("ASB", "SSB")]:
        B = pd.DataFrame(rcs_basis(A[b], KNOTS), columns=[f"s{i}" for i in range(len(KNOTS) - 1)])
        X = B.assign(**{other: A[other].to_numpy()})
        res, m = cox(X, Z2, elig, ev, T, terms=list(B.columns), return_model=True)
        beta = m.params_[list(B.columns)].to_numpy()
        V = m.variance_matrix_.loc[list(B.columns), list(B.columns)].to_numpy()
        G = rcs_basis(GRID, KNOTS)
        lp, se = G @ beta, np.sqrt(np.einsum("ij,jk,ik->i", G, V, G))
        rows += [{"outcome": name, "exposure": b, "servings": g, "hr": np.exp(l), "lci": np.exp(l - 1.96 * s),
                  "uci": np.exp(l + 1.96 * s)} for g, l, s in zip(GRID, lp, se)]
        nl = beta[1:]
        w = float(nl @ np.linalg.solve(V[1:, 1:], nl))
        tests.append({"outcome": name, "exposure": b, "p_nonlinear": stats.chi2.sf(w, len(nl)),
                      "p_overall": stats.chi2.sf(float(beta @ np.linalg.solve(V, beta)), len(beta))})
    return rows, tests


# ---------------------------------------------------------------- subgroups
SUBGROUPS = {
    "Sex": (A["sex"].map({0: "Women", 1: "Men"})),
    "Age": np.where(A["age_start"] < 60, "<60 years", ">=60 years"),
    "BMI": pd.cut(A["bmi"], [0, 25, 30, 100], right=False, labels=["<25", "25 to <30", ">=30"]).astype(str),
    "Alcohol": np.where(A["alcohol_freq"].isin([1, 2, 3]), "Regular drinker",
                        np.where(A["alcohol_freq"].notna(), "Non-regular drinker", "nan")),
    "Smoking": np.where(A["smoking"] == 0, "Never", np.where(A["smoking"].isin([1, 2]), "Ever", "nan")),
    "Diabetes": np.where(A["diabetes_sr"] == 1, "Yes", "No"),
}


def subgroups(name, elig, ev, T):
    rows = []
    for g, lab in SUBGROUPS.items():
        lab = pd.Series(np.asarray(lab), index=A.index).astype(str)
        levels = [l for l in lab.unique() if l not in ("nan", "None")]
        for lv in sorted(levels):
            m = elig & (lab == lv)
            r = cox(XB, Z2, m, ev, T)
            for b in BEVS:
                rows.append({"outcome": name, "subgroup": g, "level": lv, "exposure": b, "n": r["n"],
                             "events": r["events"], **{k: r[b][k] for k in ("hr", "lci", "uci", "p")}})
        # interaction: likelihood-ratio test of exposure x subgroup terms
        ok = elig & lab.isin(levels)
        D = pd.get_dummies(lab[ok], drop_first=True, dtype=float)
        for b in BEVS:
            base_X = XB[ok].reset_index(drop=True)
            inter = D.mul(XB.loc[ok, b], axis=0).add_prefix(f"{b}_x_").reset_index(drop=True)
            Zg = Z2[ok].reset_index(drop=True).join(D.reset_index(drop=True).add_prefix("g_"))
            _, m0 = cox(base_X, Zg, np.ones(ok.sum(), bool), ev[ok], T[ok], return_model=True)
            _, m1 = cox(base_X.join(inter), Zg, np.ones(ok.sum(), bool), ev[ok], T[ok], terms=BEVS, return_model=True)
            lr = 2 * (m1.log_likelihood_ - m0.log_likelihood_)
            p = stats.chi2.sf(lr, inter.shape[1])
            for r_ in rows:
                if r_["subgroup"] == g and r_["exposure"] == b and r_["outcome"] == name:
                    r_["p_interaction"] = p
    return rows


# ---------------------------------------------------------------- substitution
def substitution(name, elig, ev, T):
    X = A[["SSB", "ASB", "water", "tea", "coffee", "juice"]]
    _, m = cox(X, Z2, elig, ev, T, return_model=True)
    b, V = m.params_, m.variance_matrix_
    rows = []
    for src, alts in [("SSB", ["ASB", "water", "tea", "coffee", "juice"]), ("ASB", ["water", "tea", "coffee", "juice"])]:
        for alt in alts:
            est = b[alt] - b[src]
            se = np.sqrt(V.at[alt, alt] + V.at[src, src] - 2 * V.at[alt, src])
            rows.append({"outcome": name, "replace": src, "with": alt, "hr": np.exp(est), "lci": np.exp(est - 1.96 * se),
                         "uci": np.exp(est + 1.96 * se), "p": 2 * stats.norm.sf(abs(est / se))})
    return rows


# ---------------------------------------------------------------- per-outcome worker
def row(name, analysis, r, terms=BEVS):
    out = {"outcome": name, "analysis": analysis, "n": r["n"], "events": r["events"]}
    for t in terms:
        if t in r:
            out.update({f"{t}_{k}": r[t][k] for k in ("hr", "lci", "uci", "p")})
    return out


def person_years(name, elig, ev, T):
    rows = []
    for b in BEVS:
        for lv in CAT_LABELS:
            m = elig & (A[f"{b}_cat"] == lv)
            rows.append({"outcome": name, "exposure": b, "category": lv, "n": int(m.sum()), "events": int(ev[m].sum()),
                         "person_years": float(T[m].sum())})
    return rows


def run(name):
    out = {k: [] for k in ("main", "cat", "py", "spline", "spline_test", "sens", "sub", "subst")}
    elig, ev, T = DEFS[name]
    for lab, Z in [("M1", Z1), ("M2", Z2), ("M3", Z3)]:
        out["main"].append(row(name, lab, cox(XB, Z, elig, ev, T)))
    rc = cox(XC, Z2, elig, ev, T)
    rt = cox(XT, Z2, elig, ev, T)
    for b in BEVS:
        for c in ("lt1", "ge1"):
            k = f"{b}_cat_{c}"
            out["cat"].append({"outcome": name, "exposure": b, "category": c, **{x: rc[k][x] for x in ("hr", "lci", "uci", "p")},
                               "p_trend": rt[f"{b}_med"]["p"]})
    out["py"] = person_years(name, elig, ev, T)
    if name not in FULL:
        return out
    r_, t_ = spline(name, elig, ev, T)
    out["spline"], out["spline_test"] = r_, t_
    S = out["sens"]
    S.append(row(name, "Main (M2, mutually adjusted)", cox(XB, Z2, elig, ev, T)))
    for b in BEVS:
        S.append(row(name, f"{b} alone (not mutually adjusted)", cox(A[[b]], Z2, elig, ev, T), [b]))
    el, e_l, T_l = DEFS_LAG[name]
    S.append(row(name, "Excluding the first 2 years of follow-up", cox(XB, Z2, el, e_l, T_l)))
    S.append(row(name, ">=2 valid 24h recalls", cox(XB, Z2, elig & (A["n_valid_recalls"] >= 2), ev, T)))
    S.append(row(name, "Excluding baseline diabetes", cox(XB, Z2, elig & A["diabetes_sr"].fillna(0).eq(0), ev, T)))
    S.append(row(name, "Excluding BMI >=40 kg/m2", cox(XB, Z2, elig & ~(A["bmi"] >= 40), ev, T)))
    S.append(row(name, "+ gallstone-related confounders", cox(XB, Z2e, elig, ev, T)))
    S.append(row(name, "+ diet quality (hPDI, fibre) and other beverages", cox(XB, Z2d, elig, ev, T)))
    S.append(row(name, "Complete cases", cox(XB, Z2, elig & cc, ev, T)))
    if name != "Gallstones" and name != "Cholecystitis":
        S.append(row(name, "Excluding prevalent gallstones", cox(XB, Z2, elig & ~prev_gs, ev, T)))
    if name in DEFS_CGS:
        eg, e_g, T_g = DEFS_CGS[name]
        S.append(row(name, "Censored at incident gallstones", cox(XB, Z2, eg, e_g, T_g)))
    if name in ("Gallstones", "AP", "AP_nonbiliary", "AP_biliary", "CP"):
        out["sub"] = subgroups(name, elig, ev, T)
        out["subst"] = substitution(name, elig, ev, T)
    return out


res = Parallel(n_jobs=10)(delayed(run)(o) for o in OUTS)
files = {"main": "intake_main", "cat": "intake_categories", "py": "intake_person_years", "spline": "intake_spline",
         "spline_test": "intake_spline_tests", "sens": "intake_sensitivity", "sub": "intake_subgroups",
         "subst": "intake_substitution"}
for k, f in files.items():
    pd.DataFrame([r for o in res for r in o[k]]).to_csv(RES / f"{f}.csv", index=False)

pd.set_option("display.width", 250)
m = pd.read_csv(RES / "intake_main.csv")
fmtc = lambda df, b: df[f"{b}_hr"].map("{:.2f}".format) + " (" + df[f"{b}_lci"].map("{:.2f}".format) + "-" + \
    df[f"{b}_uci"].map("{:.2f}".format) + ")"
for b in BEVS:
    m[b] = fmtc(m, b) + np.where(m[f"{b}_p"] < 0.05, "*", "")
print(m.pivot(index="outcome", columns="analysis", values=["SSB", "ASB"]).reindex(OUTS).to_string())
c = pd.read_csv(RES / "intake_categories.csv")
c["est"] = c["hr"].map("{:.2f}".format) + " (" + c["lci"].map("{:.2f}".format) + "-" + c["uci"].map("{:.2f}".format) + ")"
print(c[c.outcome.isin(FULL)].pivot_table(index=["outcome", "exposure"], columns="category", values="est", aggfunc="first")
      .join(c.groupby(["outcome", "exposure"])["p_trend"].first()).to_string())
print(pd.read_csv(RES / "intake_spline_tests.csv").round(4).to_string(index=False))
