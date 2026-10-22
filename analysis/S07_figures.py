"""Main and supplementary figures (Arial throughout, italic P, SVG/PDF/PNG/TIFF at 600 dpi).
Fig 1 study design; Fig 2 intake -> outcomes (forest + splines); Fig 3 signatures -> outcomes (M1-M3);
Fig 4 pathway split (mutually adjusted signatures + individual proteins); Fig 5 abdominal MRI.
Fig S1 subgroups; Fig S2 substitution; Fig S3 biomarkers; Fig S4 beverage-proteome (volcano + enrichment);
Fig S5 signature tertiles.
"""
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch

from common import FIG, NEGATIVE, OUT_LABEL, RES

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold", "font.size": 7,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
RED, BLUE, GREY, DARK = "#c0392b", "#2c6fbb", "#bdbdbd", "#333333"
MODEL_COL = {"M1": "#9e9e9e", "M2": "#1f1f1f", "M3": "#e08214"}
SPECTRUM = ["Gallstones", "Cholecystitis", "AP", "AP_biliary", "AP_nonbiliary", "CP", "Pancreatitis_any"]
CONTROLS = ["MASLD", "Cirrhosis", "Dental_caries"] + NEGATIVE
SHORT = {**OUT_LABEL, "AP_biliary": "Gallstone-related AP", "AP_nonbiliary": "Non-gallstone AP", "AP": "Acute pancreatitis (AP)",
         "CP": "Chronic pancreatitis", "Pancreatitis_any": "Any pancreatitis", "Gallstones": "Gallstone disease",
         "NC_carpal_tunnel": "Upper-limb mononeuropathy"}


# FRONTIERS=1 writes Frontiers-style main figures (bold "(A)" panel labels, width <=180 mm) to FIG_DIR;
# FIG_MAX_MM caps the final width for other journals (Food & Function: 170); UNIT_STYLE=rsc writes "per day"
FRONTIERS = os.environ.get("FRONTIERS") == "1"
MAX_MM = 180.0 if FRONTIERS else (float(os.environ["FIG_MAX_MM"]) if os.environ.get("FIG_MAX_MM") else None)
RSC_UNITS = os.environ.get("UNIT_STYLE") == "rsc"
OUT = Path(os.environ["FIG_DIR"]) if os.environ.get("FIG_DIR") else FIG
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    if RSC_UNITS:                              # RSC style: "servings per day" rather than "servings/day"
        for t in fig.findobj(matplotlib.text.Text):
            if "/day" in t.get_text():
                t.set_text(t.get_text().replace("per serving/day", "per daily serving").replace("/day", " per day"))
    if MAX_MM:                                 # final (tight) width must not exceed MAX_MM
        for _ in range(6):
            fig.canvas.draw()
            tw = fig.get_tightbbox(fig.canvas.get_renderer()).width
            if tw <= (MAX_MM - 6) / 25.4:      # savefig adds 0.1-inch padding on each side
                break
            w, h = fig.get_size_inches()
            k = ((MAX_MM - 7) / 25.4) / tw
            fig.set_size_inches(w * k, h * k)
    for ext in ["svg", "pdf", "png", "tiff"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=600, bbox_inches="tight")
    plt.close(fig)


def tag(ax, t, x=-0.02, y=1.02):
    if FRONTIERS:
        t = f"({t.upper()})"
    ax.text(x, y, t, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


def logx(ax, lo, hi, ticks):
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(mt.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mt.FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_locator(mt.NullLocator())
    ax.axvline(1, color=GREY, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)


def forest_rows(ax, rows, labels, series, lo, hi, ticks, xlabel, sep_after=()):
    """rows: list of outcome keys; series: [(name, color, offset, {outcome: (hr, lci, uci)})]."""
    y = np.arange(len(rows))[::-1].astype(float)
    for name, col, off, est in series:
        for yi, r in zip(y, rows):
            if r not in est or not np.isfinite(est[r][0]):
                continue
            h, l, u = est[r]
            ax.plot([max(l, lo), min(u, hi)], [yi + off] * 2, color=col, lw=0.9)
            ax.plot(h, yi + off, "s", color=col, ms=3.2, mec="none")
        ax.plot([], [], "s-", color=col, ms=3.2, lw=0.9, label=name)
    for k in sep_after:
        ax.axhline(y[k] - 0.5, color=GREY, lw=0.4, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    logx(ax, lo, hi, ticks)
    ax.set_xlabel(xlabel)


# ---------------------------------------------------------------- Fig 1 design
def fig1():
    fi = pd.read_csv(RES / "flow_intake.csv", index_col=0)["n"]
    fs = pd.read_csv(RES / "flow_signature.csv", index_col=0)["n"]
    perf = pd.read_csv(RES / "signature_performance.csv").set_index("signature")
    mf = pd.read_csv(RES / "mri_flow.csv", index_col=0)["n"]
    n_int = int(fi.iloc[-1])
    mi = mf[mf.index.str.startswith("intake sample")]
    ms = mf[mf.index.str.startswith("signature sample (Independent sample)")]
    fig, ax = plt.subplots(figsize=(7.2, 5.45))
    ax.set_xlim(0, 100); ax.set_ylim(0, 77); ax.axis("off")

    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color, ec="#4a4a4a", lw=0.6))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top", fontsize=7.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 6.2, body, ha="center", va="top", fontsize=6.2, linespacing=1.35)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.7, color="#4a4a4a"))

    box(1, 56, 29, 20, "Beverage intake",
        f"UK Biobank Oxford WebQ 24-h recalls\n(2009–2012), analytic n = {n_int:,}\n"
        "SSB: fizzy drinks + squash\nASB: low-calorie drinks\n(servings/day, mutually adjusted)", "#fdecea")
    box(36, 56, 28, 20, "Beverage–proteome sample",
        f"Randomly selected Olink participants\nwith ≥1 valid recall, n = 17,183\n"
        "2,919 plasma proteins\nprotein-wise linear models\nelastic-net signatures", "#fdf3e7")
    box(70, 56, 29, 20, "Proteomic signatures",
        f"Training 70% (n = {int(perf.loc['sig_SSB', 'n_train']):,})\nTesting 30% (n = {int(perf.loc['sig_SSB', 'n_test']):,})\n"
        f"SSB: {int(perf.loc['sig_SSB', 'n_proteins'])} proteins, r = {perf.loc['sig_SSB', 'pearson_r_test']:.2f}\n"
        f"ASB: {int(perf.loc['sig_ASB', 'n_proteins'])} proteins, r = {perf.loc['sig_ASB', 'pearson_r_test']:.2f}", "#eef6ec")
    box(1, 28, 29, 23, "Intake → disease",
        f"n = {n_int:,}; time zero = first recall\nM1: age, sex, energy, recalls, centre\n"
        "M2: + sociodemographic, lifestyle,\nalcohol, BMI, diabetes\nM3: + TG, HbA1c, CRP, GGT", "#fdecea")
    box(36, 28, 28, 23, "Biliary–pancreatic outcomes",
        "Gallstone disease, cholecystitis\nAcute pancreatitis (AP):\ngallstone-related vs non-gallstone\nChronic pancreatitis\n"
        "Positive controls: MASLD, cirrhosis,\ndental caries; 5 negative controls", "#f3eef8")
    box(70, 28, 29, 23, "Signature → disease",
        f"Independent Olink sample\nn = {int(fs.iloc[3]):,} (training set excluded)\n"
        "M1: + covariates, BMI, diabetes\nM2: + TG, HbA1c, CRP, GGT, kidney\nM3: + BMI proteomic signature", "#eef6ec")
    box(1, 2, 29, 20, "Intake → organ fat",
        f"Recall participants with MRI\nn = {int(mi.min()):,}–{int(mi.max()):,}\n"
        "M2 + years to imaging, imaging centre\n± BMI at the imaging visit\nSSB and ASB mutually adjusted", "#fdecea")
    box(36, 2, 28, 20, "Abdominal MRI (2014–2020)",
        "Pancreas and liver fat (PDFF)\nVisceral fat volume\nAbdominal subcutaneous fat volume\nPancreas volume\n"
        "Linear regression", "#e8f1f8")
    box(70, 2, 29, 20, "Signature → organ fat",
        f"Independent Olink sample with MRI\nn = {int(ms.min()):,}–{int(ms.max()):,}\n"
        "M1 + years to imaging, imaging centre\n± BMI at the imaging visit\neach signature alone and together", "#eef6ec")
    arrow(30.5, 66, 35.5, 66); arrow(64.5, 66, 69.5, 66)
    arrow(15.5, 55.5, 15.5, 51.5); arrow(84.5, 55.5, 84.5, 51.5)
    arrow(30.5, 39.5, 35.5, 39.5); arrow(69.5, 39.5, 64.5, 39.5)
    arrow(15.5, 27.5, 15.5, 22.5); arrow(84.5, 27.5, 84.5, 22.5)
    arrow(30.5, 12, 35.5, 12); arrow(69.5, 12, 64.5, 12)
    save(fig, "Fig1_design")


# ---------------------------------------------------------------- Fig 2 intake
def fig2():
    m = pd.read_csv(RES / "intake_main.csv")
    m = m[m.analysis == "M2"].set_index("outcome")
    sp = pd.read_csv(RES / "intake_spline.csv")
    st = pd.read_csv(RES / "intake_spline_tests.csv").set_index(["outcome", "exposure"])
    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.55, 1, 1], wspace=0.45, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    rows = SPECTRUM + CONTROLS
    labels = [SHORT[r] + f" ({int(m.at[r, 'events']):,})" for r in rows]
    series = [(b, c, o, {r: (m.at[r, f"{b}_hr"], m.at[r, f"{b}_lci"], m.at[r, f"{b}_uci"]) for r in rows})
              for b, c, o in [("SSB", RED, 0.17), ("ASB", BLUE, -0.17)]]
    forest_rows(ax, rows, labels, series, 0.85, 1.5, [0.9, 1, 1.2, 1.4], "HR per serving/day (95% CI)",
                sep_after=(len(SPECTRUM) - 1, len(SPECTRUM) + 2))
    ax.legend(frameon=False, loc="lower right", fontsize=6.5)
    ax.text(1.47, len(rows) - 1 - len(SPECTRUM) + 0.25, "Controls", ha="right", fontsize=6.3, color="#666")
    tag(ax, "a", x=-0.62)
    for i, (out, b, col) in enumerate([("Gallstones", "SSB", RED), ("Gallstones", "ASB", BLUE), ("AP", "SSB", RED),
                                       ("AP", "ASB", BLUE)]):
        a = fig.add_subplot(gs[i // 2, 1 + i % 2])
        s = sp[(sp.outcome == out) & (sp.exposure == b)]
        a.fill_between(s.servings, s.lci, s.uci, color=col, alpha=0.18, lw=0)
        a.plot(s.servings, s.hr, color=col, lw=1.1)
        a.axhline(1, color=GREY, lw=0.6)
        a.set_yscale("log")
        a.yaxis.set_major_locator(mt.FixedLocator([0.8, 1, 1.25, 1.5, 2]))
        a.yaxis.set_major_formatter(mt.FixedFormatter(["0.8", "1", "1.25", "1.5", "2"]))
        a.yaxis.set_minor_locator(mt.NullLocator())
        a.set_ylim(0.8, 2.1 if out == "AP" else 1.6)
        a.set_xlim(0, 4)
        a.set_xlabel(f"{b} (servings/day)")
        a.set_ylabel("HR (95% CI)")
        a.set_title(SHORT[out].replace(" (AP)", ""), fontsize=7)
        t = st.loc[(out, b)]
        a.text(0.04, 0.95, f"$P$ overall = {t.p_overall:.3f}\n$P$ nonlinear = {t.p_nonlinear:.2f}", transform=a.transAxes,
               va="top", fontsize=5.8)
        for sname in ("top", "right"):
            a.spines[sname].set_visible(False)
        tag(a, "bcde"[i], x=-0.25)
    save(fig, "Fig2_intake_outcomes")


# ---------------------------------------------------------------- Supplementary Fig 4 proteome
def figS4():
    a = pd.read_csv(RES / "protein_assoc.csv")
    a = a[a.model == "main"]
    e = pd.read_csv(RES / "enrichment_SSB_FDR.csv").sort_values("p_adj").drop_duplicates("term").head(12)
    fig = plt.figure(figsize=(7.2, 5.2))
    gs = fig.add_gridspec(1, 2, left=0.08, right=0.98, top=0.95, bottom=0.6, wspace=0.3)
    gs2 = fig.add_gridspec(1, 1, left=0.45, right=0.98, top=0.45, bottom=0.07)
    for i, b in enumerate(["SSB", "ASB"]):
        ax = fig.add_subplot(gs[0, i])
        r = a[a.exposure == b].copy()
        r["y"] = -np.log10(r.p.clip(lower=1e-300))
        sig = r.fdr < 0.05
        ax.scatter(r.beta[~sig], r.y[~sig], s=2.5, color=GREY, lw=0, rasterized=True)
        ax.scatter(r.beta[sig & (r.beta > 0)], r.y[sig & (r.beta > 0)], s=4, color=RED, lw=0, rasterized=True)
        ax.scatter(r.beta[sig & (r.beta < 0)], r.y[sig & (r.beta < 0)], s=4, color=BLUE, lw=0, rasterized=True)
        if sig.any():
            ax.axhline(r.loc[sig, "y"].min(), color="#888", lw=0.5, ls="--")
        top = r.sort_values("p").head(9)
        ymax = r.y.max()
        for side in (1, -1):                      # simple vertical de-overlap of labels, per side
            t = top[np.sign(top.beta) == side].sort_values("y", ascending=False)
            last = None
            for _, row in t.iterrows():
                ly = row.y if last is None else min(row.y, last - 0.055 * ymax)
                last = ly
                ax.annotate(row.protein.upper().split("_")[0], (row.beta, row.y), xytext=(row.beta + side * 0.006, ly),
                            textcoords="data", fontsize=5.5, va="center", ha="left" if side > 0 else "right",
                            arrowprops=dict(arrowstyle="-", lw=0.3, color="#777"))
        lim = r.beta.abs().max() * 1.45
        ax.set_xlim(-lim, lim)
        ax.xaxis.set_major_locator(mt.MaxNLocator(5, symmetric=True))
        ax.set_xlabel(f"β per serving/day of {b}")
        ax.set_ylabel("−log$_{10}$ $P$")
        ax.set_title(f"{b}: {int(sig.sum())} proteins at FDR < 0.05", fontsize=7)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        tag(ax, "ab"[i], x=-0.12)
    ax = fig.add_subplot(gs2[0, 0])
    y = np.arange(len(e))[::-1]
    ax.barh(y, -np.log10(e.p_adj), color="#d98880", height=0.65)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{t} ({src})" for t, src in zip(e.term, e.source)], fontsize=6)
    ax.set_xlabel("−log$_{10}$ FDR-adjusted $P$ (g:Profiler; background = 2,919 measured proteins)")
    ax.set_title(f"Pathways enriched among the {int((a[(a.exposure == 'SSB')].fdr < 0.05).sum())} SSB-related proteins",
                 fontsize=7)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    tag(ax, "c", x=-0.78, y=1.0)
    save(fig, "FigS4_beverage_proteome")


# ---------------------------------------------------------------- Fig 3 signatures
def fig3():
    m = pd.read_csv(RES / "signature_main.csv")
    m = m[m.analysis == "per SD"]
    rows = SPECTRUM + CONTROLS
    ev = m[(m.model == "M2") & (m.signature == "sig_SSB")].set_index("outcome")["events"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, s, t in zip(axes, ["sig_SSB", "sig_ASB"], "ab"):
        series = []
        for mod, off in [("M1", 0.24), ("M2", 0), ("M3", -0.24)]:
            g = m[(m.signature == s) & (m.model == mod)].set_index("outcome")
            series.append((mod, MODEL_COL[mod], off, {r: (g.at[r, "hr"], g.at[r, "lci"], g.at[r, "uci"]) for r in rows}))
        labels = [SHORT[r] + f" ({int(ev[r]):,})" for r in rows]
        forest_rows(ax, rows, labels, series, 0.6, 3.5, [0.7, 1, 1.5, 2, 3], "HR per SD (95% CI)",
                    sep_after=(len(SPECTRUM) - 1, len(SPECTRUM) + 2))
        ax.set_title(f"{s.split('_')[1]} proteomic signature", fontsize=7.5, fontweight="bold")
        tag(ax, t, x=-0.02 if t == "b" else -0.6)
    axes[1].legend(frameon=False, loc="lower right", fontsize=6.3,
                   title="M1 covariates; M2 + biomarkers;\nM3 + BMI signature", title_fontsize=5.8)
    save(fig, "Fig3_signature_outcomes")


# ---------------------------------------------------------------- Fig 4 pathway split
def fig4():
    m = pd.read_csv(RES / "signature_main.csv")
    mm = m[(m.analysis == "per SD, mutually adjusted") & (m.model == "M2")]
    het = pd.read_csv(RES / "heterogeneity_ap.csv")
    het = het[het.analysis == "signature per SD, mutually adjusted (M2)"].set_index("exposure")["p_heterogeneity"]
    rows = ["Gallstones", "Cholecystitis", "AP_biliary", "AP_nonbiliary", "CP"]
    fig = plt.figure(figsize=(7.2, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.55)
    ax = fig.add_subplot(gs[0, 0])
    series = []
    for s, col, off in [("sig_SSB", RED, 0.15), ("sig_ASB", BLUE, -0.15)]:
        g = mm[mm.signature == s].set_index("outcome")
        series.append((s.split("_")[1] + " signature", col, off, {r: (g.at[r, "hr"], g.at[r, "lci"], g.at[r, "uci"]) for r in rows}))
    forest_rows(ax, rows, [SHORT[r] for r in rows], series, 0.6, 2.6, [0.7, 1, 1.5, 2, 2.5],
                "HR per SD, mutually adjusted (95% CI)")
    ax.legend(frameon=False, loc="upper right", fontsize=6.2)
    ax.text(-0.45, -0.2, f"Gallstone-related vs non-gallstone AP: $P$ for heterogeneity = {het['sig_SSB']:.3f} (SSB), "
            f"{het['sig_ASB']:.2f} (ASB)", transform=ax.transAxes, fontsize=5.6)
    tag(ax, "a", x=-0.55)
    # heatmap of individual signature proteins
    pc = pd.read_csv(RES / "signature_protein_cox.csv")
    outs = ["Gallstones", "AP_biliary", "AP_nonbiliary", "CP"]
    best = pc[pc.outcome.isin(outs)].sort_values("p").groupby("outcome").head(6).protein.unique()[:22]
    H = pc[pc.protein.isin(best)].pivot(index="protein", columns="outcome", values="hr")[outs]
    F = pc[pc.protein.isin(best)].pivot(index="protein", columns="outcome", values="fdr")[outs]
    W = pc.drop_duplicates("protein").set_index("protein").loc[H.index, ["weight_SSB", "weight_ASB"]]
    order = np.argsort(-(np.log(H)).abs().max(axis=1).to_numpy() * np.sign(np.log(H["Gallstones"]).to_numpy() + 1e-9))
    H, F, W = H.iloc[order], F.iloc[order], W.iloc[order]
    ax2 = fig.add_subplot(gs[0, 1])
    cmap = LinearSegmentedColormap.from_list("bwr2", [BLUE, "#ffffff", RED])
    im = ax2.imshow(np.log(H.to_numpy()), cmap=cmap, norm=TwoSlopeNorm(0, -0.45, 0.45), aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            f = F.iat[i, j]
            mark = "**" if f < 0.01 else ("*" if f < 0.05 else "")
            ax2.text(j, i, f"{H.iat[i, j]:.2f}{mark}", ha="center", va="center", fontsize=5)
    ax2.set_xticks(range(len(outs)))
    ax2.set_xticklabels(["Gallstone\ndisease", "Gallstone-\nrelated AP", "Non-gallstone\nAP", "Chronic\npancreatitis"],
                        fontsize=6)
    src = ["SSB" if pd.notna(a) and pd.isna(b) else "ASB" if pd.isna(a) else "both" for a, b in W.to_numpy()]
    ax2.set_yticks(range(H.shape[0]))
    ax2.set_yticklabels([f"{p.upper().split('_')[0]} ({s})" for p, s in zip(H.index, src)], fontsize=5.8)
    ax2.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02, ticks=np.log([0.7, 0.85, 1, 1.2, 1.5]))
    cb.ax.set_yticklabels(["0.7", "0.85", "1", "1.2", "1.5"], fontsize=5.5)
    cb.set_label("HR per SD (M1)", fontsize=6)
    ax2.set_title("Individual signature proteins (* FDR < 0.05, ** < 0.01)", fontsize=6.5)
    tag(ax2, "b", x=-0.35)
    save(fig, "Fig4_pathway_split")


# ---------------------------------------------------------------- Fig 5 abdominal MRI
def fig5():
    ri = pd.read_csv(RES / "mri_intake.csv")
    rs = pd.read_csv(RES / "mri_signature.csv")
    phen = ["Pancreas PDFF", "Liver PDFF", "Visceral fat volume", "Abdominal subcutaneous fat volume", "Pancreas volume"]
    lab = {"Pancreas PDFF": "Pancreas fat (PDFF)", "Liver PDFF": "Liver fat (PDFF)", "Visceral fat volume": "Visceral fat",
           "Abdominal subcutaneous fat volume": "Abdominal subcutaneous fat", "Pancreas volume": "Pancreas volume"}
    n_int = ri[(ri["sample"] == "All participants")].groupby("phenotype")["n"].first()
    n_sig = rs[(rs["sample"] == "Independent sample")].groupby("phenotype")["n"].first()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True, gridspec_kw={"wspace": 0.08})
    y = np.arange(len(phen))[::-1].astype(float)

    def draw(ax, est, specs, xlabel, title, nmap):
        for (key, col, off, filled, name) in specs:
            for yi, ph in zip(y, phen):
                h, l, u = est(key, ph)
                ax.plot([l, u], [yi + off] * 2, color=col, lw=0.9)
                ax.plot(h, yi + off, "o" if filled else "o", ms=3.6, color=col, mfc=col if filled else "white", mew=0.9)
            ax.plot([], [], "o-", color=col, ms=3.6, mfc=col if filled else "white", lw=0.9, label=name)
        ax.axvline(0, color=GREY, lw=0.6, zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{lab[p]}\n(n = {int(n_int[p]):,} / {int(n_sig[p]):,})" for p in phen], fontsize=6.5)
        ax.tick_params(axis="y", length=0)
        for sp_ in ("top", "right", "left"):
            ax.spines[sp_].set_visible(False)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=7.5, fontweight="bold")
        ax.legend(frameon=False, fontsize=6, loc="lower right")

    def e_int(key, ph):
        b, m = key
        r = ri[(ri["sample"] == "All participants") & (ri.model == m) & (ri.exposure == b) & (ri.phenotype == ph)].iloc[0]
        return r.sd_estimate, r.sd_lci, r.sd_uci

    def e_sig(key, ph):
        s, mutual = key
        r = rs[(rs["sample"] == "Independent sample") & (rs.model == "A: Model 2 (baseline BMI)") & (rs.signature == s)
               & (rs.mutual == mutual) & (rs.phenotype == ph)].iloc[0]
        return r.sd_estimate, r.sd_lci, r.sd_uci

    draw(axes[0], e_int, [(("SSB", "A: Model 2 (baseline BMI)"), RED, 0.24, True, "SSB, BMI-adjusted"),
                          (("SSB", "C: without BMI"), RED, 0.08, False, "SSB, not BMI-adjusted"),
                          (("ASB", "A: Model 2 (baseline BMI)"), BLUE, -0.08, True, "ASB, BMI-adjusted"),
                          (("ASB", "C: without BMI"), BLUE, -0.24, False, "ASB, not BMI-adjusted")],
         "SD difference per serving/day (95% CI)", "Beverage intake", n_int)
    draw(axes[1], e_sig, [(("sig_SSB", False), RED, 0.15, True, "SSB signature"),
                          (("sig_SSB", True), RED, 0.05, False, "SSB signature, mutually adjusted"),
                          (("sig_ASB", False), BLUE, -0.05, True, "ASB signature"),
                          (("sig_ASB", True), BLUE, -0.15, False, "ASB signature, mutually adjusted")],
         "SD difference per SD of signature (95% CI)", "Proteomic signatures (independent sample)", n_sig)
    tag(axes[0], "a", x=-0.5)
    tag(axes[1], "b", x=-0.02)
    fig.text(0.02, -0.02, "n = participants in the intake analysis / in the independent signature sample", fontsize=6)
    save(fig, "Fig5_abdominal_mri")


# ---------------------------------------------------------------- supplementary figures
def figS2():
    s = pd.read_csv(RES / "intake_substitution.csv")
    outs = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "CP"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True, gridspec_kw={"wspace": 0.1})
    cols = {"water": "#2c6fbb", "tea": "#2e8b57", "coffee": "#8b5a2b", "juice": "#e08214", "ASB": "#7b3294"}
    for ax, src, t in zip(axes, ["SSB", "ASB"], "ab"):
        alts = [a for a in ["ASB", "water", "tea", "coffee", "juice"] if a != src]
        series = []
        for k, alt in enumerate(alts):
            g = s[(s["replace"] == src) & (s["with"] == alt)].set_index("outcome")
            series.append((alt if alt != "juice" else "fruit juice", cols[alt], 0.3 - 0.15 * k,
                           {r: (g.at[r, "hr"], g.at[r, "lci"], g.at[r, "uci"]) for r in outs}))
        forest_rows(ax, outs, [SHORT[r] for r in outs], series, 0.55, 1.35, [0.6, 0.8, 1, 1.2], "HR (95% CI)")
        ax.set_title(f"Replacing 1 serving/day of {src} with…", fontsize=7)
        ax.legend(frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=5,
                  handlelength=1.4, columnspacing=1.0)
        tag(ax, t, x=-0.02 if t == "b" else -0.5)
    save(fig, "FigS2_substitution")


def figS1():
    g = pd.read_csv(RES / "intake_subgroups.csv")
    LEVELS = {"Sex": ["Men", "Women"], "Age": ["<60 years", ">=60 years"], "BMI": ["<25", "25 to <30", ">=30"],
              "Alcohol": ["Non-regular drinker", "Regular drinker"], "Smoking": ["Never", "Ever"], "Diabetes": ["No", "Yes"]}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0), sharey=False, gridspec_kw={"wspace": 0.95})
    for ax, out, t in zip(axes, ["Gallstones", "AP"], "ab"):
        d = g[g.outcome == out]
        pint = d.groupby(["subgroup", "exposure"])["p_interaction"].first().unstack()
        keys, labels = [], []
        for sg, lvls in LEVELS.items():
            keys.append((sg, None))
            labels.append(f"{sg} ($P$ int. {pint.at[sg, 'SSB']:.2f} / {pint.at[sg, 'ASB']:.2f})")
            for lv in lvls:
                keys.append((sg, lv))
                n = int(d[(d.subgroup == sg) & (d.level == lv)]["events"].iloc[0])
                labels.append(f"    {lv.replace('>=', '≥')} ({n:,})")
        series = []
        for b, col, off in [("SSB", RED, 0.15), ("ASB", BLUE, -0.15)]:
            e = d[d.exposure == b].set_index(["subgroup", "level"])
            series.append((b, col, off, {k: (e.at[k, "hr"], e.at[k, "lci"], e.at[k, "uci"]) for k in keys if k[1]}))
        forest_rows(ax, keys, labels, series, 0.55, 1.8, [0.6, 0.8, 1, 1.4, 1.8], "HR per serving/day (95% CI)")
        for lab in ax.get_yticklabels():
            if not lab.get_text().startswith("    "):
                lab.set_fontweight("bold")
        ax.set_title(SHORT[out] + " (events)", fontsize=7.5)
        ax.legend(frameon=False, fontsize=6, loc="upper right", bbox_to_anchor=(1.02, 1.0))
        tag(ax, t, x=-0.9)
    fig.text(0.5, 0.0, "$P$ int. = $P$ for interaction (SSB / ASB), likelihood-ratio test", ha="center", fontsize=6)
    save(fig, "FigS1_subgroups")


def figS5():
    t = pd.read_csv(RES / "signature_tertiles.csv")
    t = t[t.model == "M2"]
    outs = ["Gallstones", "AP", "AP_biliary", "AP_nonbiliary", "CP"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, s, tg in zip(axes, ["sig_SSB", "sig_ASB"], "ab"):
        series = []
        for tert, col, off in [("T2", "#f4a582", 0.13), ("T3", RED if s == "sig_SSB" else BLUE, -0.13)]:
            g = t[(t.signature == s) & (t.tertile == tert)].set_index("outcome")
            series.append((f"{tert} vs T1", col, off, {r: (g.at[r, "hr"], g.at[r, "lci"], g.at[r, "uci"]) for r in outs}))
        ptr = t[(t.signature == s)].groupby("outcome")["p_trend"].first()
        labels = [f"{SHORT[r]} ($P$ trend {ptr[r]:.3f})" for r in outs]
        forest_rows(ax, outs, labels, series, 0.5, 4, [0.5, 1, 2, 4], "HR (95% CI), M2")
        ax.set_title(f"{s.split('_')[1]} signature tertiles", fontsize=7)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
        tag(ax, tg, x=-0.02 if tg == "b" else -0.7)
    save(fig, "FigS5_signature_tertiles")


def figS3():
    b = pd.read_csv(RES / "biomarkers.csv")
    b = b[b.adjustment == "with BMI"]
    marks = b.biomarker.unique().tolist()
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    y = np.arange(len(marks))[::-1]
    for ex, col, off in [("SSB", RED, 0.15), ("ASB", BLUE, -0.15)]:
        g = b[b.exposure == ex].set_index("biomarker").loc[marks]
        ax.errorbar(g.pct, y + off, xerr=[g.pct - g.lci, g.uci - g.pct], fmt="s", ms=3, color=col, lw=0.9, label=ex)
    ax.axvline(0, color=GREY, lw=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(marks)
    ax.set_xlabel("% difference per serving/day (95% CI)")
    ax.legend(frameon=False, fontsize=6)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "FigS3_biomarkers")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4, fig5, figS1, figS2, figS3, figS4, figS5):
        f()
        print("done", f.__name__, flush=True)
