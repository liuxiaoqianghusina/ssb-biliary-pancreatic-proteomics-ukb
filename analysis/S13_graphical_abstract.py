"""Graphical abstract for Clinical Nutrition (min 531 x 1328 px, h x w; readable at 5 x 13 cm).
Drawn at 13.28 x 5.31 in and saved at 300 dpi (3984 x 1593 px) as TIFF/PDF/PNG/SVG. Arial throughout."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Circle, PathPatch
from matplotlib.path import Path

from common import FIG

plt.rcParams.update({"font.family": "Arial", "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "svg.fonttype": "none", "pdf.fonttype": 42})
RED, BLUE, GREY, DARK, GREEN = "#c0392b", "#2c6fbb", "#9e9e9e", "#222222", "#3c8d5a"

fig = plt.figure(figsize=(13.28, 5.31))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 132.8); ax.set_ylim(0, 53.1); ax.axis("off")


def box(x, y, w, h, fc, ec="#555555", lw=1.2, r=1.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.3,rounding_size={r}", fc=fc, ec=ec, lw=lw))


def can(x, y, color, label, s=1.0):
    """Simple beverage can icon with label underneath."""
    w, h = 4.2 * s, 7.2 * s
    ax.add_patch(FancyBboxPatch((x - w / 2, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.8", fc=color, ec="none"))
    ax.add_patch(Ellipse((x, y + h), w, 1.2 * s, fc="#dddddd", ec=color, lw=1))
    ax.add_patch(FancyBboxPatch((x - w / 2 + 0.4, y + h * 0.35), w - 0.8, h * 0.28, boxstyle="round,pad=0,rounding_size=0.3",
                                fc="white", ec="none", alpha=0.85))
    ax.text(x, y + h * 0.49, label, ha="center", va="center", fontsize=11 * s, fontweight="bold", color=color)


def arrow(x1, y1, x2, y2, color=DARK, lw=2.2, style="-|>", ms=18, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=ms, lw=lw, color=color, linestyle=ls))


# ---------------------------------------------------------------- title banner
ax.text(66.4, 50.6, "Sugar-sweetened beverages, gallstone disease and pancreatitis", ha="center", va="center", fontsize=21,
        fontweight="bold", color=DARK)

# ---------------------------------------------------------------- left: cohort
box(1.5, 9.5, 27, 36.5, "#f6f6f6")
ax.text(15, 42.6, "Prospective cohort", ha="center", fontsize=15, fontweight="bold")
ax.text(15, 37.4, "193,356 adults", ha="center", fontsize=19, fontweight="bold", color=DARK)
ax.text(15, 33.7, "24-h dietary recalls\n11.8-year follow-up", ha="center", va="top", fontsize=13, linespacing=1.3)
can(9, 17.0, RED, "SSB", 1.25)
can(21, 17.0, BLUE, "ASB", 1.25)
ax.text(15, 12.4, "Plasma proteome (2,919 proteins)\nAbdominal MRI organ fat\nMendelian randomization", ha="center",
        va="center", fontsize=10, color="#444444", linespacing=1.2)

arrow(29.6, 28, 33.6, 28)

# ---------------------------------------------------------------- middle: intake findings
box(34.6, 27.9, 46, 18.1, "#fdecea", ec=RED)
ax.text(57.6, 43.5, "Sugar-sweetened beverages (SSB)", ha="center", fontsize=14.5, fontweight="bold", color=RED)
ax.text(36.4, 40.0, "↑ Gallstone disease", fontsize=13.5, va="center")
ax.text(36.4, 36.9, "↑ Acute & chronic pancreatitis", fontsize=13.5, va="center")
ax.text(36.4, 33.8, "↑ Pancreatic, liver & visceral fat (MRI)", fontsize=13.5, va="center")
ax.text(36.4, 30.9, "Habitual intake: acute pancreatitis HR 1.86", fontsize=12, va="center", color="#555555")
ax.plot([74.6, 76.4, 79.6], [38.0, 35.8, 40.4], color=GREEN, lw=4.5, solid_capstyle="round")   # check mark
ax.text(57.6, 28.8, "Robust after excluding diet/weight change, diabetes, obesity", ha="center", fontsize=9.8,
        color=GREEN, style="italic")

box(34.6, 9.5, 46, 15.9, "#eaf1fa", ec=BLUE)
ax.text(57.6, 22.7, "Artificially sweetened beverages (ASB)", ha="center", fontsize=14.5, fontweight="bold", color=BLUE)
ax.text(36.4, 18.9, "Intake associations weaken after", fontsize=13, va="center")
ax.text(36.4, 16.0, "restriction → reverse causation", fontsize=13, va="center")
ax.text(36.4, 13.3, "Organ fat on MRI largely explained by BMI", fontsize=11.5, va="center", color="#555555")
ax.text(36.4, 11.0, "Proteomic signature still marks pancreatitis risk", fontsize=11.5, va="center", color="#555555")

arrow(81.6, 37.3, 86.3, 37.3, color=RED)
arrow(81.6, 17.5, 86.3, 17.5, color=BLUE)

# ---------------------------------------------------------------- right: organ schematic
box(87.3, 9.5, 44, 36.5, "white", ec="#555555")
ax.text(109.3, 42.6, "Proteomic signatures (independent sample)", ha="center", fontsize=12.5, fontweight="bold")
# liver (smooth outline; cubic Bezier segments)
M, C4, CL = Path.MOVETO, Path.CURVE4, Path.CLOSEPOLY


def bez(start, segs):
    """Closed path from a start point and a list of cubic segments ((c1, c2, end), ...)."""
    v, c = [start], [M]
    for c1, c2, e in segs:
        v += [c1, c2, e]; c += [C4, C4, C4]
    return Path(v + [start], c + [CL])


liver = bez((90.8, 32.2), [((92.0, 37.6), (100.0, 38.4), (105.5, 37.0)), ((108.6, 36.2), (110.0, 33.6), (108.4, 32.6)),
                           ((105.0, 31.2), (103.0, 30.8), (101.5, 30.6)), ((98.5, 30.2), (94.0, 28.8), (90.8, 32.2))])
ax.add_patch(PathPatch(liver, fc="#e8c8b8", ec="#a5735e", lw=1.3))
ax.text(98.6, 33.6, "Liver", ha="center", fontsize=10.5, color="#7a4b3a")
# gallbladder (pear shape) with stones
gb = bez((101.8, 30.4), [((99.4, 28.0), (100.0, 23.4), (103.6, 23.2)), ((107.4, 23.0), (107.2, 26.4), (105.6, 28.4)),
                         ((104.6, 29.8), (103.2, 30.8), (101.8, 30.4))])
ax.add_patch(PathPatch(gb, fc="#b9dcb0", ec="#4f8a45", lw=1.4))
for cx, cy in [(102.7, 25.6), (104.6, 25.0), (103.8, 27.0)]:
    ax.add_patch(Circle((cx, cy), 0.6, fc="#d9b44a", ec="#8c6d13", lw=0.6))
ax.text(98.6, 21.6, "Gallbladder", ha="center", fontsize=10.5, color="#3d6b35")
# common bile duct to the pancreatic head
ax.plot([106.6, 110.0, 113.0, 114.6], [25.6, 23.0, 20.6, 18.2], color="#4f8a45", lw=2.2, solid_capstyle="round")
# pancreas: head (left), body, tapering tail (right)
pan = bez((112.4, 17.0), [((111.6, 20.4), (114.6, 21.2), (117.2, 20.0)), ((121.0, 19.2), (125.0, 20.6), (129.4, 22.4)),
                          ((130.6, 21.6), (130.2, 20.2), (128.4, 19.4)), ((124.6, 17.8), (120.4, 17.6), (118.4, 16.6)),
                          ((117.2, 13.6), (113.6, 13.0), (112.4, 17.0))])
ax.add_patch(PathPatch(pan, fc="#f5c59a", ec="#c27a36", lw=1.4))
ax.text(123.5, 13.0, "Pancreas", ha="center", fontsize=10.5, color="#9a5a1f")
# signature annotations
ax.text(121.0, 34.6, "SSB signature", fontsize=12.5, fontweight="bold", color=RED, ha="center")
ax.text(121.0, 31.8, "gallstones,\ngallstone-related\npancreatitis", fontsize=10.5, color=RED, ha="center", va="top",
        linespacing=1.1)
arrow(114.6, 30.6, 107.6, 27.0, color=RED, lw=2.0, ms=15)
ax.text(96.5, 17.4, "ASB signature", fontsize=12.5, fontweight="bold", color=BLUE, ha="center")
ax.text(96.5, 15.0, "non-gallstone acute\n& chronic pancreatitis", fontsize=10.5, color=BLUE, ha="center", va="top",
        linespacing=1.1)
arrow(104.6, 15.0, 113.6, 16.0, color=BLUE, lw=2.0, ms=15)
ax.text(109.3, 10.4, "Signature proteins: markers, not causal (MR, colocalization)", ha="center", fontsize=9.2,
        color="#555555", style="italic")

# ---------------------------------------------------------------- bottom banner: take-home
box(1.5, 0.8, 129.8, 6.4, "#eef6ec", ec=GREEN, lw=1.5)
ax.text(66.4, 4.0, "Replace sugar-sweetened beverages with water, tea or coffee", ha="center", va="center", fontsize=16,
        fontweight="bold", color=GREEN)

for ext, kw in [("tiff", {"dpi": 300, "pil_kwargs": {"compression": "tiff_lzw"}}), ("pdf", {}), ("png", {"dpi": 300}),
                ("svg", {})]:
    fig.savefig(FIG / f"Graphical_abstract.{ext}", **kw)
print("saved", FIG / "Graphical_abstract.tiff")
