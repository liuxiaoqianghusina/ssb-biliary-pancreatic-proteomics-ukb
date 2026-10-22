"""Food & Function table-of-contents graphic: 8 x 4 cm, 600 dpi RGB TIFF + PNG (layout follows the omega-3 project's F02_toc.py)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image

from common import PROJ

OUT = PROJ / "manuscript_FoodFunction" / "figures"
plt.rcParams.update({"font.family": "Arial", "font.size": 6})
fig = plt.figure(figsize=(8 / 2.54, 4 / 2.54)); ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 80); ax.set_ylim(0, 40); ax.axis("off")


def box(x, y, w, h, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec="#4a4a4a", lw=0.5))


def arrow(x1, x2, y):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle="-|>", mutation_scale=7, lw=0.8, color="#4a4a4a"))


box(1.5, 3, 17, 34, "#fdecea")
ax.text(10, 34, "Beverages", ha="center", va="top", fontsize=7, fontweight="bold")
ax.text(10, 26.5, "Sugar-\nsweetened", ha="center", va="top", fontsize=6.0, color="#c0392b", fontweight="bold", linespacing=1.2)
ax.text(10, 17.5, "Artificially\nsweetened", ha="center", va="top", fontsize=6.0, color="#2c6fbb", fontweight="bold", linespacing=1.2)
ax.text(10, 8.0, "n = 193,356", ha="center", va="top", fontsize=5.4, color="#555555")
arrow(19.3, 23.2, 20)
box(24, 3, 21, 34, "#eef6ec")
ax.text(34.5, 34, "Biology", ha="center", va="top", fontsize=7, fontweight="bold")
ax.text(34.5, 26.5, "2,919 plasma\nproteins\n↑ pancreatic fat\n↑ liver fat (MRI)", ha="center", va="top", fontsize=5.6, linespacing=1.45)
ax.text(34.5, 8.0, "Signature proteins:\nmarkers, not causal", ha="center", va="top", fontsize=4.9, color="#555555",
        style="italic", linespacing=1.2)
arrow(45.8, 49.7, 20)
box(50.5, 3, 28, 34, "#fbe5d6")
ax.text(64.5, 34, "SSB, HR per serving", ha="center", va="top", fontsize=6.6, fontweight="bold")
for i, (t, v) in enumerate([("Gallstone disease", "1.07"), ("Acute pancreatitis", "1.13"), ("Chronic pancreatitis", "1.18")]):
    ax.text(52, 26 - i * 5.8, t, ha="left", va="center", fontsize=5.6)
    ax.text(77.5, 26 - i * 5.8, v, ha="right", va="center", fontsize=5.6, fontweight="bold", color="#c0392b")
ax.text(64.5, 6.5, "Robust to reverse causation;\nASB associations attenuated", ha="center", va="center", fontsize=4.9,
        color="#555555", style="italic", linespacing=1.2)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "TOC_graphic.png", dpi=600)
fig.savefig(OUT / "TOC_graphic.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
Image.open(OUT / "TOC_graphic.tiff").convert("RGB").save(OUT / "TOC_graphic.tiff", compression="tiff_lzw", dpi=(600, 600))
print("TOC saved")
