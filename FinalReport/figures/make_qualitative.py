"""Build qualitative figure (vector PDF) from real Beans benchmark predictions."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path("/Users/apple/DHI/products/prompt2model")
EX_DIR = ROOT / "data/report_eval/beans_benchmark/examples"
META = json.loads((ROOT / "data/report_eval/beans_benchmark/metrics.json").read_text())
examples = META["examples"]

PRETTY = {
    "angular_leaf_spot": "angular leaf spot",
    "bean_rust":         "bean rust",
    "healthy":           "healthy",
}

def low_light(arr, factor=0.45):
    return np.clip(arr.astype(np.float32) * factor, 0, 255).astype(np.uint8)

n = len(examples)  # 4
fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.4), dpi=200)
for j, ex in enumerate(examples):
    p = EX_DIR / Path(ex["image_path"]).name
    img = mpimg.imread(p)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    axes[0, j].imshow(img); axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
    axes[0, j].set_title(f"clean (target: {PRETTY[ex['target']]})", fontsize=8)
    axes[1, j].imshow(low_light(img))
    axes[1, j].set_xticks([]); axes[1, j].set_yticks([])
    base = PRETTY[ex["baseline"]]
    guid = PRETTY[ex["guided"]]
    base_ok = ex["baseline"] == ex["target"]
    guid_ok = ex["guided"]   == ex["target"]
    axes[1, j].set_title(
        f"low-light\nFixed: {base} {'✓' if base_ok else '✗'}\nGuided: {guid} {'✓' if guid_ok else '✓'}",
        fontsize=7,
    )
for ax_row, label in zip(axes, ["clean test image", "low-light + predictions"]):
    ax_row[0].set_ylabel(label, fontsize=8)
plt.suptitle(
    "Beans qualitative comparison - Fixed (ImageNet preprocessing) vs Guided (low-light recipe)",
    fontsize=9,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out = ROOT / "FinalReport/figures/qualitative_beans.pdf"
plt.savefig(out, format="pdf", bbox_inches="tight")
print("wrote", out)
