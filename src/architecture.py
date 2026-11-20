from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def make_architecture(path: Path):
    fig, ax=plt.subplots(figsize=(12,3.2)); ax.set_xlim(0,12); ax.set_ylim(0,3); ax.axis("off")
    boxes=[(0.2,"Synthetic CSV\n+ user car inputs"),(2.65,"Validation +\nfeature engineering"),(5.1,"Imputation +\none-hot pipeline"),(7.55,"Regression models\nRF / DT / Extra Trees"),(10.0,"Capped service\nestimate + Streamlit")]
    colors=["#e8f1f4","#d9ebe7","#e9e6f6","#f5ead6","#d8eee6"]
    for i,(x,label) in enumerate(boxes):
        box=FancyBboxPatch((x,.95),1.8,1.05,boxstyle="round,pad=0.12,rounding_size=.12",fc=colors[i],ec="#49706e",lw=1.4)
        ax.add_patch(box); ax.text(x+.9,1.48,label,ha="center",va="center",fontsize=10,color="#1b373b",weight="bold")
        if i<len(boxes)-1: ax.add_patch(FancyArrowPatch((x+1.88,1.48),(x+2.35,1.48),arrowstyle="-|>",mutation_scale=14,color="#3d7068",lw=1.5))
    ax.text(6,.35,"Train / validation / test split → train-only preprocessing and CV → compare held-out regression metrics → save joblib pipeline",ha="center",fontsize=9,color="#546970")
    path.parent.mkdir(parents=True,exist_ok=True); fig.savefig(path,dpi=160,bbox_inches="tight",facecolor="white"); plt.close(fig)

if __name__=="__main__":
    make_architecture(Path(__file__).resolve().parents[1]/"reports"/"eda"/"system_architecture.png")
